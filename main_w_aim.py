import hydra
import torch
from aim import Run, Text
from datasets import load_dataset
from omegaconf import OmegaConf
from sacrebleu.metrics import BLEU
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim.lr_scheduler import StepLR


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids_padded = pad_sequence(
            [torch.tensor(seq) for seq in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels_padded = pad_sequence(
            [torch.tensor(seq) for seq in labels],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": input_ids_padded.ne(self.tokenizer.pad_token_id),
        }


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # alternatively resolve the config into a dict
    # cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    aim_run = Run(repo=cfg.aim.repo)
    aim_run["hparams"] = cfg

    device = "cuda" if cfg.training.device == "cuda" and torch.cuda.is_available() else "cpu"

    dataset = load_dataset(cfg.dataset.name, cfg.dataset.lang_pair, trust_remote_code=True)
    tokenizer = T5Tokenizer.from_pretrained(cfg.tokenizer.name, legacy=False)

    def preprocess_function(examples):
        inputs = [ex for ex in examples[cfg.dataset.input_key_name]]
        targets = [ex for ex in examples[cfg.dataset.output_key_name]]
        model_inputs = tokenizer(inputs, max_length=cfg.tokenizer.max_length, truncation=True)

        labels = tokenizer(
            text_target=targets,
            max_length=cfg.tokenizer.max_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    collator = Collator(tokenizer)
    train_dataloader = DataLoader(
        encoded_dataset["train"],
        batch_size=cfg.training.batch_size,
        collate_fn=collator,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        encoded_dataset["validation"],
        batch_size=cfg.training.batch_size,
        collate_fn=collator,
    )

    model = T5ForConditionalGeneration.from_pretrained(cfg.model.name)
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    scheduler = StepLR(optimizer, step_size=cfg.training.scheduler.step_size, gamma=cfg.training.scheduler.gamma)

    bleu = BLEU()

    def validate(model, tokenizer, dataloader, device, step):
        model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        all_inputs = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                loss = outputs.loss
                total_loss += loss.item()

                predictions = tokenizer.batch_decode(outputs.logits.argmax(-1), skip_special_tokens=True)
                references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                all_predictions.extend(predictions)
                all_references.extend([[ref] for ref in references])  # sacrebleu expects a list of lists for refs
                all_inputs.extend(inputs)

        for idx in range(len(all_inputs)):
            aim_run.track(Text(all_inputs[idx]), name="input", context={"idx": idx}, step=step)
            aim_run.track(Text(all_predictions[idx]), name="prediction", context={"idx": idx}, step=step)
            aim_run.track(Text(all_references[idx][0]), name="reference", context={"idx": idx}, step=step)

        avg_loss = total_loss / len(dataloader)
        bleu_score = bleu.corpus_score(all_predictions, all_references).score
        return avg_loss, bleu_score

    model.train()
    step = 0
    for epoch in range(cfg.training.num_epochs):  # number of epochs
        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % cfg.training.log_interval == 0:
                val_loss, val_bleu = validate(model, tokenizer, valid_dataloader, device, step)
                loop.set_postfix(loss=loss.item(), val_loss=val_loss, val_bleu=val_bleu)

                aim_run.track(loss.item(), name="loss", context={"subset": "train"}, step=step, epoch=epoch)
                aim_run.track(val_loss, name="loss", context={"subset": "val"}, step=step, epoch=epoch)
                aim_run.track(val_bleu, name="bleu", context={"subset": "val"}, step=step, epoch=epoch)
                aim_run.track(optimizer.param_groups[0]["lr"], name="lr", step=step, epoch=epoch)
                aim_run.track(optimizer.param_groups[0]["weight_decay"], name="weight_decay", step=step, epoch=epoch)

            step += 1

            loop.set_description(f"Epoch {epoch + 1}")

        scheduler.step()

    model.save_pretrained("./translation_model")


if __name__ == "__main__":
    main()
