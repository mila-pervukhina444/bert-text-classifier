import argparse
import json
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from nlp.data import TextDataset, read_records
from nlp.model import ID_TO_LABEL, evaluate, select_device


def train(config_path: Path):
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    root = config_path.resolve().parent
    for key in ("epochs", "batch_size", "max_length"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    output_dir = root / config["output_dir"]
    if output_dir.exists():
        raise ValueError("Output directory exists; choose another output_dir")

    train_records = read_records(root / config["train_file"])
    valid_records = read_records(root / config["valid_file"])
    train_texts = {row["text"].strip().casefold() for row in train_records}
    if any(row["text"].strip().casefold() in train_texts for row in valid_records):
        raise ValueError("Training and validation texts overlap")

    set_seed(config["seed"])
    device = select_device(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.model_max_length = config["max_length"]
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"],
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id={label: index for index, label in ID_TO_LABEL.items()},
    ).to(device)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loaders = [
        DataLoader(
            TextDataset(records, tokenizer, config["max_length"]),
            batch_size=config["batch_size"],
            shuffle=shuffle,
            collate_fn=collator,
        )
        for records, shuffle in ((train_records, True), (valid_records, False))
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    output_dir.mkdir(parents=True, exist_ok=False)
    best_accuracy = -1.0
    history = []
    print(f"Device: {device}; train: {len(train_records)}; valid: {len(valid_records)}")

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in loaders[0]:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch["labels"].size(0)

        metrics = evaluate(model, loaders[1], device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_records),
            "valid_loss": metrics["loss"],
            "valid_accuracy": metrics["accuracy"],
        }
        history.append(row)
        print(json.dumps(row))
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    (output_dir / "metrics.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Best checkpoint saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT on local text data.")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    train(parser.parse_args().config)
