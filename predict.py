import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nlp.model import select_device


def predict(text: str, model_path: Path, device_name: str = "auto") -> dict:
    if not text.strip():
        raise ValueError("Text cannot be empty")
    if not model_path.is_dir():
        raise ValueError("Checkpoint not found; run train.py first")
    device = select_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).to(device)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.inference_mode():
        probabilities = model(**inputs).logits.softmax(dim=-1)[0]
    index = probabilities.argmax().item()
    return {
        "label": model.config.id2label[index],
        "score": round(probabilities[index].item(), 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify text with a local checkpoint."
    )
    parser.add_argument("text")
    parser.add_argument("--model", type=Path, default=Path("outputs/bert-sentiment"))
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    args = parser.parse_args()
    print(json.dumps(predict(args.text, args.model, args.device)))
