import json
from pathlib import Path

from torch.utils.data import Dataset


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("text"), str)
                or not record["text"].strip()
                or type(record.get("label")) is not int
                or record["label"] not in (0, 1)
            ):
                raise ValueError(f"Invalid text or label at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


class TextDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        encoded = self.tokenizer(
            record["text"], truncation=True, max_length=self.max_length
        )
        encoded["labels"] = record["label"]
        return encoded
