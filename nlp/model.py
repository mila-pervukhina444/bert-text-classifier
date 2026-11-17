import torch

ID_TO_LABEL = {0: "negative", 1: "positive"}


def select_device(name="auto"):
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        size = batch["labels"].size(0)
        total_loss += output.loss.item() * size
        correct += (output.logits.argmax(-1) == batch["labels"]).sum().item()
        count += size
    if not count:
        raise ValueError("Validation dataset is empty")
    return {"loss": total_loss / count, "accuracy": correct / count}
