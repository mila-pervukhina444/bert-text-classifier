# BERT text classifier

Fine-tune BERT on local text data and use the saved checkpoint for sentiment
classification. Built with PyTorch and Hugging Face Transformers.

## Setup

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Train

```bash
uv run train.py --config config.toml
```

Edit `config.toml` to change the model, data paths, batch size, and learning rate.
Paths in the config are relative to the config file. The first run downloads
`google-bert/bert-base-uncased`; training runs locally on CUDA, Apple MPS, or CPU.
Set `device = "cpu"` to force CPU execution.

The checkpoint with the highest validation accuracy is saved to
`outputs/bert-sentiment`, along with per-epoch metrics and the training config.
Choose a new `output_dir` for each run.

## Predict

```bash
uv run predict.py "A thoughtful movie with excellent acting."
```

The command returns a JSON object with `label` and `score`. Inference loads only
the local checkpoint and does not require network access.

## Data

Each JSONL record has `text` and `label`: `0` is negative, `1` is positive.
The included 12 training and 4 validation sentences are hand-written fixtures
for checking the workflow. Replace them with a larger, independently split
dataset before assessing model quality. Validation accuracy on these fixtures
is not a benchmark.

`nlp/data.py` handles data loading, `nlp/model.py` handles device selection and
evaluation. Model weights and generated outputs are excluded from Git.

Reference: [Transformers text classification](https://huggingface.co/docs/transformers/tasks/sequence_classification).

## License

MIT. See [LICENSE](LICENSE). Pretrained weights retain their original license.
