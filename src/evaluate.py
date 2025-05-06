"""
Evaluate a fine‑tuned BERT/DistilBERT classifier on RAID dataset splits.

This version assumes the **training script now copies the best checkpoint to**

    outputs/checkpoints/<run‑name>/best_model/

If that folder exists we load it directly; otherwise we fall back to the
checkpoint recorded in `trainer_state.json`, and finally to the first
sub‑folder that contains `pytorch_model.bin`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG_SEED = 42
set_seed(RNG_SEED)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_split(dataset_root: Path, split: str):
    """Return 🤗 *Dataset* stored under `<root>/<split>/`."""
    dpath = dataset_root / split
    return load_from_disk(dpath.as_posix()) if dpath.exists() else None


def has_weights(path: Path) -> bool:
    """Check if *path* contains model weight files (.bin or .safetensors)."""
    return any((path / fname).exists() for fname in ("pytorch_model.bin", "model.safetensors"))


def find_first_complete_ckpt(root: Path) -> Optional[Path]:
    """Return first sub‑dir in *root* that has weight files, or *None*."""
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and has_weights(sub):
            return sub
    return None


def best_checkpoint(ckpt_root: Path) -> Path:
    """Resolve the correct checkpoint directory to load."""
    # 1) If the training script copied best_model/, use it
    best_model_dir = ckpt_root / "best_model"
    if best_model_dir.exists() and has_weights(best_model_dir):
        return best_model_dir

    # 2) Otherwise inspect trainer_state.json
    state_file = ckpt_root / "trainer_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        best = state.get("best_model_checkpoint")
        if best and has_weights(Path(best)):
            return Path(best)

    # 3) Fallback: first sub‑folder with weights
    fallback = find_first_complete_ckpt(ckpt_root)
    if fallback is not None:
        return fallback

    raise FileNotFoundError(
        "Could not locate a checkpoint with model weights under " f"{ckpt_root}"
    )


def compute_metrics(labels: np.ndarray | None, preds: np.ndarray) -> Dict[str, float]:
    """Return accuracy, precision, recall, F1 for binary classification."""
    if labels is None:
        return {}

    acc = float((preds == labels).mean())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_split(split_name: str, ds, trainer: Trainer, run_name: str, save_preds: bool):
    """Predict on a dataset split, return metrics, optionally save CSV."""
    logits = trainer.predict(ds).predictions
    preds = np.argmax(logits, axis=-1)

    labels = None
    if "labels" in ds.column_names:
        labels = np.array(ds["labels"], dtype=int)

    metrics = compute_metrics(labels, preds)

    if save_preds:
        out_dir = Path("outputs/predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{run_name}_{split_name}_predictions.csv"
        with csv_path.open("w+", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["index", "prediction"]
            if labels is not None:
                header.append("label")
            if 'id' in ds.column_names:
                header.append("id")
            if "text" in ds.column_names:
                header.append("text")
            writer.writerow(header)
            if labels is not None:
                for idx, p in enumerate(preds):
                    row = [idx, int(p)]
                    if labels is not None and int(labels[idx])!=p:
                        row.append(int(labels[idx]))
                        row.append(ds["id"][idx])
                        if "text" in ds.column_names:
                            row.append(ds["text"][idx].replace("\n", " ").strip())
                        writer.writerow(row)
        print("Predictions saved to", csv_path)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    dataset_root = Path(args.dataset_root).resolve()
    ckpt_root = Path(args.checkpoints_dir).resolve() if args.checkpoints_dir else Path("outputs/checkpoints") / args.run_name

    assert dataset_root.exists(), f"Dataset root not found: {dataset_root}"
    assert ckpt_root.exists(), f"Checkpoints root not found: {ckpt_root}"

    ckpt_path = best_checkpoint(ckpt_root)
    print("Loading checkpoint from", ckpt_path)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)

    collator = DataCollatorWithPadding(tokenizer)
    eval_args = TrainingArguments(
        output_dir="outputs/eval_runs",
        per_device_eval_batch_size=args.batch_size,
        dataloader_drop_last=False,
        report_to="none",
        seed=RNG_SEED,
    )

    trainer = Trainer(model=model, args=eval_args, tokenizer=tokenizer, data_collator=collator)

    # load dataset splits
    split_names = [s.strip() for s in args.split.split(",") if s.strip()]
    datasets = {s: load_split(dataset_root, s) for s in split_names}
    for s, ds in datasets.items():
        assert ds is not None, f"No '{s}' split under {dataset_root}"

    all_metrics: Dict[str, Dict] = {}
    for split_name, ds in datasets.items():
        print(f"Evaluating {split_name} ({len(ds):,} rows)")
        m = evaluate_split(split_name, ds, trainer, args.run_name, args.save_preds)
        if m:
            all_metrics[split_name] = {k: float(v) for k, v in m.items()}
        else:
            all_metrics[split_name] = "no_labels"

    # save metrics
    out_dir = Path("outputs/metrics"); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{args.run_name}_eval_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print("Metrics written to", metrics_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser("Evaluate a fine‑tuned BERT/DistilBERT classifier")
    p.add_argument("--dataset-root", required=True, help="Folder with train/val/test splits")
    p.add_argument("--run-name", required=True, help="Run‑name used during training")
    p.add_argument("--checkpoints-dir", help="Override path to checkpoints root")
    p.add_argument("--split", default="test", help="Comma‑separated list of splits (default: test)")
    p.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    p.add_argument("--save-preds", action="store_true", help="Save per‑example predictions to CSV")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
