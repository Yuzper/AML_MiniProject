"""
Train a BERT/DistilBERT detector with PyTorch + Trainer
=========================================================

The script expects splits produced by *data_preprocessing.py*, saved with
`datasets.Dataset.save_to_disk(<root>/<split>/)`.

─────────────────────────────────────────────────────────────────────────────
Example (local debug run):
    python -m src.train_torch \
        --dataset-root data/processed/raid_local \
        --model-name distilbert-base-uncased \
        --epochs 2 \
        --batch-size 4 \
        --run-name distil_local

Example (full production):
    python -m src.train_torch \
        --dataset-root data/processed/raid_full \
        --model-name bert-base-uncased \
        --epochs 5 \
        --batch-size 8 \
        --learning-rate 2e-5 \
        --run-name bert_full
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import argparse, json, math, random, os
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from src.model import get_model

RNG_SEED = 42
set_seed(RNG_SEED)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def load_split(dataset_root: Path, split: str):
    """
    Load a 🤗 datasets arrow dataset if available, otherwise return None.
    Expect folders  <root>/<split>/  produced via  ds.save_to_disk().
    """
    path = dataset_root / split
    return load_from_disk(path.as_posix()) if path.exists() else None


def compute_metrics(eval_preds):
    """
    Accuracy for 2-class classification.
    `eval_preds` is (logits, labels) when using Trainer.
    """
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def any_ampere_plus() -> bool:
    """
    Check compute capability ≥ 7.0 for *any* visible GPU (purely cosmetic;
    Trainer will still enable fp16 if `fp16=True` in TrainingArguments).
    """
    if not torch.cuda.is_available():
        return False
    from torch.cuda import get_device_capability, device_count
    return any(get_device_capability(i)[0] >= 7 for i in range(torch.cuda.device_count()))


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace):
    root = Path(args.dataset_root).resolve()

    train_ds = load_split(root, "train")
    val_ds   = load_split(root, "val") or load_split(root, "val_ood")

    assert train_ds is not None, f"No train split found under {root}"
    print("🔍 First training example:", train_ds[0])

    # ------------------------------------------------------------------ model
    model = get_model(args.model_name)

    # ----------------------------------------------------------------- trainer
    training_args = TrainingArguments(
        output_dir=f"outputs/checkpoints/{args.run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch" if val_ds is not None else "no",
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),      # mixed precision on Ampere+
        seed=RNG_SEED,
        report_to="none",
    )

    callbacks = []
    if val_ds is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # --------------------------------------------------------------- train/run
    trainer.train()

    # ----------------------------------------------------------------- save
    ckpt_dir = Path(training_args.output_dir)
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # `trainer.state.log_history` is a list of dicts; keep the final metrics
    history = trainer.state.log_history
    (metrics_dir / f"{args.run_name}_history.json").write_text(
        json.dumps(history, indent=2)
    )
    print(f"✅ Model & history saved under 'outputs/' with run-name '{args.run_name}'")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True,
                   help="Folder created by data_preprocessing.py (contains train/ val/ …)")
    p.add_argument("--model-name", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--run-name", default="run_001",
                   help="Folder name for checkpoints/metrics")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())