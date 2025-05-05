"""
Train a BERT/DistilBERT detector with PyTorch + Trainer
======================================================

This script trains on pre‑processed RAID splits (see *data_preprocessing.py*)
and **always leaves a complete, self‑contained copy of the best model in**

    outputs/checkpoints/<run‑name>/best_model/

so that evaluation can reliably load it. The `best_model` folder contains
`pytorch_model.bin` (or `model.safetensors` if you enable it), `config.json`,
all tokenizer files, and `training_args.bin`.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.model import get_model

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG_SEED = 42
set_seed(RNG_SEED)
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(dataset_root: Path, split: str):
    """Return a 🤗 *Dataset* stored under `<root>/<split>/`, or *None*."""
    dpath = dataset_root / split
    return load_from_disk(dpath.as_posix()) if dpath.exists() else None


def compute_metrics(eval_preds):
    """Simple accuracy for 2‑class classification."""
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    root = Path(args.dataset_root).resolve()

    train_ds = load_split(root, "train")
    val_ds = load_split(root, "val") or load_split(root, "val_ood")
    assert train_ds is not None, f"No train split found under {root}"

    print("First training example:", train_ds[0])

    # ------------------------------ model/tokeniser
    model = get_model(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = DataCollatorWithPadding(tokenizer)

    # ------------------------------ training args
    ckpt_root = Path("outputs/checkpoints") / args.run_name
    training_args = TrainingArguments(
        output_dir=ckpt_root.as_posix(),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        seed=RNG_SEED,
        report_to="none",
        # --- saving / evaluation ------------------------------------
        evaluation_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch" if val_ds is not None else "no",
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,          # keep best + last
        save_safetensors=True,      # True if you prefer .safetensors
    )

    # ------------------------------ trainer
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if val_ds is not None else []
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ------------------------------ train
    trainer.train()

    # ------------------------------ always export best model
    best_ckpt_path = Path(trainer.state.best_model_checkpoint or training_args.output_dir)
    assert best_ckpt_path.exists(), "Best checkpoint folder not found!"

    best_model_dir = ckpt_root / "best_model"
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    shutil.copytree(best_ckpt_path, best_model_dir)
    print("Best model copied to", best_model_dir)

    # ------------------------------ metrics history
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{args.run_name}_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2)
    )
    print("Training run complete – outputs under 'outputs/'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True, help="Folder with train/val/test splits")
    p.add_argument("--model-name", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--run-name", default="run_001")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())