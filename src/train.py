# src/train.py
"""
Train a BERT detector on tokenised RAID splits produced by data_preprocessing.py
---------------------------------------------------------------------------
Example (local debug run):
    python -m src.train \
        --dataset-root data/processed/raid_local \
        --model-name distilbert-base-uncased \
        --epochs 2 --run-name debug_run

Example (full production):
    python -m src.train --dataset-root data/processed/raid_full --epochs 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import tensorflow as tf
from transformers import create_optimizer

from src.model import get_model

RNG_SEED = 42
tf.keras.utils.set_random_seed(RNG_SEED)


def load_split(dataset_root: Path, split: str) -> Optional[tf.data.Dataset]:
    """Load a cached tf.data.Dataset if it exists, otherwise None."""
    tf_path = dataset_root / split / "tf_dataset"
    return tf.data.Dataset.load(tf_path.as_posix()) if tf_path.exists() else None


def main(args: argparse.Namespace) -> None:
    root = Path(args.dataset_root).resolve()
    train_ds = load_split(root, "train")
    val_ds = load_split(root, "val")          # may be None
    val_ood = load_split(root, "val_ood")     # optional extra split
    if val_ds is None and val_ood is not None:
        val_ds = val_ood                      # fall back to OOD set

    assert train_ds is not None, f"No train split found under {root}"

    # ── model & optimiser ───────────────────────────────────────────────────
    model = get_model(args.model_name)
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    optimiser, _ = create_optimizer(
        init_lr=args.learning_rate,
        num_warmup_steps=0,
        num_train_steps=steps_per_epoch * args.epochs,
    )
    model.compile(
        optimizer=optimiser,
        loss=model.compute_loss,
        metrics=["accuracy"],
    )

    # ── callbacks ───────────────────────────────────────────────────────────
    callbacks = []
    if val_ds is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=2,
                restore_best_weights=True,
            )
        )

    # ── train ───────────────────────────────────────────────────────────────
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    # ── save artefacts ──────────────────────────────────────────────────────
    ckpt_dir = Path("outputs/checkpoints") / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir.as_posix())

    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{args.run_name}_history.json").write_text(
        json.dumps(history.history, indent=2)
    )

    print(f"✅ Model & history saved under outputs/ with run‑name '{args.run_name}'")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-root",
        required=True,
        help="Folder created by data_preprocessing.py (contains train/ val/ ...)",
    )
    p.add_argument("--model-name", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)  
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--run-name", default="run_001",
                   help="Folder name for checkpoints/metrics")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
