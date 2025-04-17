
"""Preprocessing pipeline for the RAID anti‑AI‑text dataset.

Features
--------
* Works in **two modes**
  * **local**  – small, stratified slice of the train set
  * **prod**   – full official splits (train / extra / raid_test)
* Produces ready‑to‑use HuggingFace Arrow datasets **and** cached `tf.data.Dataset`s
* Keeps class balance via stratified sampling when creating val & test
* Saves all artefacts under `data/processed/<run_name>/<split>/`

Example
-------
Local quick run:
```bash
python -m src.data_preprocessing --local --sample-size 3000
```

Full production preprocessing:
```bash
python -m src.data_preprocessing --run-name raid_full --prod
```
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import tensorflow as tf
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
tf.random.set_seed(RNG_SEED)

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_data(
    *,
    local: bool = True,
    sample_size: int = 3_000,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> DatasetDict:
    """Return a :class:`~datasets.DatasetDict` with *train*, *val*, *test*.

    Parameters
    ----------
    local
        If *True*, download only a slice of `raid/train` and create in‑domain
        validation and test splits.  If *False*, follow the official scheme
        (`raid/train`, `raid/extra`, `raid_test`).
    sample_size
        Number of rows to draw from `raid/train` when *local* is *True*.
    validation_fraction, test_fraction
        Fractions for the in‑domain val and test splits (ignored in prod mode).
    """
    if local:
        print("▶ Local mode – downloading a small slice of raid/train")
        ds = load_dataset("liamdugan/raid", "raid", split=f"train[:{sample_size}]")

        # ── create val / test partitions ────────────────────────────────────
        tmp = ds.train_test_split(
            test_size=test_fraction,
            seed=RNG_SEED,
            stratify_by_column="model",
        )
        train_val, test_ds = tmp["train"], tmp["test"]
        val_size = validation_fraction / (1.0 - test_fraction)
        tmp = train_val.train_test_split(
            test_size=val_size,
            seed=RNG_SEED,
            stratify_by_column="model",
        )
        return DatasetDict(train=tmp["train"], val=tmp["test"], test=test_ds)

    # ───────────────────────── Production mode ─────────────────────────────
    print("▶ Production mode – pulling full official splits")
    train_ds = load_dataset("liamdugan/raid", "raid", split="train")
    val_ds = load_dataset("liamdugan/raid", "raid", split="extra")  # labelled OOD
    test_ds = load_dataset("liamdugan/raid", "raid_test", split="train")  # no labels
    return DatasetDict(train=train_ds, val=val_ds, test=test_ds)


# ---------------------------------------------------------------------------
# Row‑level transforms
# ---------------------------------------------------------------------------

def prepare_text(example: Dict) -> Dict:
    """Concatenate *title* and *generation* into a single *text* field."""
    if example.get("title") and example.get("generation"):
        example["text"] = f"{example['title'].strip()} {example['generation'].strip()}"
    return example


def encode_label(example: Dict, label_map: Dict[str, int] | None = None) -> Dict:
    """Map the *model* column to an integer label."""
    label_map = label_map or {"human": 0, "machine": 1}
    example["label"] = label_map.get(example.get("model"), -1)
    return example


def tokenize(
    example: Dict,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> Dict:
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    # Preserve label if present
    if "label" in example:
        tokens["label"] = example["label"]
    return tokens


# ---------------------------------------------------------------------------
# Higher‑level helpers
# ---------------------------------------------------------------------------

def build_tf_dataset(hf_ds: Dataset, batch_size: int = 16, shuffle: bool = True) -> tf.data.Dataset:
    """Convert a tokenised HuggingFace Dataset into a `tf.data.Dataset`."""
    hf_ds.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "label"])
    return hf_ds.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="label" if "label" in hf_ds.column_names else None,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ---------------------------------------------------------------------------
# Main processing routine
# ---------------------------------------------------------------------------

def preprocess_data(
    *,
    local: bool = True,
    sample_size: int = 3_000,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    tokenizer_name: str = "bert-base-uncased",
    batch_size: int = 16,
    run_name: str = "raid_local",
) -> Dict[str, tf.data.Dataset]:
    """Download, clean, tokenise and cache the dataset.

    Returns a dict `{split_name: tf.data.Dataset}`.
    """
    splits = download_data(
        local=local,
        sample_size=sample_size,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    out_root = Path("data/processed") / run_name

    tf_datasets: Dict[str, tf.data.Dataset] = {}
    for name, subset in splits.items():
        print(f"▶ Pre‑processing {name} split ({len(subset):,} rows)…")

        # 1. basic cleaning
        subset = subset.filter(lambda x: x["title"] is not None and x["generation"] is not None)
        subset = subset.map(prepare_text)
        

        # 2. labels (skip for un‑labelled test)
        if name != "test":
            subset = subset.map(encode_label)
            subset = subset.filter(lambda x: x["label"] != -1)

        # 3. tokenisation
        subset = subset.map(lambda x: tokenize(x, tokenizer), batched=False)

        # 4. serialise HF dataset
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        subset.save_to_disk(out_dir.as_posix())

        # 5. build & cache tf.data
        tf_ds = build_tf_dataset(subset, batch_size=batch_size, shuffle=(name == "train"))
        tf_dir = out_dir / "tf_dataset"
        tf.data.Dataset.save(tf_ds, tf_dir.as_posix())
        tf_datasets[name] = tf_ds

    print("✅ All splits processed and saved to", out_root.resolve())
    return tf_datasets


# ---------------------------------------------------------------------------
# CLI entry‑point 
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess RAID for BERT models")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--local", action="store_true", help="Run in local mode")
    g.add_argument("--prod", action="store_true", help="Run on full official splits")

    p.add_argument("--sample-size", type=int, default=3_000,
                   help="Rows to sample from raid/train when --local is set")
    p.add_argument("--val-frac", type=float, default=0.1, dest="val_frac")
    p.add_argument("--test-frac", type=float, default=0.1, dest="test_frac")
    p.add_argument("--tokenizer", default="bert-base-uncased")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--run-name", default="raid_local",
                   help="Folder under data/processed for artefacts")
    return p.parse_args()


def main():  # pragma: no cover
    args = _parse_args()
    preprocess_data(
        local=args.local,
        sample_size=args.sample_size,
        validation_fraction=args.val_frac,
        test_fraction=args.test_frac,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
