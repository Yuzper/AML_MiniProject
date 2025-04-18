# src/data_preprocessing.py
"""Light‑weight preprocessing pipeline for the RAID anti‑AI‑text dataset.

Changes (2025‑04‑17)
-------------------
* **Fix** stratified splitting error when `model` column is a plain `Value`.
  We now create a temporary binary column `source_type` (human vs machine),
  encode it as a `ClassLabel`, and stratify on that. The column is removed
  before returning the splits.
* Minor: expose `--no-stratify` flag to allow purely random slicing when the
  user doesn't care about class balance.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

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

def _make_source_type(example):
    """Add a binary *source_type* column used for stratification."""
    example["source_type"] = "human" if example["model"] == "human" else "machine"
    return example


def download_data(
    *,
    local: bool = True,
    sample_size: int = 3_000,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    stratify: bool = True,
) -> DatasetDict:
    """Return a DatasetDict with *train*, *val*, *test* splits.

    Parameters
    ----------
    local : bool
        If *True*, download only a slice of `raid/train` and derive in‑domain
        val & test. Otherwise pull the full authors' splits.
    stratify : bool
        If *True* and *local*, attempt stratified splitting on the temporary
        *source_type* column. Set to *False* to disable.
    """
    if local:
        print("▶ Local mode – downloading a small slice of raid/train")
        ds = load_dataset("liamdugan/raid", "raid", split=f"train[:{sample_size}]")

        if stratify:
            # Add binary column & encode as ClassLabel for stratification
            ds = ds.map(_make_source_type)
            ds = ds.class_encode_column("source_type")
            strat_col = "source_type"
        else:
            strat_col = None

        print("▶ Creating validation / test splits (stratified =", bool(strat_col), ")")
        tmp = ds.train_test_split(
            test_size=test_fraction,
            seed=RNG_SEED,
            stratify_by_column=strat_col,
        )
        train_val, test_ds = tmp["train"], tmp["test"]

        val_size = validation_fraction / (1.0 - test_fraction)
        tmp = train_val.train_test_split(
            test_size=val_size,
            seed=RNG_SEED,
            stratify_by_column=strat_col,
        )
        train_ds, val_ds = tmp["train"], tmp["test"]

        # Drop helper column if it was added
        if strat_col:
            train_ds = train_ds.remove_columns(strat_col)
            val_ds = val_ds.remove_columns(strat_col)
            test_ds = test_ds.remove_columns(strat_col)
        
        return DatasetDict(train=train_ds, val=val_ds, test=test_ds)

    # ───────────────────────── Production mode ─────────────────────────────
    print("▶ Production mode – pulling full official splits")
    train_ds = load_dataset("liamdugan/raid", "raid", split="train")
    val_ds = load_dataset("liamdugan/raid", "raid", split="extra")
    test_ds = load_dataset("liamdugan/raid", "raid_test", split="train")
    return DatasetDict(train=train_ds, val=val_ds, test=test_ds)


# ---------------------------------------------------------------------------
# Row‑level transforms
# ---------------------------------------------------------------------------

def prepare_text(example: Dict) -> Dict:
    if example.get("title") and example.get("generation"):
        example["text"] = f"{example['title'].strip()} {example['generation'].strip()}"
    return example


def encode_label(example, label_map=None):
    example["labels"] = np.int32(0 if example["model"] == "human" else 1)
    return example

def tokenize(example: Dict, tokenizer, max_length: int = 512) -> Dict:
    toks = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    if "labels" in example:
        toks["labels"] = example["labels"]
    return toks


# ---------------------------------------------------------------------------
# tf.data conversion helper
# ---------------------------------------------------------------------------

def build_tf_dataset(hf_ds: Dataset, batch_size: int = 16, shuffle: bool = True) -> tf.data.Dataset:
    """Convert a tokenised HF dataset into a `tf.data.Dataset`.

    Returns *(x_dict, y_tensor)* tuples when labels are present, otherwise just
    *x_dict* for the public `raid_test` split.
    """
    
    has_labels = "labels" in hf_ds.column_names
    cols = ["input_ids", "attention_mask"] + (["labels"] if has_labels else [])

    hf_ds.set_format("tensorflow", columns=cols)

    return hf_ds.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["labels"] if has_labels else None,
        shuffle=shuffle if has_labels else False,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Orchestrator
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
    stratify: bool = True,
):
    splits = download_data(
        local=local,
        sample_size=sample_size,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        stratify=stratify,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    out_root = Path("data/processed") / run_name

    for name, ds in splits.items():
        print(f"▶ Processing {name} split ({len(ds):,} rows)…")
        ds = ds.filter(lambda x: x["title"] is not None and x["generation"] is not None)
        ds = ds.map(prepare_text)

        if name != "test":
            ds = ds.map(encode_label)
            print(f"{name} labels -1 count:", sum(1 for x in ds if x["labels"] == -1))
            ds = ds.filter(lambda x: x["labels"] != -1)

        print("Before tokenizing:", ds.column_names)
        ds = ds.map(lambda ex: tokenize(ex, tokenizer), batched=False)
        print("After tokenizing:", ds.column_names)

        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(out_dir.as_posix())

        tf_ds = build_tf_dataset(ds, batch_size=batch_size, shuffle=(name == "train"))

        tf.data.Dataset.save(tf_ds, (out_dir / "tf_dataset").as_posix())

    print("✅ All splits cached in", out_root.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser("Preprocess RAID for BERT models")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true", help="local debug mode")
    mode.add_argument("--prod", action="store_true", help="full dataset mode")

    ap.add_argument("--sample-size", type=int, default=3_000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--no-stratify", action="store_true")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--run-name", default="raid_local")
    return ap.parse_args()


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
        stratify=not args.no_stratify,
    )


if __name__ == "__main__":
    main()
