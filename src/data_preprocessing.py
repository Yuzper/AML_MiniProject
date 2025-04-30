"""
Light-weight preprocessing pipeline for the RAID anti-AI-text dataset
=====================================================================

Exports Hugging Face **arrow** datasets ready for PyTorch/🤗 Trainer.

─────────────────────────────────────────────────────────────────────────────
Example (local debug run):
    python -m src.data_preprocessing \
        --local \
        --sample-size 3_000 \
        --tokenizer distilbert-base-uncased \
        --run-name raid_local

Example (full production):
    python -m src.data_preprocessing \
        --prod \
        --tokenizer bert-base-uncased \
        --run-name raid_full
─────────────────────────────────────────────────────────────────────────────

Updates (2025-04-20)
--------------------
* **Fix**: public *raid_test* split has no `title`/`generation` columns – we now
  build a unified `text` field that works for *all* splits.
* **Refactor**: removed schema-specific filters; the pipeline now checks only
  the presence of the new `text` column, so local and production paths share
  identical logic.
* Minor: renamed `prepare_text` → `build_text` for clarity.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, set_seed

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG_SEED = 42
set_seed(RNG_SEED)               # sets Python, NumPy and PyTorch seeds
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------------------
# Download & basic filtering
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
    """Return a `DatasetDict` with *train*, *val*, *test* splits."""
    ds = load_dataset("liamdugan/raid", "raid", split="train")

    # Keep only Reddit domain (as in the original script)
    ds = ds.filter(lambda x: x["domain"] == "reddit")

    if local:
        ds = ds.shuffle(seed=RNG_SEED).select(range(sample_size))

    strat_col = None
    if stratify:
        ds = ds.map(_make_source_type, batched=False, load_from_cache_file=False)
        ds = ds.class_encode_column("source_type")
        strat_col = "source_type"

    # train / test split
    tmp = ds.train_test_split(
        test_size=test_fraction,
        seed=RNG_SEED,
        stratify_by_column=strat_col,
    )
    train_val, test_ds = tmp["train"], tmp["test"]

    # train / val split
    val_size = validation_fraction / (1.0 - test_fraction)
    tmp = train_val.train_test_split(
        test_size=val_size,
        seed=RNG_SEED,
        stratify_by_column=strat_col,
    )
    train_ds, val_ds = tmp["train"], tmp["test"]

    # Remove helper column
    if strat_col:
        train_ds = train_ds.remove_columns(strat_col)
        val_ds   = val_ds.remove_columns(strat_col)
        test_ds  = test_ds.remove_columns(strat_col)

    return DatasetDict(train=train_ds, val=val_ds, test=test_ds)


# ---------------------------------------------------------------------------
# Row-level transforms
# ---------------------------------------------------------------------------
def build_text(example: Dict) -> Dict:
    """Unify heterogeneous column names into a single `text` string."""
    pieces = []
    if example.get("title"):
        pieces.append(example["title"].strip())
    if example.get("generation"):
        pieces.append(example["generation"].strip())
    if example.get("text"):
        pieces.append(example["text"].strip())

    example["text"] = " ".join(pieces).strip()
    return example


def encode_label(example: Dict) -> Dict:
    """Map human → 0, machine → 1."""
    example["labels"] = np.int64(0 if example["model"] == "human" else 1)
    return example


def tokenize_fn(example: Dict, tokenizer, max_length: int) -> Dict:
    """Tokenise text and keep the label, if present."""
    toks = tokenizer(
        example["text"],
        padding="longest",
        truncation=True,
        max_length=max_length,
    )
    if "labels" in example:
        toks["labels"] = example["labels"]
    return toks


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
    max_length: int = 256,
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
        # ➊ Build/keep a unified `text` column
        ds = ds.map(build_text, desc=f"{name}: build_text")

        # ➋ Drop rows that still have empty text
        ds = ds.filter(lambda x: x["text"] != "", desc=f"{name}: filter_non_empty")

        # ➌ Label encoding only for train / val
        if name != "test":
            ds = ds.map(encode_label, desc=f"{name}: encode_label")

        # ➍ Tokenise
        ds = ds.map(
            lambda ex: tokenize_fn(ex, tokenizer, max_length),
            batched=False,
            desc=f"{name}: tokenize",
        )

        # ➎ Persist to disk (arrow format)
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(out_dir.as_posix())

        print(f"✅ Saved {name} split to {out_dir}")

    print("📦 All splits cached under:", out_root.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    ap = argparse.ArgumentParser("Preprocess RAID for BERT/DistilBERT models")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true", help="local debug mode")
    mode.add_argument("--prod",  action="store_true", help="full dataset mode")

    ap.add_argument("--sample-size", type=int, default=3_000)
    ap.add_argument("--val-frac",  type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--no-stratify", action="store_true")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--run-name",  default="raid_local")
    return ap.parse_args()


def main():  # pragma: no cover
    args = _parse_args()
    preprocess_data(
        local=args.local,
        sample_size=args.sample_size,
        validation_fraction=args.val_frac,
        test_fraction=args.test_frac,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        run_name=args.run_name,
        stratify=not args.no_stratify,
    )


if __name__ == "__main__":
    main()