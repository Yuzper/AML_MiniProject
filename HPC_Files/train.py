import importlib
import subprocess
import sys

# Ensure required packages are available
required_packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "tensorflow": "tensorflow",
    "datasets": "datasets",
    "transformers": "transformers",
}

def install_and_import(pkg_name, install_name=None):
    install_name = install_name or pkg_name
    try:
        importlib.import_module(pkg_name)
        print(f"{pkg_name} is already installed.")
    except ImportError:
        print(f"{pkg_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

for pkg, pip_name in required_packages.items():
    install_and_import(pkg, pip_name)

# Imports
import numpy as np
import random
import tensorflow as tf
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Dict

# Set random seeds
RandomState = 42
random.seed(RandomState)
np.random.seed(RandomState)
tf.random.set_seed(RandomState)

def prepare_text(dataset: Dict) -> Dict:
    if dataset["title"] is None or dataset["generation"] is None:
        return None
    dataset["text"] = dataset["title"].strip() + " " + dataset["generation"].strip()
    return dataset

def encode_label(dataset: Dict, label_map={"human": 0, "machine": 1}) -> Dict:
    dataset["label"] = label_map.get(dataset["model"], -1)
    return dataset

def tokenize_example(dataset: Dict, tokenizer) -> Dict:
    tokens = tokenizer(
        dataset["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokens["label"] = dataset["label"]
    return tokens

def prepare_dataset_for_bert(dataset, tokenizer_name="bert-base-uncased"):
    dataset = dataset.filter(lambda x: x["title"] is not None and x["generation"] is not None)
    dataset = dataset.map(prepare_text)
    dataset = dataset.map(encode_label)
    dataset = dataset.filter(lambda x: x["label"] != -1)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = dataset.map(lambda x: tokenize_example(x, tokenizer), batched=False)

    dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "label"])
    
    tf_dataset = dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="label",
        shuffle=True,
        batch_size=16,
        collate_fn=None
    )
    
    return tf_dataset

if __name__ == "__main__":
    print("📥 Loading RAID dataset...")
    data_all = load_dataset("liamdugan/raid", "raid")
    train_data_subset = data_all["train"]

    print("🧹 Preprocessing...")
    train_tf_dataset = prepare_dataset_for_bert(train_data_subset)

    output_path = "data/processed/raid_local/train/tf_dataset"
    train_tf_dataset.save(output_path)
    print(f"✅ Saved cleaned dataset to {output_path}")