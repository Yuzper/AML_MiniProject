import importlib
import subprocess
import sys

# Dict of required packages
required_packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "tensorflow": "tensorflow",
    "datasets": "datasets",
    "transformers": "transformers",
    "tf-keras": "tf-keras",
    "typing": "typing"
}

def install_and_import(pkg_name, install_name=None):
    install_name = install_name or pkg_name
    try:
        importlib.import_module(pkg_name)
        print(f"{pkg_name} is already installed.")
    except ImportError:
        print(f"{pkg_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "conda", "install", install_name])

# Loop and ensure all are installed
for pkg, pip_name in required_packages.items():
    install_and_import(pkg, pip_name)

####################################################################################################################################
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sklearn
from sklearn.utils import check_random_state
import tensorflow as tf
from datasets import load_dataset
from transformers import TFBertForSequenceClassification
from transformers import AutoTokenizer
from datasets import Dataset
from typing import Dict


# Set random states for reproducability
RandomState = 42
random.seed(RandomState)
np.random.seed(RandomState)
skl_rand = check_random_state(RandomState)
tf.random.set_seed(RandomState)

print("Random seeds defined.")

# Different selection of data
# Takes a long time to load first time around...
data_all = load_dataset("liamdugan/raid", "raid")
train_data_subset = data_all["train"]


# Combine title + generation into one for training
def prepare_text(dataset: Dict) -> Dict:
    if dataset["title"] is None or dataset["generation"] is None:
        return None
    dataset["text"] = dataset["title"].strip() + " " + dataset["generation"].strip()
    return dataset

# Encode binary labels
def encode_label(dataset: Dict, label_map={"human": 0, "machine": 1}) -> Dict:
    dataset["label"] = label_map.get(dataset["model"], -1)
    return dataset

# Tokenization function
def tokenize_example(dataset: Dict, tokenizer) -> Dict:
    tokens = tokenizer(
        dataset["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokens["label"] = dataset["label"]
    return tokens

# Full "main" pipeline
def prepare_dataset_for_bert(dataset, tokenizer_name="bert-base-uncased"):
    # Filter rows with missing title or generation
    dataset = dataset.filter(lambda x: x["title"] is not None and x["generation"] is not None)
    dataset = dataset.map(prepare_text)
    
    # Encode labels
    dataset = dataset.map(encode_label)
    dataset = dataset.filter(lambda x: x["label"] != -1)
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = dataset.map(lambda x: tokenize_example(x, tokenizer), batched=False)

    # Set format for tf.data.Dataset
    dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "label"])
    
    # Convert to tf.data.Dataset
    features = {
        "input_ids": tf.TensorSpec(shape=(512,), dtype=tf.int32),
        "attention_mask": tf.TensorSpec(shape=(512,), dtype=tf.int32),
    }

    tf_dataset = dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="label",
        shuffle=True,
        batch_size=16,
        collate_fn=None
    )
    
    return tf_dataset, tokenizer

train_tf_dataset, tokenizer = prepare_dataset_for_bert(train_data_subset)
train_tf_dataset.save('cleaned_train_dataset')