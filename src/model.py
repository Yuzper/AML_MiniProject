"""
Minimal BERT sequence‑classifier wrapper (TensorFlow/Keras).
"""

# src/model.py
from transformers import AutoConfig, AutoModelForSequenceClassification


def get_model(model_name: str, num_labels: int = 2):
    """
    Return a PyTorch sequence-classifier (BERT, DistilBERT, …) with `num_labels`.
    """
    cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)