"""
Minimal BERT sequence‑classifier wrapper (TensorFlow/Keras).
"""

# src/model.py
from transformers import (
    TFBertForSequenceClassification,
    TFDistilBertForSequenceClassification,
    AutoConfig,
)

def get_model(model_name: str, num_labels: int = 2):
    cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    if cfg.model_type == "distilbert":
        return TFDistilBertForSequenceClassification.from_pretrained(
            model_name, config=cfg
        )
    elif cfg.model_type == "bert":
        return TFBertForSequenceClassification.from_pretrained(
            model_name, config=cfg
        )
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")
