"""
Minimal BERT sequence‑classifier wrapper (TensorFlow/Keras).
"""

from transformers import TFBertForSequenceClassification

def get_model(model_name: str = "bert-base-uncased", num_labels: int = 2):
    """
    Load a pre‑trained BERT and initialise the classification head.

    Parameters
    ----------
    model_name
        Any checkpoint from the Hugging Face Hub.
    num_labels
        Number of classes (binary = 2).

    Returns
    -------
    TFBertForSequenceClassification
    """
    
    return TFBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
