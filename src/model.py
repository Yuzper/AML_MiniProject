from transformers import TFBertForSequenceClassification

def get_model(model_name: str = "bert-base-uncased", num_labels: int = 2):
    return TFBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
