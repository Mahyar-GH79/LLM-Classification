from transformers import AutoModelForSequenceClassification

def load_model(model_name="bert-base-uncased", num_labels=3):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, force_download=True)
