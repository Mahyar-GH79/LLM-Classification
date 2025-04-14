from transformers import Trainer, TrainingArguments, AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import evaluate
import torch
import zipfile
from dataset import LLMTripleDataset
from model import load_model
from utils import preprocess_multiclass

# Config
MODEL_NAME = "bert-base-uncased"
EPOCHS = 30
BATCH_SIZE = 4

# Load and preprocess data
zf = zipfile.ZipFile('dataset/train.csv.zip')
df = pd.read_csv(zf.open('train.csv'))
df = preprocess_multiclass(df)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Tokenizer and datasets
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_ds = LLMTripleDataset(train_df, tokenizer)
val_ds = LLMTripleDataset(val_df, tokenizer)

# Model 
model = load_model(MODEL_NAME, num_labels=3)

# Freeze Bert and finetune the MLP head
for param in model.bert.parameters():
    param.requires_grad = False

# Training Arguments
args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=50,
    report_to="tensorboard"
)

# Metrics
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    return accuracy.compute(predictions=preds, references=labels)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train and save
trainer.train()
trainer.save_model("output_model")
