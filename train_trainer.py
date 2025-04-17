import zipfile
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer
from dataset import LLMTripleDataset
from models import CausalLMClassifier
from utils import preprocess_multiclass, compute_metrics
from torch.utils.data import random_split
import torch

# Config

MODEL_NAME = "tiiuae/falcon-rw-1b"
EPOCHS = 4
BATCH_SIZE = 8

# Load and preprocess data
zf = zipfile.ZipFile('dataset/train.csv.zip')
df = pd.read_csv(zf.open('train.csv'))
df = preprocess_multiclass(df)

# Tokenizer and datasets
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
ds = LLMTripleDataset(df=df, tokenizer=tokenizer)
train_ds, val_ds = random_split(ds, [0.8, 0.2])

# Calculate training steps per epoch
num_train_samples = len(train_ds)
steps_per_epoch = num_train_samples // BATCH_SIZE
eval_steps = steps_per_epoch // 2

# Model 
model = CausalLMClassifier(lm_name=MODEL_NAME, num_classes=3)


# Training Arguments
args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=5e-6,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=eval_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=50,
    save_safetensors=False,
    report_to="tensorboard",
    fp16=False,
)

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


