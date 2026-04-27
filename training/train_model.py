import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score

# ================= SET SEED =================
set_seed(42)

# ================= LOAD DATASET =================
dataset = load_dataset("shreyaspullehf/emotion-dataset-20-emotions")

# Create validation split if not present
if "train" in dataset and "test" not in dataset:
    split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset["train"] = split["train"]
    dataset["validation"] = split["test"]

# ================= LABEL HANDLING =================
dataset = dataset.rename_column("emotion", "labels")

# 🔥 Deterministic label order (CRITICAL FIX)
label_list = sorted(dataset["train"].unique("labels"))

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

def encode_labels(example):
    example["labels"] = label2id[example["labels"]]
    return example

dataset["train"] = dataset["train"].map(encode_labels)
dataset["validation"] = dataset["validation"].map(encode_labels)

num_labels = len(label_list)

# ================= TOKENIZER =================
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# 🔥 batched=True → faster
dataset["train"] = dataset["train"].map(tokenize, batched=True)
dataset["validation"] = dataset["validation"].map(tokenize, batched=True)

dataset["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataset["validation"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ================= MODEL =================
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,   # 🔥 ensures correct config.json
    label2id=label2id
)

# ================= METRICS =================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# ================= TRAINING CONFIG =================
training_args = TrainingArguments(
    output_dir="./results_20emo",
    logging_dir="./logs_20emo",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

# ================= TRAINER =================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics
)

# ================= TRAIN =================
trainer.train()

# ================= SAVE MODEL =================
trainer.save_model("./my_emotion_model_20")
tokenizer.save_pretrained("./my_emotion_model_20")

# 🔥 Save label mapping (optional but useful)
with open("./my_emotion_model_20/label_mapping.json", "w") as f:
    json.dump(id2label, f, indent=2)

print("Training complete. Model saved to ./my_emotion_model_20")