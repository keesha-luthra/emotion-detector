from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import json
import os

# ================= LOAD DATASET =================
dataset = load_dataset("shreyaspullehf/emotion-dataset-20-emotions")

# same split logic as training
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset["validation"] = split["test"]

# ================= LOAD LABEL MAPPING =================
with open("./my_emotion_model_20/label_mapping.json", "r") as f:
    id2label = json.load(f)

label2id = {v: int(k) for k, v in id2label.items()}

def encode_labels(example):
    example["labels"] = label2id[example["emotion"]]
    return example

dataset["validation"] = dataset["validation"].map(encode_labels)

# ================= LOAD MODEL =================
model = AutoModelForSequenceClassification.from_pretrained("./my_emotion_model_20")
tokenizer = AutoTokenizer.from_pretrained("./my_emotion_model_20")

# ================= TOKENIZE =================
def tokenize(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset["validation"] = dataset["validation"].map(tokenize, batched=True)

dataset["validation"].set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# ================= PREDICT =================
trainer = Trainer(model=model)

predictions = trainer.predict(dataset["validation"])

y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

# ================= RESULTS =================
accuracy = accuracy_score(y_true, y_pred)

print("\n================ RESULTS ================\n")
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
report = classification_report(y_true, y_pred)
print(report)

# ================= SAVE RESULTS =================
os.makedirs("results", exist_ok=True)
with open("results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)

print("\nResults saved to results/metrics.txt")