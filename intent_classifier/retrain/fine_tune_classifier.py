import pandas as pd
import os
import json
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# ✅ Load data
data = pd.read_csv("retrain/weekly_dataset.csv")
queries = data["query"].tolist()
labels = data["intent"].tolist()

# ✅ Encode intent labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# ✅ Save label mapping
intent_map = {label: int(code) for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
os.makedirs("model_output", exist_ok=True)
with open("model_output/intent_map.json", "w") as f:
    json.dump(intent_map, f)

# ✅ Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
encodings = tokenizer(queries, truncation=True, padding=True)

# ✅ Dataset formatting
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {"labels": torch.tensor(self.labels[idx])}

# ✅ Split queries and labels before tokenizing
train_queries, val_queries, train_labels, val_labels = train_test_split(queries, encoded_labels, test_size=0.2)

# ✅ Tokenize separately
train_encodings = tokenizer(train_queries, truncation=True, padding=True)
val_encodings = tokenizer(val_queries, truncation=True, padding=True)

# ✅ Create datasets
train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)


# ✅ Model definition
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_encoder.classes_))

# ✅ Training args
training_args = TrainingArguments(
    output_dir="model_output",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# ✅ Metrics
def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, f1_score
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ✅ Start training
trainer.train()
trainer.save_model("model_output")
tokenizer.save_pretrained("model_output")
print("✅ Model fine-tuned and saved to `model_output/`")
