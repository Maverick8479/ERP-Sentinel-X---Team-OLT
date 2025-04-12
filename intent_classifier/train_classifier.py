import os
import json
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer

# 1. Load CSV
df = pd.read_csv("data/intent_dataset.csv")

# 2. Encode labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["intent"])

# ✅ FIX: Create intent_map before saving it
# Create intent_map with native Python int values
intent_map = {label: int(idx) for label, idx in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

# Ensure output folder exists
os.makedirs("model_output", exist_ok=True)

# Save as JSON
with open("model_output/intent_map.json", "w") as f:
    json.dump(intent_map, f)

# 4. Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 5. Hugging Face Dataset
dataset = Dataset.from_pandas(df)

def preprocess(example):
    return tokenizer(example["query"], truncation=True, padding=True)

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

# 6. Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(intent_map)
)

# 7. Training config
args = TrainingArguments(
    output_dir="model_output",
    evaluation_strategy="epoch",     # ← Evaluate every epoch
    save_strategy="epoch",           # ← Save model every epoch
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_dir="logs",
    save_total_limit=1,
    load_best_model_at_end=True,
)


# 8. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

# 9. Train and Save
trainer.train()
trainer.save_model("model_output")
tokenizer.save_pretrained("model_output")
