import torch
import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import numpy as np
from logger import log_query
# Load fine-tuned model
model_path = "./model_output"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Load intent mapping
intent_map = json.load(open(f"{model_path}/intent_map.json"))
id_to_intent = {v: k for k, v in intent_map.items()}
label_list = list(intent_map.keys())

# Load zero-shot model for fallback
zero_shot_model = SentenceTransformer("all-MiniLM-L6-v2")

def classify_query(query: str, threshold: float = 0.7):
    # Step 1: Use Fine-tuned model
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy().flatten()
    top_idx = int(np.argmax(probs))
    top_score = probs[top_idx]
    predicted_intent = id_to_intent[top_idx]

    if top_score >= threshold:
        result = {
            "intent": predicted_intent,
            "confidence": float(top_score),
            "method": "fine-tuned"
        }
        log_query(query, result["intent"], result["confidence"], resolved=True, user_feedback="Yes")
        return result

    # Step 2: Fallback to Zero-Shot intent matching
    query_embedding = zero_shot_model.encode(query, convert_to_tensor=True)
    label_embeddings = zero_shot_model.encode(label_list, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, label_embeddings)[0]
    top_fallback_idx = int(torch.argmax(cos_scores))
    fallback_intent = label_list[top_fallback_idx]
    fallback_score = float(cos_scores[top_fallback_idx])

    result = {
        "intent": fallback_intent,
        "confidence": fallback_score,
        "method": "zero-shot"
    }

    # ✅ Only log after result is constructed
    log_query(query, result["intent"], result["confidence"], resolved=True, user_feedback="Yes")
    return result