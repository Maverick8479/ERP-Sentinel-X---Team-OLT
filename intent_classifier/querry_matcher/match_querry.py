import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import os

# Add the parent folder (intent_classifier) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load FAISS index and metadata
index = faiss.read_index("idms_faq_index.index")
with open("idms_faq_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

questions = metadata["questions"]
answers = metadata["answers"]

# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Optional: Import your intent classifier here
from inference import classify_query
from related_suggestions import suggest_related_questions
def match_query(user_query: str, top_k: int = 3, threshold: float = 0.5):
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, top_k)
    top_matches = []

    for score, idx in zip(D[0], I[0]):
        top_matches.append({
            "score": float(score),
            "question": questions[idx],
            "answer": answers[idx]
        })

    # High confidence
    if D[0][0] > 0.8:
        return {
            "match_type": "high_confidence",
            "best_match": top_matches[0],
            "suggestions": top_matches[1:]
        }

    # Medium confidence
    elif D[0][0] >= threshold:
        return {
            "match_type": "medium_confidence",
            "best_match": top_matches[0],
            "suggestions": top_matches[1:]
        }

    # Low confidence: Trigger intent fallback
    else:
        intent_result = classify_query(user_query)
        return {
            "match_type": "fallback",
            "fallback_method": intent_result["method"],
            "predicted_intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"],
            "suggestions": top_matches  # still return some FAQ matches
        }
        if intent_result["match_type"] == "fallback":
            print("\n🤖 Sorry, I'm not confident about your query.")
            print("Did you mean:")

            suggestions = suggest_related_questions(user_query)
            for i, suggestion in enumerate(suggestions):
                print(f"{i+1}. {suggestion['question']}")

            # Simulate user selection (could be input in a UI)
            user_choice = int(input("Choose an option (1-3): ")) - 1
            selected_query = suggestions[user_choice]["question"]

# # 🔍 Example Usage
if __name__ == "__main__":
    query = "I want to file GSTR-3B"
    result = match_query(query)
    print(result)   
