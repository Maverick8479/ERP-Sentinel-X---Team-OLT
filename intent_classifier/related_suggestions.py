import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load vector index & metadata
index = faiss.read_index("idms_faq_index.index")
with open("idms_faq_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

questions = metadata["questions"]
answers = metadata["answers"]

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")


def suggest_related_questions(user_query, top_k=3):
    query_vector = model.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)

    D, I = index.search(query_vector, top_k)

    suggestions = []
    for score, idx in zip(D[0], I[0]):
        suggestions.append({
            "score": round(float(score), 3),
            "question": questions[idx],
            "answer": answers[idx]
        })

    return suggestions
