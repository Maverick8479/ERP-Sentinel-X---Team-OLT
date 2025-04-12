import pandas as pd
import os

LOG_FILE = "logs/query_log.csv"
TAG_FILE = "logs/manual_tags.csv"
os.makedirs("logs", exist_ok=True)

def tag_low_confidence_queries(threshold=0.7):
    df = pd.read_csv(LOG_FILE)
    low_conf = df[df["confidence"] < threshold]

    if low_conf.empty:
        print("✅ No low-confidence queries to tag.")
        return

    print("\n🔍 Admin Review Required:")
    tagged_rows = []

    for _, row in low_conf.iterrows():
        print(f"\nQuery: {row['query']}")
        print(f"Predicted Intent: {row['predicted_intent']} | Confidence: {row['confidence']}")
        correct_intent = input("👨‍💼 Enter correct intent: ")

        tagged_rows.append({
            "timestamp": row["timestamp"],
            "query": row["query"],
            "correct_intent": correct_intent
        })

    if tagged_rows:
        tag_df = pd.DataFrame(tagged_rows)
        tag_df.to_csv(TAG_FILE, index=False, mode='a', header=not os.path.exists(TAG_FILE))
        print("✅ Tags saved to manual_tags.csv")
