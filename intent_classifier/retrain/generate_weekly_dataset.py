import pandas as pd
import os

log_file = "logs/query_log.csv"
tag_file = "logs/manual_tags.csv"
output_file = "retrain/weekly_dataset.csv"

os.makedirs("retrain", exist_ok=True)

def generate_training_data():
    if not os.path.exists(log_file):
        print("❌ query_log.csv not found.")
        return

    df_log = pd.read_csv(log_file)
    if df_log.empty:
        print("❌ query_log.csv is empty.")
        return

    # ✅ Fix: Safely read manual_tags only if non-empty
    if os.path.exists(tag_file) and os.path.getsize(tag_file) > 0:
        df_tags = pd.read_csv(tag_file)
    else:
        df_tags = pd.DataFrame(columns=["query", "correct_intent"])

    df_merged = df_log.merge(df_tags, on="query", how="left")
    df_merged["intent"] = df_merged["correct_intent"].fillna(df_merged["predicted_intent"])
    df_final = df_merged[["query", "intent"]]

    if df_final.empty:
        print("⚠️ No data to write in final dataset.")
        return

    df_final.to_csv(output_file, index=False)
    print("📦 Generated retraining dataset:", output_file)
    print(df_final.head())

# Run it
generate_training_data()
