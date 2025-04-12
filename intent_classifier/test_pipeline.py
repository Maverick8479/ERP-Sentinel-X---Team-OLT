import sys
import os

# Add parent directory to the path to access all modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from querry_matcher.match_querry  import match_query
from inference import classify_query
from intent_action_engine import execute_intent_action
from related_suggestions import suggest_related_questions

def run_test(user_query):
    print(f"\n🔍 User Query: {user_query}")
    
    # Step 1: Try matching via FAQ
    faq_result = match_query(user_query)

    if faq_result["match_type"] == "high_confidence":
        print(f"✅ FAQ Match: {faq_result['best_match']['question']}")
        print(f"📘 Answer: {faq_result['best_match']['answer']}")
        return

    # Step 2: If not confident, use intent classification
    intent_result = classify_query(user_query)
    print(f"\n🧠 Intent Classification Result:")
    print(f"Intent: {intent_result['intent']} | Confidence: {round(intent_result['confidence']*100, 2)}% | Method: {intent_result['method']}")

    # Step 3: Handle low confidence
    if intent_result["confidence"] < 0.7:
        print("\n🤔 Not confident. Suggesting related questions:")
        suggestions = suggest_related_questions(user_query)
        for i, suggestion in enumerate(suggestions):
            print(f"{i+1}. {suggestion['question']}")

        user_choice = int(input("Select a related question (1/2/3): ")) - 1
        refined_query = suggestions[user_choice]["question"]
        intent_result = classify_query(refined_query)  # Reclassify based on selected suggestion

    # Step 4: Intent → Action Mapping
    action_result = execute_intent_action(intent_result["intent"], user_query=user_query)
    print(f"\n🎯 Final Action: {action_result['response_type']}")
    print(f"📢 Message: {action_result['message']}")
    if action_result.get("api_response"):
        print(f"📡 API Response: {action_result['api_response']}")

    if action_result["escalate"]:
        print("🚨 Escalation triggered.")

if __name__ == "__main__":
    while True:
        q = input("\nType your ERP-related query (or 'exit' to quit): ")
        if q.lower() == "exit":
            break
        run_test(q)
