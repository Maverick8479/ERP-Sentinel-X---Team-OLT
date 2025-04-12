import json
import requests

# Load mapping file
with open("intent_action_map.json", "r") as f:
    intent_actions = json.load(f)


def execute_intent_action(intent, user_query=None):
    # Find matching intent
    matched = next((item for item in intent_actions if item["intent"] == intent), None)

    # ⛔ Default fallback: escalate unknown intents
    if not matched:
        return {
            "status": "fallback",
            "intent": intent,
            "response_type": "ESCALATE",
            "message": f"❗ This intent is not recognized or supported yet. Escalating to support.",
            "api_call": "none",
            "escalate": True
        }

    # ✅ Handle mapped intent
    response = {
        "status": "success",
        "intent": intent,
        "response_type": matched["response_type"],
        "message": matched["answer"],
        "escalate": matched["escalate"]
    }

    # Simulate API call
    if matched["response_type"] == "API" and matched["api_call"] != "none":
        try:
            print(f"🔄 Calling API: {matched['api_call']}")
            # Optional: use real requests.get(...)
            simulated_response = {"status": "ok", "data": {"po_status": "Approved"}}
            response["api_response"] = simulated_response
        except Exception as e:
            response["status"] = "error"
            response["message"] = f"API call failed: {str(e)}"

    return response
