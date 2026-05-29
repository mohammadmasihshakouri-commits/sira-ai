import json
import os
import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")


def analyze_intent(user_text, conversation_state):
    prompt = f"""
You are an intent classifier for a Persian customer support agent.

Return ONLY valid JSON. No explanation.

User message:
{user_text}

Current conversation state:
{conversation_state}

Classify the message into one of these intents:
- greeting
- thanks
- closing_no_more_questions
- has_more_question
- cancellation_request
- reservation_code
- seat_lock_issue
- payment_issue
- technical_issue
- unknown

JSON format:
{{
  "intent": "...",
  "confidence": 0.0,
  "topic": "...",
  "is_conversation_closing": false,
  "is_new_question": false
}}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=30,
    )

    response.raise_for_status()

    raw_text = response.json().get("response", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "topic": "unknown",
            "is_conversation_closing": False,
            "is_new_question": False,
            "raw": raw_text,
        }