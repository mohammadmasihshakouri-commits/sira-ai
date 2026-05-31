from __future__ import annotations

import json
import os
from typing import Any

import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")


RECEPTIONIST_INTENTS = [
    "general_info",
    "book_appointment",
    "ask_availability",
    "change_appointment",
    "cancel_appointment",
    "take_message",
    "transfer_call",
    "greeting",
    "thanks",
    "unknown",
]


def _safe_json_loads(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {
        "intent": "unknown",
        "confidence": 0.0,
        "needs_customer_identity": False,
        "service": "",
        "date_text": "",
        "time_text": "",
        "datetime_text": "",
        "is_general_question": False,
        "requires_calendar": False,
        "requires_client_lookup": False,
        "clarification_needed": True,
        "clarification_question": "لطفاً بفرمایید دقیقاً چه کاری براتون انجام بدم؟",
        "summary": "",
        "raw": raw_text,
    }


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    intent = result.get("intent", "unknown")

    if intent not in RECEPTIONIST_INTENTS:
        intent = "unknown"

    confidence = result.get("confidence", 0.0)

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(confidence, 1.0))

    appointment_intents = {
        "book_appointment",
        "ask_availability",
        "change_appointment",
        "cancel_appointment",
    }

    is_general_question = bool(result.get("is_general_question", False))
    requires_calendar = bool(result.get("requires_calendar", False))
    requires_client_lookup = bool(result.get("requires_client_lookup", False))
    needs_customer_identity = bool(result.get("needs_customer_identity", False))

    if intent == "general_info":
        is_general_question = True
        requires_calendar = False
        requires_client_lookup = False
        needs_customer_identity = False

    if intent in appointment_intents:
        is_general_question = False
        requires_calendar = True

        if intent in {"book_appointment", "change_appointment", "cancel_appointment"}:
            requires_client_lookup = True
            needs_customer_identity = True

    if intent in {"greeting", "thanks", "unknown"}:
        requires_calendar = False
        requires_client_lookup = False
        needs_customer_identity = False

    clarification_needed = bool(result.get("clarification_needed", False))
    clarification_question = str(result.get("clarification_question", "") or "")

    if intent == "unknown" and not clarification_question:
        clarification_needed = True
        clarification_question = "لطفاً بفرمایید دقیقاً چه کاری براتون انجام بدم؟"

    return {
        "intent": intent,
        "confidence": confidence,
        "needs_customer_identity": needs_customer_identity,
        "service": str(result.get("service", "") or ""),
        "date_text": str(result.get("date_text", "") or ""),
        "time_text": str(result.get("time_text", "") or ""),
        "datetime_text": str(result.get("datetime_text", "") or ""),
        "is_general_question": is_general_question,
        "requires_calendar": requires_calendar,
        "requires_client_lookup": requires_client_lookup,
        "clarification_needed": clarification_needed,
        "clarification_question": clarification_question,
        "summary": str(result.get("summary", "") or ""),
    }


def analyze_receptionist_intent(
    user_text: str,
    conversation_state: dict[str, Any] | None = None,
    business_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conversation_state = conversation_state or {}
    business_context = business_context or {}

    prompt = f"""
You are an intent analyzer for a Persian receptionist AI agent.

The receptionist must NOT ask for name or phone number for general questions.
Only ask for customer identity when the user's request needs customer-specific action.

Return ONLY valid JSON. No markdown. No explanation.

User message:
{user_text}

Conversation state:
{conversation_state}

Business context:
{business_context}

Classify the user message into exactly one of these intents:
- general_info
- book_appointment
- ask_availability
- change_appointment
- cancel_appointment
- take_message
- transfer_call
- greeting
- thanks
- unknown

Rules:
0. Appointment/action keywords must override general_info.
   If the user says phrases like "وقت میخوام", "وقت می‌خوام", "نوبت میخوام", "نوبت می‌خوام",
   "میخوام وقت بگیرم", "می‌خوام وقت بگیرم", "رزرو وقت", "ثبت وقت", "برای ... وقت ... میخوام",
   classify as book_appointment, not general_info.
   This remains true even if the user mentions a service name like hair color, nails, consultation, massage, etc.

1. If the user asks about working hours, prices, address, services, rules, or general business info:
   intent = general_info
   needs_customer_identity = false
   requires_calendar = false
   requires_client_lookup = false
   is_general_question = true

2. If the user wants to book an appointment, reserve a time, get a turn, or says they want an appointment:
   intent = book_appointment
   needs_customer_identity = true
   requires_calendar = true
   requires_client_lookup = true
   is_general_question = false

3. If the user asks whether a time is available:
   intent = ask_availability
   requires_calendar = true
   If no booking/change/cancel is requested yet, needs_customer_identity can be false.

4. If the user wants to change or cancel an appointment:
   intent = change_appointment or cancel_appointment
   needs_customer_identity = true
   requires_calendar = true
   requires_client_lookup = true

5. Extract service, date_text, time_text, and datetime_text if present.
   Keep date_text/time_text as natural text from the user.
   Do NOT convert dates to Gregorian dates here.

6. If the message is ambiguous, set clarification_needed = true and write a short Persian clarification_question.

JSON schema:
{{
  "intent": "general_info",
  "confidence": 0.0,
  "needs_customer_identity": false,
  "service": "",
  "date_text": "",
  "time_text": "",
  "datetime_text": "",
  "is_general_question": false,
  "requires_calendar": false,
  "requires_client_lookup": false,
  "clarification_needed": false,
  "clarification_question": "",
  "summary": ""
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
            },
        },
        timeout=30,
    )

    response.raise_for_status()

    raw_text = response.json().get("response", "").strip()
    parsed = _safe_json_loads(raw_text)

    booking_keywords = [
        "وقت میخوام",
        "وقت می‌خوام",
        "نوبت میخوام",
        "نوبت می‌خوام",
        "وقت بگیرم",
        "نوبت بگیرم",
        "رزرو وقت",
        "ثبت وقت",
    ]

    if any(keyword in user_text for keyword in booking_keywords):
        parsed["intent"] = "book_appointment"
        parsed["needs_customer_identity"] = True
        parsed["requires_calendar"] = True
        parsed["requires_client_lookup"] = True
        parsed["is_general_question"] = False

    return _normalize_result(parsed)
