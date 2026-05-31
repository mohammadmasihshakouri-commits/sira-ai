from __future__ import annotations

from typing import Any


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _selected_slot_display(plan: dict[str, Any]) -> str:
    selected_slot = plan.get("selected_slot") or {}
    slot_selection = plan.get("slot_selection") or {}

    if not selected_slot and slot_selection:
        selected_slot = slot_selection.get("selected_slot") or {}

    return _safe_text(selected_slot.get("display"))


def build_receptionist_reply(plan: dict[str, Any]) -> dict[str, Any]:
    action = plan.get("action", "ask_clarification")
    reply_hint = _safe_text(plan.get("reply_hint"))

    if reply_hint:
        reply = reply_hint

    elif action == "answer_general_info":
        reply = "بفرمایید، سوالتون درباره چه موردیه؟"

    elif action == "smalltalk":
        reply = "در خدمتم، بفرمایید چطور می‌تونم کمک‌تون کنم؟"

    elif action == "ask_datetime_clarification":
        reply = "منظورتون صبحه یا عصر؟"

    elif action == "ask_slot_selection_clarification":
        reply = "لطفاً یکی از تایم‌های پیشنهادی رو انتخاب کنید."

    elif action == "collect_customer_identity":
        slot_display = _selected_slot_display(plan)

        if slot_display:
            reply = f"برای ثبت تایم {slot_display}، لطفاً نام و شماره موبایلتون رو بفرمایید."
        else:
            reply = "برای ثبت وقت، لطفاً نام و شماره موبایلتون رو بفرمایید."

    elif action == "check_calendar_availability":
        reply = "الان تایم‌های خالی رو بررسی می‌کنم."

    elif action == "lookup_existing_appointment":
        reply = "الان نوبت ثبت‌شده‌تون رو بررسی می‌کنم."

    elif action == "create_appointment_from_selected_slot":
        slot_display = _selected_slot_display(plan)

        if slot_display:
            reply = f"بسیار خوب، تایم {slot_display} رو برای ثبت نهایی آماده می‌کنم."
        else:
            reply = "بسیار خوب، این تایم رو برای ثبت نهایی آماده می‌کنم."

    elif action in {"take_message", "transfer_call"}:
        reply = "حتماً، لطفاً چند لحظه فرصت بدید."

    else:
        reply = "لطفاً بفرمایید دقیقاً چه کاری براتون انجام بدم؟"

    return {
        "reply": reply,
        "action": action,
        "plan": plan,
    }
