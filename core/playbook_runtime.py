import re

from core.conversation_state import (
    WAITING_FOR_RESERVATION_CODE,
    WAITING_FOR_CONFIRMATION,
    RESOLVED,
    set_state,
)
from core.tools.order_tools import lookup_order

def extract_reservation_code(text):
    digits = re.findall(r"\d+", text)
    joined = "".join(digits)

    if len(joined) == 8:
        return joined

    return None


def run_playbook_turn(user_text, conversation_state, enabled_playbooks):
    current_state = conversation_state.get("state")

    if current_state == WAITING_FOR_CONFIRMATION:
        positive_words = ["بله", "آره", "درسته", "تایید", "تأیید", "همینه", "صحیحه"]
        negative_words = ["نه", "خیر", "اشتباه", "غلط", "نیست"]

        if any(word in user_text for word in positive_words):
            reservation_code = conversation_state.get("pending_value")

            order_result = lookup_order(reservation_code)

            if order_result["found"]:
                set_state(conversation_state, RESOLVED)

                return {
                    "handled": True,
                    "reply": (
                        f"رزرو با کد {reservation_code} پیدا شد و درخواست کنسلی برای شما ثبت شد. "
                        f"نتیجه از طریق پیامک اطلاع‌رسانی میشه."
                    ),
                    "source": "Playbook Runtime / order_lookup_success",

                    "tool_call": {
                        "tool_name": "order_lookup",
                        "input": {
                            "reservation_code": reservation_code
                        },
                        "output": order_result
                    },
                    
                    "state": conversation_state,
                }

            return {
                "handled": True,
                "reply": "متأسفم، رزروی با این کد پیدا نشد.",
                "source": "Playbook Runtime / order_lookup_failed",
                "state": conversation_state,
            }

        if any(word in user_text for word in negative_words):
            conversation_state["pending_value"] = None
            conversation_state["pending_value_type"] = None

            set_state(conversation_state, WAITING_FOR_RESERVATION_CODE)

            return {
                "handled": True,
                "reply": "متوجه شدم. لطفاً کد رزرو صحیح رو دوباره بفرمایید.",
                "source": "Playbook Runtime / reservation_code_rejected",
                "state": conversation_state,
            }

        return {
            "handled": True,
            "reply": "لطفاً تأیید می‌کنید که کد رزرو گفته‌شده درسته؟",
            "source": "Playbook Runtime / waiting_for_confirmation",
            "state": conversation_state,
        }
    if current_state == WAITING_FOR_RESERVATION_CODE:
        reservation_code = extract_reservation_code(user_text)

        if reservation_code:
            conversation_state["pending_value"] = reservation_code
            conversation_state["pending_value_type"] = "reservation_code"
            set_state(conversation_state, WAITING_FOR_CONFIRMATION)

            return {
                "handled": True,
                "reply": f"کد رزرو {reservation_code} دریافت شد. لطفاً تأیید می‌کنید که همین کد درسته؟",
                "source": "Playbook Runtime / reservation_code_received",
                "state": conversation_state,
            }

        return {
            "handled": True,
            "reply": "لطفاً کد رزرو ۸ رقمی رو بفرمایید.",
            "source": "Playbook Runtime / waiting_for_reservation_code",
            "state": conversation_state,
        }

    if "cancellation" in enabled_playbooks:
        if any(word in user_text for word in ["کنسل", "لغو", "استرداد"]):
            set_state(conversation_state, WAITING_FOR_RESERVATION_CODE)

            return {
                "handled": True,
                "reply": "حتماً. برای بررسی درخواست، لطفاً کد رزرو ۸ رقمی رو بفرمایید.",
                "source": "Playbook Runtime / cancellation_request",
                "state": conversation_state,
            }

    return {
        "handled": False,
        "reply": None,
        "source": None,
        "state": conversation_state,
    }