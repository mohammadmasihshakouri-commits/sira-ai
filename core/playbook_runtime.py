import re

from core.conversation_state import (
    WAITING_FOR_CONFIRMATION,
    WAITING_FOR_RESERVATION_CODE,
    WAITING_FOR_SEAT_LOCK_CONFIRMATION,
    POST_RESOLUTION_CHECK,
    RESOLVED,
    set_state,
)
from core.tools.order_tools import lookup_order
from core.intent_analyzer import analyze_intent
def extract_reservation_code(text):
    digits = re.findall(r"\d+", text)
    joined = "".join(digits)

    if len(joined) == 8:
        return joined

    return None

SEAT_LOCK_KEYWORDS = [
    "صندلی زرد",
    "زرد شده",
    "زرد شد",
    "صندلی قفل",
    "قفل شده",
    "صندلی قابل انتخاب نیست",
    "قابل انتخاب نیست",
    "نمیتونم صندلی انتخاب کنم",
    "نمی‌تونم صندلی انتخاب کنم",
    "صندلی انتخاب نمیشه",
    "صندلی آزاد نمیشه",
    "کیف پول",
    "شارژ کیف پول",
    "مبلغ کافی نداره",
    "موجودی کافی نداره",
]

SEAT_COLOR_CONFIRMED_KEYWORDS = [
    "زرد",
    "طلایی",
    "رنگ زرد",
    "رنگ طلایی",
]

THANKS_WORDS = [
    "ممنون",
    "مرسی",
    "تشکر",
    "سپاس",
]
def run_playbook_turn(user_text, conversation_state, enabled_playbooks):
    current_state = conversation_state.get("state")
    intent_result = analyze_intent(user_text, conversation_state)
    intent = intent_result.get("intent")
    if current_state == POST_RESOLUTION_CHECK:
        if intent == "thanks":
            return {
                "handled": True,
                "reply": (
                    "خواهش می‌کنم، وظیفه بود.\n"
                    "خوشحالم که تونستم راهنماییتون کنم.\n"
                    "مورد دیگه‌ای هست که بتونم راهنمایی‌تون کنم؟"
                ),
                "source": "Playbook Runtime / llm_thanks_after_resolution",
                "state": conversation_state,
            }

        if intent == "closing_no_more_questions":
            return {
                "handled": True,
                "reply": (
                    "خوشحالم که تونستم راهنماییتون کنم.\n"
                    "خوشحال میشم در نظرسنجی شرکت کنید 😊"
                ),
                "source": "Playbook Runtime / llm_close_after_resolution",
                "state": conversation_state,
            }

        if intent == "has_more_question":
            return {
                "handled": True,
                "reply": "بفرمایید، در خدمتم.",
                "source": "Playbook Runtime / llm_more_question_after_resolution",
                "state": conversation_state,
            }
        
    if current_state == WAITING_FOR_SEAT_LOCK_CONFIRMATION:
        positive_words = ["بله", "آره", "اره", "درسته", "زرد", "زرد شده", "همینه"]
        negative_words = ["نه", "خیر", "نه زرد نیست", "زرد نیست"]

        if any(word in user_text for word in positive_words):
            set_state(conversation_state, POST_RESOLUTION_CHECK)

            return {
                "handled": True,
                "reply": (
                    "این یعنی صندلی وارد حالت رزرو موقت شده. "
                    "معمولاً تا ۱۰ دقیقه بعد دوباره قابل انتخاب میشه."
                ),
                "source": "Playbook Runtime / seat_lock_confirmed",
                "topic": "seat_lock_issue",
                "state": conversation_state,
            }

        if any(word in user_text for word in negative_words):
            set_state(conversation_state, POST_RESOLUTION_CHECK)

            return {
                "handled": True,
                "reply": (
                    "متوجه شدم. اگر صندلی زرد نشده، احتمالاً مشکل از به‌روزرسانی صفحه یا وضعیت انتخاب صندلیه. "
                    "یک بار صفحه رو رفرش کنید و دوباره انتخاب رو بررسی کنید."
                ),
                "source": "Playbook Runtime / seat_lock_not_confirmed",
                "topic": "seat_selection_issue",
                "state": conversation_state,
            }

        return {
            "handled": True,
            "reply": "یعنی صندلی به رنگ زرد دراومده؟",
            "source": "Playbook Runtime / waiting_for_seat_lock_confirmation",
            "topic": "seat_lock_issue",
            "state": conversation_state,
        }
    if current_state == WAITING_FOR_CONFIRMATION:
        positive_words = ["بله", "آره", "درسته", "تایید", "تأیید", "همینه", "صحیحه"]
        negative_words = ["نه", "خیر", "اشتباه", "غلط", "نیست"]

        if any(word in user_text for word in positive_words):
            reservation_code = conversation_state.get("pending_value")

            order_result = lookup_order(reservation_code)

            if order_result["found"]:
                set_state(conversation_state, POST_RESOLUTION_CHECK)

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
        
    if any(keyword in user_text for keyword in SEAT_LOCK_KEYWORDS):
        if any(keyword in user_text for keyword in SEAT_COLOR_CONFIRMED_KEYWORDS):
            set_state(conversation_state, POST_RESOLUTION_CHECK)

            return {
                "handled": True,
                "reply": (
                    "بله، به این حالت رزرو موقت گفته میشه. "
                    "صندلی معمولاً تا ۱۰ دقیقه بعد دوباره قابل انتخاب میشه."
                ),
                "source": "Playbook Runtime / seat_lock_confirmed_direct",
                "topic": "seat_lock_issue",
                "state": conversation_state,
            }

        set_state(conversation_state, WAITING_FOR_SEAT_LOCK_CONFIRMATION)

        return {
            "handled": True,
            "reply": "یعنی صندلی به رنگ زرد دراومده؟",
            "source": "Playbook Runtime / seat_lock_clarification",
            "topic": "seat_lock_issue",
            "state": conversation_state,
        }
    if any(keyword in user_text for keyword in SEAT_LOCK_KEYWORDS):
        if any(keyword in user_text for keyword in SEAT_COLOR_CONFIRMED_KEYWORDS):
            set_state(conversation_state, POST_RESOLUTION_CHECK)

            return {
                "handled": True,
                "reply": (
                    "بله، به این حالت رزرو موقت گفته میشه. "
                    "صندلی معمولاً تا ۱۰ دقیقه بعد دوباره قابل انتخاب میشه."
                ),
                "source": "Playbook Runtime / seat_lock_confirmed_direct",
                "topic": "seat_lock_issue",
                "state": conversation_state,
            }

        set_state(conversation_state, WAITING_FOR_SEAT_LOCK_CONFIRMATION)

        return {
            "handled": True,
            "reply": "یعنی صندلی به رنگ زرد دراومده؟",
            "source": "Playbook Runtime / seat_lock_clarification",
            "topic": "seat_lock_issue",
            "state": conversation_state,
        }

    if "cancellation" in enabled_playbooks:
        if any(word in user_text for word in ["کنسل", "لغو", "استرداد"]):
            set_state(conversation_state, WAITING_FOR_RESERVATION_CODE)

            return {
                "handled": True,
                "reply": "حتماً. برای بررسی درخواست، لطفاً کد رزرو ۸ رقمی رو بفرمایید.",
                "source": "Playbook Runtime / cancellation_request",
                "topic": "cancellation",
                "state": conversation_state,
            }
    return {
        "handled": False,
        "reply": None,
        "source": None,
        "state": conversation_state,
    }