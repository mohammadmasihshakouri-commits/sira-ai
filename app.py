import os
import time
import re
import random
import yaml
import tempfile
import subprocess
from core.runtime_config import (
    get_platform_name,
    get_workspace_name,
    get_agent_label,
    get_agent_greeting,
)
from core.normalization import normalize_text_light

from core.number_parser import (
    extract_identifier_candidate,
    normalize_requested_identifier,
    enrich_text_with_detected_number,
)

from core.playbook_engine import load_playbook

from core.state import (
    last_assistant_message,
    get_pending_state,
    set_pending_state,
    clear_pending_states,
    get_state_value,
    set_state_value,
    clear_state_value,
    get_audio_failure_count,
    set_audio_failure_count,
    reset_audio_failure_count,
)
# -----------------------------
# Fix CUDA DLL paths on Windows
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CUDA_DLL_PATHS = [
    os.path.join(BASE_DIR, "venv", "Lib", "site-packages", "nvidia", "cublas", "bin"),
    os.path.join(BASE_DIR, "venv", "Lib", "site-packages", "nvidia", "cudnn", "bin"),
    os.path.join(BASE_DIR, "venv", "Lib", "site-packages", "nvidia", "cuda_nvrtc", "bin"),
]

DLL_HANDLES = []

print("Checking CUDA DLL paths...")

for path in CUDA_DLL_PATHS:
    if os.path.isdir(path):
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        handle = os.add_dll_directory(path)
        DLL_HANDLES.append(handle)
        print("Added:", path)
    else:
        print("Not found:", path)

# -----------------------------
# Main imports
# -----------------------------

import requests
import gradio as gr
from faster_whisper import WhisperModel

# -----------------------------
# Settings
# -----------------------------

DEVICE = os.getenv("STT_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16")

WHISPER_MODEL_NAME = os.getenv(
    "WHISPER_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "faster-whisper-large-v3"),
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen-callcenter")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

MAX_AUDIO_CONFIRMATION_FAILURES = 3


PLATFORM_NAME = get_platform_name()
WORKSPACE_NAME = get_workspace_name()
AGENT_LABEL = get_agent_label()
AGENT_GREETING_FA = get_agent_greeting("fa")
AGENT_GREETING_EN = get_agent_greeting("en")

print("Loading Whisper model... Please wait.")
print(f"Device: {DEVICE}, Compute type: {COMPUTE_TYPE}, Model: {WHISPER_MODEL_NAME}")

stt_model = WhisperModel(
    WHISPER_MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=8,
    num_workers=1
)

print("Whisper model is ready.")

CALL_CENTER_STT_PROMPT = (
    "This is a Persian and English call center conversation. "
    "Transcribe exactly what the speaker says. "
    "Do not translate. Do not answer. Do not complete missing words. "

    "این یک مکالمه واقعی مرکز تماس فارسی و انگلیسی است. "
    "متن را دقیقاً همان‌طور که شنیده می‌شود بنویس. "
    "ترجمه نکن. پاسخ نده. جمله را کامل‌سازی نکن. "

    "کلمات رایج فارسی: سلام، جناب، خسته نباشید، وقتتون بخیر، بلیت، رزرو، کد رزرو، پرداخت، پیامک، کنسلی. "
    "در تشخیص عددها دقت کن، مخصوصاً کدهای ۸ رقمی که دو رقم دو رقم گفته می‌شوند."
)

def preprocess_audio_for_stt(audio_path):
    """
    Cleans and standardizes audio before Whisper:
    - mono
    - 16kHz
    - volume normalization
    - light noise reduction
    """

    if not audio_path or not os.path.exists(audio_path):
        return audio_path

    output_path = os.path.join(
        tempfile.gettempdir(),
        f"cinematicket_clean_{int(time.time() * 1000)}.wav"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-ac", "1",
        "-ar", "16000",
        "-af", "highpass=f=80,lowpass=f=7800,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
        output_path
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        if os.path.exists(output_path):
            return output_path

    except Exception as e:
        print("Audio preprocessing error:", e)

    return audio_path

# -----------------------------
# General helpers
# -----------------------------

def pick(options):
    return random.choice(options)

# -----------------------------
# Playbook helpers
# -----------------------------

def fast_playbook_reply(user_text, language, history):
    playbook = load_playbook()

    intents = playbook.get("intents", {})

    text = normalize_text_light(user_text)
    lower_text = text.lower()

    matches = []

    for intent_name, intent_data in intents.items():
        triggers = intent_data.get("triggers", [])
        priority = int(intent_data.get("priority", 0))

        for trigger in triggers:
            trigger_text = normalize_text_light(str(trigger))
            trigger_lower = trigger_text.lower()

            if not trigger_text:
                continue

            if trigger_text in text or trigger_lower in lower_text:
                matches.append((priority, intent_name, intent_data))
                break

    if not matches:
        return None, None

    matches.sort(key=lambda item: item[0], reverse=True)

    _, intent_name, intent_data = matches[0]

    responses = intent_data.get("response", {})
    response_list = responses.get(language) or responses.get("fa") or []

    if isinstance(response_list, str):
        response_list = [response_list]

    if not response_list:
        return None, None

    return intent_name, pick(response_list)


def get_playbook_domain_reply(user_text, language):
    playbook = load_playbook()

    domain_gate = playbook.get("domain_gate", {})

    allowed_topics = domain_gate.get("allowed_topics", [])
    responses = domain_gate.get("out_of_domain_response", {})

    text = normalize_text_light(user_text)
    lower_text = text.lower()

    for topic in allowed_topics:
        topic_text = normalize_text_light(str(topic))
        topic_lower = topic_text.lower()

        if topic_text and (topic_text in text or topic_lower in lower_text):
            return None

    response_list = responses.get(language) or responses.get("fa") or []

    if isinstance(response_list, str):
        response_list = [response_list]

    if not response_list:
        return None

    return pick(response_list)


def fast_playbook_slot_flow(user_text, language, history, detected_number):
    return None, None, history

    
def detect_language_simple(text, whisper_language=None):
    if whisper_language in ["en", "fa"]:
        return whisper_language

    english_chars = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    persian_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")

    if english_chars > persian_chars:
        return "en"

    return "fa"
def clean_text_by_language(text, language):
    if language == "en":
        replacements = {
            "پیامک": "SMS",
            "بلیت": "ticket",
            "بلیتم": "my ticket",
            "کد رزرو": "reservation code",
            "پرداخت": "payment",
        }

        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)

    return text


def get_last_user_language(history):
    """
    Finds the most likely previous user language from conversation memory.
    If there is no useful history, default to Persian for this call center demo.
    """

    for item in reversed(history):
        if item.get("role") == "UserLanguage":
            language = item.get("content", "")
            if language in ["fa", "en"]:
                return language

    return "fa"


def build_history_text(history):
    if not history:
        return ""

    lines = []

    for item in history[-10:]:
        role = item.get("role", "")
        content = item.get("content", "")

        if role and content and role not in ["SystemState", "UserLanguage"]:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def has_phone_or_tracking(text):
    normalized_digits = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
    digit_count = len(re.findall(r"\d", normalized_digits))
    return digit_count >= 6



# -----------------------------
# STT quality helpers
# -----------------------------

def looks_like_bad_stt(text):
    """
    Detects clearly broken STT output.
    If the transcription is repeated, meaningless, or too noisy,
    we should not send it to Ollama and should not save it in memory.
    """

    text = normalize_text_light(text)

    cleaned_short_text = re.sub(
        r"[^\w\u0600-\u06FF\s]",
        "",
        text
    ).strip().lower()

    safe_short_utterances = [
        "سلام",
        "درود",
        "hi",
        "hello",
    ]

    if cleaned_short_text in safe_short_utterances:
        return False

    if not text or len(text.strip()) < 3:
        return True

    tokens = re.findall(r"[\w\u0600-\u06FF]+", text)

    if len(tokens) < 2:
        return True

    if len(tokens) >= 8:
        unique_ratio = len(set(tokens)) / len(tokens)

        if unique_ratio < 0.35:
            return True

    for n in [2, 3, 4]:
        if len(tokens) >= n * 3:
            chunks = [
                " ".join(tokens[i:i + n])
                for i in range(len(tokens) - n + 1)
            ]

            for chunk in set(chunks):
                if chunks.count(chunk) >= 3:
                    return True

    bad_repeated_phrases = [
        "اینجا باشید",
        "اینجا هستید",
        "اینجا اینجا",
        "باشید باشید",
        "هستید هستید",
    ]

    for phrase in bad_repeated_phrases:
        if text.count(phrase) >= 2:
            return True

    return False


def unclear_stt_reply(language):
    """
    Human-like reply when the assistant could not understand the user's audio.
    """

    if language == "en":
        return pick([
            "Sorry, I couldn’t hear that clearly. Could you please repeat it?",
            "Sorry, the audio wasn’t clear for a moment. Could you say that again?",
            "I’m sorry, I didn’t fully understand that. Could you repeat it once more?",
        ])

    return pick([
        "ببخشید، صداتون درست نیومد. می‌تونید لطفاً یک بار دیگه بفرمایید؟",
        "ببخشید، کامل متوجه نشدم. می‌تونید لطفاً یک بار دیگه بفرمایید؟",
        "ببخشید، صدا یک لحظه واضح نبود. ممکنه دوباره بفرمایید؟",
        "عذر می‌خوام، جمله رو کامل متوجه نشدم. لطفاً یک بار دیگه می‌فرمایید؟",
    ])


def clean_persian_tone(reply, language):
    """
    Cleans unnatural Persian phrases generated by the LLM.
    """

    if language != "fa":
        return reply

    replacements = {
        "خبرتون برایم روشن شد": "متوجه شدم",
        "خبرتون روشن شد": "متوجه شدم",
        "برایم روشن شد": "متوجه شدم",
        "درخواست شما برایم روشن شد": "متوجه شدم",
        "درخواستتون برایم روشن شد": "متوجه شدم",
        "موضوع برایم روشن شد": "متوجه شدم",
        "مسئله برایم روشن شد": "متوجه شدم",

        "خبرتون برایم مهم است": "متوجه شدم",
        "خبرتون برام مهم است": "متوجه شدم",
        "شماره موبایل خرید": "شماره موبایل",
        "مشکلاتش رو حل کنم": "راهنمایی‌تون کنم",
        "مشکلش رو حل کنم": "راهنمایی‌تون کنم",

        "شماره موبایل شماره موبایلی": "شماره موبایلی",
        "شماره موبایل شماره موبایل": "شماره موبایل",
        "لطفاً شماره موبایل شماره موبایلی": "لطفاً شماره موبایلی",

        "خریدت رو بفرمایید": "شماره موبایلی که باهاش خرید انجام دادید یا کد رزرو رو بفرمایید",
        "شماره موبایلت رو": "شماره موبایلی که باهاش خرید انجام دادید",
        "بلیتت رو": "بلیت‌تون رو",
        "بلیطت رو": "بلیت‌تون رو",
        "کدت رو": "کدتون رو",
        "شماره‌تو": "شماره موبایل‌تون رو",
        "شمارتو": "شماره موبایل‌تون رو",
    }

    for wrong, correct in replacements.items():
        reply = reply.replace(wrong, correct)

    reply = reply.replace(
        "شماره موبایلی که باهاش خرید انجام دادید و یا",
        "شماره موبایلی که باهاش خرید انجام دادید یا"
    )

    reply = re.sub(r"\s+", " ", reply).strip()

    return reply

# -----------------------------
# Fast courtesy handler
# -----------------------------

def fast_courtesy_reply(user_text, language, history):
    """
    Handles pure greetings and courtesy phrases without calling Ollama.
    If the user starts with courtesy phrases like خسته نباشید / وقتتون بخیر,
    the assistant must thank them warmly and guide the conversation forward.
    """

    text = normalize_text_light(user_text)

    issue_keywords_fa = [
        "پرداخت",
        "بلیت",
        "بلیط",
        "پیامک",
        "استرداد",
        "کد رزرو",
        "سفارش",
        "پول",
        "تراکنش",
        "اپراتور",
        "پشتیبانی",
        "مشکل",
        "نیومده",
        "نرسیده",
        "ارسال نشده",
        "دریافت نکردم",
        "کم شده",
        "صداتون",
        "صدا",
        "تکرار",
        "دوباره",
        "صندلی",
        "رزرو موقت",
        "رزرو شده",
        "خالی نمیشه",
        "کیف پول",
        "شارژ کیف پول",
        "زرد",
        
        # Cancellation / refund wording
        "کنسلش",
        "کنسلش کنم",
        "کانسل",
        "کانسلش",
        "کانسلش کنم",
        "کنسل",
        "کنسلی",
        "لغو",
        "عودت",
        "مرجوع",
        "تیوال",
        "باطل",
        "پسش بدم",
        "پسش بدن",
        "پس بدم",
        "پس بدن",
        "می‌خوام پسش بدم",
        "می‌خوام پسش بدن",
        "میخوام پسش بدم",
        "میخوام پسش بدن",
    ]

    issue_keywords_en = [
        "payment",
        "ticket",
        "sms",
        "refund",
        "reservation code",
        "order",
        "money",
        "transaction",
        "operator",
        "support",
        "problem",
        "not received",
        "didn't receive",
        "did not receive",
        "charged",
        "repeat",
        "hear",
        "audio",
        "voice",
        "cancel",
        "cancellation",
        "expired",
    ]

    greeting_fa = [
        "سلام",
        "درود",
        "وقتتون بخیر",
        "عصرتون بخیر",
        "عصرتون بخیر باشه",
        "صبحتون بخیر",
        "صبحتون بخیر باشه",
        "شبتون بخیر",
        "شبتون بخیر باشه",
        "خسته نباشید",
    ]

    courtesy_phrases_fa = [
        "خسته نباشید",
        "وقتتون بخیر",
        "عصرتون بخیر",
        "عصرتون بخیر باشه",
        "صبحتون بخیر",
        "صبحتون بخیر باشه",
        "شبتون بخیر",
        "شبتون بخیر باشه",
    ]

    greeting_en = [
        "hi",
        "hello",
        "good morning",
        "good evening",
        "good afternoon",
    ]

    if language == "fa":
        has_greeting = any(phrase in text for phrase in greeting_fa)
        has_courtesy_phrase = any(phrase in text for phrase in courtesy_phrases_fa)
        has_issue = any(keyword in text for keyword in issue_keywords_fa)

        if has_greeting and not has_issue:
            if has_courtesy_phrase:
                return "greeting_with_courtesy", pick([
                    "سلام، ممنونم ازتون. بفرمایید چطور می‌تونم کمک‌تون کنم؟",
                    "سلام، خیلی ممنون. بفرمایید چطور می‌تونم راهنمایی‌تون کنم؟",
                    "سلام، لطف دارید. بفرمایید چه کمکی از دستم برمیاد؟",
                    "سلام، ممنون از لطفتون. بفرمایید چطور می‌تونم راهنمایی‌تون کنم؟",
                    "سلام، ممنونم. چطور می‌تونم کمک‌تون کنم؟",
                ])

            return "greeting", pick([
                "سلام، بفرمایید چطور می‌تونم کمک‌تون کنم؟",
                "سلام، چطور می‌تونم راهنمایی‌تون کنم؟",
                "سلام، بفرمایید چه کمکی از دستم برمیاد؟",
                "سلام، در خدمتم. چطور می‌تونم کمک‌تون کنم؟",
            ])

    else:
        lower_text = text.lower()
        has_greeting = any(phrase in lower_text for phrase in greeting_en)
        has_issue = any(keyword in lower_text for keyword in issue_keywords_en)

        if has_greeting and not has_issue:
            return "greeting", pick([
                "Hello, how can I help you today?",
                "Hi, how can I help you?",
                "Hello, what can I help you with?",
                "Hi, how can I guide you today?",
            ])

    return None, None

# -----------------------------
# Fast seat temporary reservation handler
# -----------------------------

def fast_seat_reserved_issue_reply(user_text, language, history):
    """
    Handles the common scenario where a customer selected a seat,
    left the purchase flow to charge wallet/payment, and the seat
    remains temporarily reserved / yellow / unavailable.
    """

    text = normalize_text_light(user_text)
    lower_text = text.lower()

    seat_keywords_fa = [
        "صندلی",
        "جایگاه",
        "صندلیش",
        "صندلی رو",
        "صندلیمو",
    ]

    reserved_keywords_fa = [
        "رزرو",
        "رزرو شده",
        "رزرو موقت",
        "خالی نمیشه",
        "آزاد نمیشه",
        "انتخاب نمیشه",
        "نمیتونم انتخاب کنم",
        "نمی‌تونم انتخاب کنم",
        "زرد",
        "زرده",
    ]

    payment_context_fa = [
        "کیف پول",
        "شارژ",
        "موجودی نداشت",
        "پول",
        "پرداخت",
        "خارج شدم",
        "اومدم بیرون",
    ]

    seat_keywords_en = [
        "seat",
        "chair",
    ]

    reserved_keywords_en = [
        "reserved",
        "temporary reserved",
        "unavailable",
        "yellow",
        "not available",
        "can't select",
        "cannot select",
    ]

    if language == "en":
        has_seat = any(keyword in lower_text for keyword in seat_keywords_en)
        has_reserved = any(keyword in lower_text for keyword in reserved_keywords_en)

        if has_seat and has_reserved:
            return "seat_reserved_issue", (
                "If the seat is temporarily reserved or unavailable, please wait about 10 minutes "
                "for it to be released, then try selecting the same seat again."
            )

        return None, None

    has_seat = any(keyword in text for keyword in seat_keywords_fa)
    has_reserved = any(keyword in text for keyword in reserved_keywords_fa)
    has_payment_context = any(keyword in text for keyword in payment_context_fa)

    if has_seat and has_reserved:
        if has_payment_context:
            return "seat_reserved_after_wallet_charge", pick([
                "بله، متوجه شدم. اگر برای شارژ کیف پول از خرید خارج شدید و صندلی به رنگ زرد یا همون حالت رزرو موقت نمایش داده می‌شه، لطفاً حدود ۱۰ دقیقه صبر کنید تا از حالت رزرو خارج بشه، بعد دوباره می‌تونید همون صندلی رو انتخاب کنید.",
                "اگر صندلی به رنگ زرد یا همون حالت رزرو موقت نمایش داده می‌شه، لطفاً حدود ۱۰ دقیقه صبر کنید تا از حالت رزرو خارج بشه، بعد دوباره می‌تونید همون صندلی رو انتخاب کنید.",
                "بله، این حالت معمولاً رزرو موقته. لطفاً حدود ۱۰ دقیقه صبر کنید تا صندلی از حالت رزرو خارج بشه، بعد دوباره می‌تونید همون صندلی رو انتخاب کنید.",
                "متوجه شدم. وقتی صندلی به رنگ زرد یا همون حالت رزرو موقت نمایش داده می‌شه، معمولاً باید حدود ۱۰ دقیقه صبر کنید تا از حالت رزرو خارج بشه و بعد مجدد انتخابش کنید.",
            ])

        return "seat_reserved_issue", pick([
            "اگر صندلی به رنگ زرد یا همون حالت رزرو موقت نمایش داده می‌شه، لطفاً حدود ۱۰ دقیقه صبر کنید تا از حالت رزرو خارج بشه، بعد دوباره می‌تونید همون صندلی رو انتخاب کنید.",
            "بله، اگر صندلی هنوز رزرو نشون داده می‌شه، حدود ۱۰ دقیقه صبر کنید تا از حالت رزرو موقت خارج بشه و بعد دوباره امتحان کنید.",
        ])

    return None, None
# -----------------------------
# Fast ticket / SMS issue handler
# -----------------------------

def fast_ticket_sms_issue_reply(user_text, language, history):
    """
    Handles clear ticket/payment/SMS missing cases without calling Ollama.
    """

    text = normalize_text_light(user_text)
    lower_text = text.lower()

    payment_fa = [
        "پرداخت",
        "پرداختی",
        "پول",
        "تراکنش",
        "کسر شده",
        "کم شده",
        "خرید کردم",
        "خرید انجام دادم",
    ]

    ticket_fa = [
        "بلیت",
        "بلیتم",
        "بلیط",
        "بلیطم",
        "تیکت",
        "اطلاعات بلیت",
        "اطلاعاتش",
    ]

    missing_fa = [
        "پیامک نشده",
        "پیامک نشد",
        "پیامک نشده بود",
        "برام پیامک نشده",
        "پیامکش نیومده",
        "نیومده",
        "نرسیده",
        "ارسال نشده",
        "دریافت نکردم",
        "هنوز نیومده",
        "هنوز نرسیده",
    ]

    payment_en = [
        "paid",
        "payment",
        "charged",
        "transaction",
        "purchase",
    ]

    ticket_en = [
        "ticket",
        "booking",
        "reservation",
    ]

    missing_en = [
        "sms",
        "message",
        "not received",
        "didn't receive",
        "did not receive",
        "haven't received",
        "hasn't arrived",
    ]

    if language == "fa":
        has_payment = any(x in text for x in payment_fa)
        has_ticket = any(x in text for x in ticket_fa)
        has_missing = any(x in text for x in missing_fa)

        if has_payment and has_ticket and has_missing:
            return "payment_ticket_sms_issue", pick([
                "بله، متوجه شدم. لطفاً شماره موبایل یا کد رزرو رو بفرمایید.",
                "حتماً، برای بررسی لطفاً شماره موبایل یا کد رزرو رو بفرمایید.",
                "درسته، لطفاً شماره موبایل یا کد رزرو رو بفرمایید تا بررسی بشه.",
            ])

        if has_ticket and has_missing:
            return "ticket_sms_issue", pick([
                "بله، متوجه شدم. لطفاً شماره موبایل یا کد رزرو رو بفرمایید.",
                "حتماً، برای بررسی لطفاً شماره موبایل یا کد رزرو رو بفرمایید.",
            ])

    else:
        has_payment = any(x in lower_text for x in payment_en)
        has_ticket = any(x in lower_text for x in ticket_en)
        has_missing = any(x in lower_text for x in missing_en)

        if has_payment and has_ticket and has_missing:
            return "payment_ticket_sms_issue", pick([
                "I understand. Please provide your phone number or reservation code so this can be checked.",
                "Sure, please provide your phone number or reservation code.",
            ])

        if has_ticket and has_missing:
            return "ticket_sms_issue", pick([
                "I understand. Please provide your phone number or reservation code.",
                "Sure, please provide your phone number or reservation code so this can be checked.",
            ])

    return None, None
# -----------------------------
# Fast refund / cancellation handler
# -----------------------------

def fast_refund_cancellation_reply(user_text, language, history):
    """
    Handles frequent refund/cancellation scenarios without calling Ollama.
    Important: never claim that cancellation/refund was completed.
    """

    text = normalize_text_light(user_text)
    lower_text = text.lower()

    has_persian_chars = any("\u0600" <= ch <= "\u06FF" for ch in text)

    latin_cancel_words = [
        "cancel",
        "cancellation",
        "refund",
        "return my money",
        "money back",
    ]

    if language == "fa" and not has_persian_chars:
        if not any(word in lower_text for word in latin_cancel_words):
            return None, None
        
    cancel_fa = [            
        "کنسل",
        "کنسلی",
        "کنسلش",
        "کنسلش کنم",
        "می‌خوام کنسلش کنم",
        "میخوام کنسلش کنم",

        "پسش بدهم",
        "می‌خوام پسش بدهم",
        "میخوام پسش بدهم",
        "بلیت رو پس بدهم",
        "بلیتم رو پس بدهم",

        "کانسل",
        "کانسلش",
        "کانسلش کنم",
        "می‌خوام کانسلش کنم",
        "میخوام کانسلش کنم",

        "پستش بدم",
        "پستش بدهم",
        "می‌خوام پستش بدم",
        "میخوام پستش بدم",

        "لغو",
        "استرداد",
        "عودت",
        "مرجوع",

        "پولم برگرده",
        "پولم رو برگردونید",
        "پول بلیتم برگرده",
        "پول بلیت رو برگردونید",

        "بلیت رو پس بدم",
        "بلیتم رو پس بدم",
        "پسش بدم",
        "پسش بدن",
        "پس بدم",
        "پس بدن",
        "میخوام پسش بدم",
        "می‌خوام پسش بدم",
        "میخوام پسش بدن",
        "می‌خوام پسش بدن",
        "میخوام بلیتم رو پس بدم",
        "می‌خوام بلیتم رو پس بدم",
    ]

    cancel_en = [
        "cancel",
        "cancellation",
        "refund",
        "return my money",
        "money back",
    ]

    reservation_code_where_fa = [
        "کد رزرو کجاست",
        "کد رزرو رو از کجا",
        "کد رزرو را از کجا",
        "کد رزرو ندارم",
        "کد رزرو کجا نوشته",
        "کد رزرو کجا هست",
        "از کجا کد رزرو",
        "کد رزرو رو کجا ببینم",
        "کد رزرو را کجا ببینم",
    ]

    reservation_code_where_en = [
        "where is the reservation code",
        "where can i find the reservation code",
        "i don't have the reservation code",
        "i do not have the reservation code",
        "where is my reservation code",
    ]

    tiwall_fa = [
        "تیوال",
        "tiwall",
        "tival",
    ]

    less_than_two_hours_fa = [
        "کمتر از دو ساعت",
        "کمتر از ۲ ساعت",
        "زیر دو ساعت",
        "دو ساعت مونده",
        "یک ساعت مونده",
        "یه ساعت مونده",
        "نیم ساعت مونده",
        "الان شروع میشه",
        "الان شروع می‌شه",
        "سانس شروع شده",
        "اکران شروع شده",
    ]

    less_than_two_hours_en = [
        "less than two hours",
        "less than 2 hours",
        "under two hours",
        "one hour left",
        "half an hour left",
        "already started",
        "show has started",
    ]

    wrong_date_or_expired_fa = [
        "تاریخ رو اشتباه",
        "تاریخش رو اشتباه",
        "فکر کردم فرداست",
        "فکر می‌کردم فرداست",
        "دیروز بود",
        "سانس گذشته",
        "اکران گذشته",
        "بلیت باطل شده",
        "قبول نکردن",
        "قبول نکردند",
    ]

    wrong_date_or_expired_en = [
        "wrong date",
        "thought it was tomorrow",
        "show was yesterday",
        "showtime passed",
        "ticket expired",
        "they did not accept it",
        "they didn't accept it",
    ]

    if language == "fa":
        has_cancel = any(x in text for x in cancel_fa)
        asks_reservation_code_where = any(x in text for x in reservation_code_where_fa)
        has_tiwall = any(x in lower_text for x in tiwall_fa)
        has_less_than_two_hours = any(x in text for x in less_than_two_hours_fa)
        has_wrong_date_or_expired = any(x in text for x in wrong_date_or_expired_fa)

        if asks_reservation_code_where:
            return "reservation_code_where", pick([
                "حتماً. از بالای سایت روی اسم حساب کاربری‌تون بزنید، وارد «بلیت‌های من» بشید، بعد روی «جزئیات بلیت» بزنید. کد رزرو پایین صفحه‌ی جزئیات، بالای بارکد نمایش داده می‌شه.",
                "کد رزرو داخل جزئیات بلیته. از بالای سایت روی اسم‌تون بزنید، وارد «بلیت‌های من» بشید، روی «جزئیات بلیت» بزنید؛ پایین صفحه، بالای بارکد، کد رزرو رو می‌بینید.",
            ])

        if has_tiwall and has_cancel:
            return "tiwall_cancellation_not_available", pick([
                "متأسفانه طبق قوانین، بلیت‌های تیوال قابل کنسلی نیستن.",
                "متأسفانه بلیت‌های تیوال طبق قوانین امکان کنسلی ندارن.",
            ])

        if has_less_than_two_hours and has_cancel:
            return "less_than_two_hours_cancellation_not_allowed", pick([
                "متأسفانه طبق قوانین، وقتی کمتر از دو ساعت به اکران مونده باشه امکان کنسلی وجود نداره.",
                "متأسفانه برای سانس‌هایی که کمتر از دو ساعت تا شروعشون مونده، امکان کنسلی وجود نداره.",
            ])

        if has_wrong_date_or_expired:
            return "expired_or_wrong_date_ticket", pick([
                "متأسفانه وقتی زمان اکران گذشته و بلیت باطل شده باشه، امکان کنسلی یا استرداد وجود نداره.",
                "متأسفانه اگر سانس گذشته باشه و بلیت باطل شده باشه، امکان پیگیری کنسلی وجود نداره.",
            ])

        if has_cancel:
            set_pending_state(history, "waiting_for_reservation_code_for_cancellation")

        return "general_cancellation_request", pick([
    "حتماً، لطفاً کد رزرو ۸ رقمی رو دو رقم دو رقم بفرمایید.",
    "حتماً، لطفاً کد رزروتون رو دو رقم دو رقم بگید.",
    "بله، برای راهنمایی کنسلی لطفاً کد رزرو ۸ رقمی رو بفرمایید.",
])

    else:
        has_cancel = any(x in lower_text for x in cancel_en)
        asks_reservation_code_where = any(x in lower_text for x in reservation_code_where_en)
        has_tiwall = any(x in lower_text for x in tiwall_fa)
        has_less_than_two_hours = any(x in lower_text for x in less_than_two_hours_en)
        has_wrong_date_or_expired = any(x in lower_text for x in wrong_date_or_expired_en)

        if asks_reservation_code_where:
            return "reservation_code_where", (
                "Sure. Click your account name at the top of the site, go to “My Tickets”, then open “Ticket Details”. The reservation code is near the bottom of the ticket details page, above the barcode."
            )

        if has_tiwall and has_cancel:
            return "tiwall_cancellation_not_available", (
                "Unfortunately, according to the rules, Tiwall tickets can’t be cancelled."
            )

        if has_less_than_two_hours and has_cancel:
            return "less_than_two_hours_cancellation_not_allowed", (
                "Unfortunately, according to the rules, cancellation isn’t available when less than two hours remain before the show."
            )

        if has_wrong_date_or_expired:
            return "expired_or_wrong_date_ticket", (
                "Unfortunately, if the showtime has already passed and the ticket is expired, cancellation or refund isn’t available."
            )

        if has_cancel:
            set_pending_state(history, "waiting_for_reservation_code_for_cancellation")
            
            return "general_cancellation_request", (
                "Sure. Please provide your reservation code so I can guide you."
            )

    return None, None
def fast_cancellation_reservation_code_flow(user_text, language, history, detected_number):
    """
    Handles reservation code confirmation for cancellation requests.
    Reservation code must be exactly 8 digits.
    This is separate from phone/tracking number confirmation.
    """

    text = normalize_text_light(user_text)
    lower_text = text.lower()

    pending_state = get_pending_state(history)

    positive_fa = [
        "بله",
        "آره",
        "اره",
        "درسته",
        "همینه",
        "همین درسته",
        "تایید",
        "تأیید",
        "تأیید می‌کنم",
        "تایید میکنم",
        "تأیید میکنم",
        "صحیحه",
    ]

    negative_fa = [
        "نه",
        "خیر",
        "اشتباهه",
        "اشتباه",
        "درست نیست",
        "نه این نیست",
        "این نیست",
    ]

    positive_en = [
        "yes",
        "yeah",
        "correct",
        "that's right",
        "that is right",
        "confirmed",
    ]

    negative_en = [
        "no",
        "wrong",
        "incorrect",
        "that's wrong",
        "that is wrong",
        "not correct",
    ]

    tiwall_fa = [
        "تیوال",
        "tiwall",
        "tival",
    ]
    recent_assistant_text = " ".join([
        item.get("content", "")
        for item in history[-6:]
        if item.get("role") == "Assistant"
    ])

    asked_for_reservation_code = any(phrase in recent_assistant_text for phrase in [
        "کد رزرو",
        "reservation code",
    ])
    if pending_state == "waiting_for_reservation_code_for_cancellation" or asked_for_reservation_code:
        # If user says it is a Tiwall ticket, answer the rule directly.
        if any(x in lower_text for x in tiwall_fa):
            history[:] = clear_pending_states(history)
            clear_state_value(history, "pending_reservation_code")

            if language == "en":
                return (
                    "tiwall_cancellation_not_available",
                    "Unfortunately, according to the rules, Tiwall tickets can’t be cancelled.",
                    history
                )

            return (
                "tiwall_cancellation_not_available",
                "متأسفانه طبق قوانین، بلیت‌های تیوال قابل کنسلی نیستن.",
                history
            )

        pending_reservation_code = get_state_value(history, "pending_reservation_code", "")

        # If user gives a new reservation code while we are waiting for confirmation,
        # treat it as a correction/new code.
        if detected_number:
            reservation_code = re.sub(r"\D", "", detected_number)

            if len(reservation_code) != 8:
                if language == "en":
                    return (
                        "reservation_code_invalid_length",
                        "The reservation code should be 8 digits. Please say the 8-digit reservation code one more time.",
                        history
                    )

                return (
                    "reservation_code_invalid_length",
                    "کد رزرو باید ۸ رقمی باشه. لطفاً کد رزرو ۸ رقمی رو یک بار دیگه بفرمایید.",
                    history
                )

            set_state_value(history, "pending_reservation_code", reservation_code)

            if language == "en":
                return (
                    "reservation_code_detected_ask_confirmation",
                    f"Okay, so the reservation code is {reservation_code}. Is that right?",
                    history
                )

            return (
                "reservation_code_detected_ask_confirmation",
                f"بله، پس کد رزرو شد {reservation_code}. درسته؟",
                history
            )

        if language == "fa":
            is_negative = any(phrase in text for phrase in negative_fa)
            is_positive = any(phrase in text for phrase in positive_fa)

            if is_negative:
                clear_state_value(history, "pending_reservation_code")

                return (
                    "reservation_code_rejected",
                    "ببخشید، پس لطفاً کد رزرو ۸ رقمی رو یک بار دیگه بفرمایید.",
                    history
                )

            if is_positive and pending_reservation_code:
                history[:] = clear_pending_states(history)
                clear_state_value(history, "pending_reservation_code")

                return (
                    "reservation_code_confirmed",
                    "ممنونم، کد رزرو ثبت شد. بعد از بررسی سفارش، وضعیت امکان کنسلی مشخص می‌شه.",
                    history
                )

        else:
            is_negative = any(phrase in lower_text for phrase in negative_en)
            is_positive = any(phrase in lower_text for phrase in positive_en)

            if is_negative:
                clear_state_value(history, "pending_reservation_code")

                return (
                    "reservation_code_rejected",
                    "Sorry about that. Please say the 8-digit reservation code one more time.",
                    history
                )

            if is_positive and pending_reservation_code:
                history[:] = clear_pending_states(history)
                clear_state_value(history, "pending_reservation_code")

                return (
                    "reservation_code_confirmed",
                    "Thanks, the reservation code is saved. After checking the order, the cancellation status can be confirmed.",
                    history
                )

    return None, None, history
# -----------------------------
# Number confirmation flow
# -----------------------------

def fast_number_confirmation_flow(user_text, language, history, detected_number):
    """
    Handles phone number / reservation code confirmation.
    Cinematicket identifiers:
    - 11 digits -> phone
    - 10 digits starting with 9 -> phone with leading zero
    - 8 digits -> reservation code
    """

    text = normalize_text_light(user_text)
    lower_text = text.lower()

    pending_state = get_pending_state(history)

    positive_fa = [
        "بله",
        "آره",
        "اره",
        "درسته",
        "درست",
        "همینه",
        "همین درسته",
        "تایید",
        "تأیید",
        "صحیحه",
    ]

    negative_fa = [
        "نه",
        "خیر",
        "اشتباهه",
        "اشتباه",
        "درست نیست",
        "نه این نیست",
        "این نیست",
    ]

    positive_en = [
        "yes",
        "yeah",
        "correct",
        "that's right",
        "that is right",
        "right",
        "confirmed",
        "ben de durustum",
        "bende durustum",
        "durustum",
        "dorustum",
        "doğru",
        "dogru",
        "evet",
    ]

    negative_en = [
        "no",
        "wrong",
        "incorrect",
        "that's wrong",
        "that is wrong",
        "not correct",
    ]

    if pending_state == "waiting_for_number_confirmation":
        pending_number_type = get_state_value(history, "pending_number_type", "reservation_code")

        is_negative = (
            any(phrase in text for phrase in negative_fa)
            or any(phrase in lower_text for phrase in negative_en)
        )

        is_positive = (
            any(phrase in text for phrase in positive_fa)
            or any(phrase in lower_text for phrase in positive_en)
        )

        if is_negative:
            history[:] = clear_pending_states(history)
            clear_state_value(history, "pending_number")
            clear_state_value(history, "pending_number_type")

            if language == "en":
                reply = "Sorry about that. Please say the phone number or reservation code one more time."
            else:
                reply = "ببخشید، پس لطفاً شماره موبایل یا کد رزرو رو یک بار دیگه بفرمایید."

            return (
                "identifier_rejected",
                reply,
                history
            )

        if is_positive:
            history[:] = clear_pending_states(history)
            clear_state_value(history, "pending_number")
            clear_state_value(history, "pending_number_type")

            if language == "en":
                if pending_number_type == "phone":
                    reply = "Thanks, the phone number is saved. To continue, this needs to be checked in the order system."
                else:
                    reply = "Thanks, the reservation code is saved. To continue, this needs to be checked in the order system."
            else:
                if pending_number_type == "phone":
                    reply = "ممنونم، شماره موبایل ثبت شد. برای ادامه باید این مورد در سیستم سفارش‌ها بررسی بشه."
                else:
                    reply = "ممنونم، کد رزرو ثبت شد. برای ادامه باید این مورد در سیستم سفارش‌ها بررسی بشه."

            return (
                "identifier_confirmed",
                reply,
                history
            )

    if detected_number:
        recent_assistant_text = " ".join([
            item.get("content", "")
            for item in history[-6:]
            if item.get("role") == "Assistant"
        ])

        asked_for_number = any(phrase in recent_assistant_text for phrase in [
            "شماره موبایل",
            "شماره موبایلی",
            "کد رزرو",
            "phone number",
            "reservation code",
        ])

        if asked_for_number:
            normalized_identifier, identifier_type = normalize_requested_identifier(detected_number)

            if identifier_type == "invalid_identifier":

                asked_only_for_reservation_code = (
                    "کد رزرو" in last_assistant_message(history)
                )

                if asked_only_for_reservation_code:

                    if language == "en":
                        return (
                            "invalid_reservation_code",
                            "The reservation code should be 8 digits. Please say the 8-digit reservation code one more time.",
                            history
                        )

                    return (
                        "invalid_reservation_code",
                        "کد رزرو باید ۸ رقمی باشه. لطفاً کد رزرو ۸ رقمی رو یک بار دیگه بفرمایید.",
                        history
                    )

                if language == "en":
                    return (
                        "invalid_identifier",
                        "Please provide either an 11-digit phone number or an 8-digit reservation code.",
                        history
                    )

                return (
                    "invalid_identifier",
                    "لطفاً شماره موبایل ۱۱ رقمی یا کد رزرو ۸ رقمی رو بفرمایید.",
                    history
                )

            set_pending_state(history, "waiting_for_number_confirmation")
            set_pending_state(history, "waiting_for_number_confirmation")
            set_state_value(history, "pending_number", normalized_identifier)
            set_state_value(history, "pending_number_type", identifier_type)

            if language == "en":
                if identifier_type == "phone":
                    return (
                        "phone_detected_ask_confirmation",
                        pick([
                            f"Okay, so the phone number is {normalized_identifier}. Is that right?",
                            f"Okay, so {normalized_identifier}; is that the correct phone number?",
                        ]),
                        history
                    )

                return (
                    "reservation_code_detected_ask_confirmation",
                    pick([
                        f"Okay, so the reservation code is {normalized_identifier}. Is that right?",
                        f"Okay, so {normalized_identifier}; is that the correct reservation code?",
                    ]),
                    history
                )

            if identifier_type == "phone":
                return (
                    "phone_detected_ask_confirmation",
                    pick([
                        f"بله، پس شماره شد {normalized_identifier}. درسته؟",
                        f"بله، پس {normalized_identifier}؛ همین شماره درسته؟",
                    ]),
                    history
                )

            return (
                "reservation_code_detected_ask_confirmation",
                pick([
                    f"بله، پس کد رزرو شد {normalized_identifier}. درسته؟",
                    f"بله، پس {normalized_identifier}؛ همین کد رزرو درسته؟",
                ]),
                history
            )

    return None, None, history
# -----------------------------
# Fast audio / repeat flow
# -----------------------------

def fast_audio_issue_flow(user_text, language, history):
    """
    Handles repeat/audio issues separately.
    """

    text = normalize_text_light(user_text)
    lower_text = text.lower()

    pending_state = get_pending_state(history)

    audio_connection_issue_fa = [
        "صداتون قطع شد",
        "صدا قطع شد",
        "صداتون نیومد",
        "صدا نیومد",
        "نشنیدم",
        "واضح نبود",
        "صدا واضح نبود",
        "صداتون واضح نبود",
        "صدا مشکل داشت",
        "صدا نمیاد",
        "صداتون نمیاد",
    ]

    repeat_request_fa = [
        "دوباره بفرمایید",
        "دوباره می‌گید",
        "دوباره میگید",
        "تکرار می‌کنید",
        "تکرار میکنید",
        "تکرار کنید",
        "یه بار دیگه",
        "یک بار دیگه",
        "می‌شه دوباره بگید",
        "میشه دوباره بگید",
    ]

    audio_connection_issue_en = [
        "your voice cut off",
        "audio cut off",
        "i couldn't hear",
        "i could not hear",
        "i didn't hear",
        "i did not hear",
        "your voice was unclear",
        "the audio was unclear",
        "i can't hear you",
        "i cannot hear you",
    ]

    repeat_request_en = [
        "can you repeat",
        "please repeat",
        "say that again",
        "could you repeat",
        "one more time",
    ]

    positive_fa = [
        "بله",
        "آره",
        "اره",
        "دارم",
        "می‌شنوم",
        "میشنوم",
        "صداتون رو دارم",
        "الان دارم",
        "الان می‌شنوم",
        "واضحه",
        "صداتون واضحه",
    ]

    negative_fa = [
        "نه",
        "خیر",
        "نه هنوز",
        "هنوز نه",
        "ندارم",
        "نمی‌شنوم",
        "نمیشنوم",
        "هنوز قطع",
        "واضح نیست",
        "صدا ندارم",
        "صداتون رو ندارم",
    ]

    positive_en = [
        "yes",
        "yeah",
        "i can hear",
        "i hear you",
        "now i can hear",
        "yes i can",
        "it's clear",
        "its clear",
    ]

    negative_en = [
        "no",
        "not yet",
        "still no",
        "i can't hear",
        "i cannot hear",
        "still can't hear",
        "not clear",
        "it's not clear",
        "its not clear",
    ]

    if pending_state == "waiting_for_audio_confirmation":
        if language == "fa":
            is_negative = any(phrase in text for phrase in negative_fa)
            is_positive = any(phrase in text for phrase in positive_fa)

            if is_negative:
                count = get_audio_failure_count(history) + 1
                set_audio_failure_count(history, count)

                if count > MAX_AUDIO_CONFIRMATION_FAILURES:
                    history[:] = clear_pending_states(history)
                    reset_audio_failure_count(history)
                    return "audio_issue_handoff", "متأسفم، به نظر می‌رسه هنوز مشکل صدا برطرف نشده. برای رسیدگی بهتر، شما رو به پشتیبانی وصل می‌کنم.", history

                reply = pick([
                    "الان دوباره چک کنیم؛ صدای من رو دارید؟",
                    "باشه، دوباره بررسی می‌کنیم. الان صدام واضح میاد؟",
                    "الان صدا رو واضح دارید؟",
                    "الان بهتر صدام رو می‌شنوید؟",
                ])

                return "audio_issue_retry", reply, history

            if is_positive:
                reset_audio_failure_count(history)
                history[:] = clear_pending_states(history)

                assistant_messages = [
                    item.get("content", "")
                    for item in history
                    if item.get("role") == "Assistant"
                ]

                previous_reply = ""
                if len(assistant_messages) >= 2:
                    previous_reply = assistant_messages[-2]
                elif len(assistant_messages) == 1:
                    previous_reply = assistant_messages[-1]

                if previous_reply:
                    return "audio_confirmed_repeat", f"ممنونم. دوباره عرض می‌کنم: {previous_reply}", history

                return "audio_confirmed_no_previous", "ممنونم. بفرمایید چطور می‌تونم کمک‌تون کنم؟", history

        else:
            is_negative = any(phrase in lower_text for phrase in negative_en)
            is_positive = any(phrase in lower_text for phrase in positive_en)

            if is_negative:
                count = get_audio_failure_count(history) + 1
                set_audio_failure_count(history, count)

                if count > MAX_AUDIO_CONFIRMATION_FAILURES:
                    history[:] = clear_pending_states(history)
                    reset_audio_failure_count(history)
                    return "audio_issue_handoff", "I’m sorry, it seems the audio issue is still not resolved. I’ll connect you to support for better assistance.", history

                reply = pick([
                    "Can you hear me now?",
                    "Let’s check again. Is my voice clear now?",
                    "Can you hear the audio clearly now?",
                    "Is it better now?",
                ])

                return "audio_issue_retry", reply, history

            if is_positive:
                reset_audio_failure_count(history)
                history[:] = clear_pending_states(history)

                assistant_messages = [
                    item.get("content", "")
                    for item in history
                    if item.get("role") == "Assistant"
                ]

                previous_reply = ""
                if len(assistant_messages) >= 2:
                    previous_reply = assistant_messages[-2]
                elif len(assistant_messages) == 1:
                    previous_reply = assistant_messages[-1]

                if previous_reply:
                    return "audio_confirmed_repeat", f"Thanks. I’ll repeat that: {previous_reply}", history

                return "audio_confirmed_no_previous", "Thanks. How can I help you today?", history

    if language == "fa" and any(phrase in text for phrase in repeat_request_fa):
        previous_reply = last_assistant_message(history)

        if previous_reply:
            intro = pick([
                "حتماً، دوباره عرض می‌کنم:",
                "بله حتماً، تکرار می‌کنم:",
                "حتماً، یک بار دیگه عرض می‌کنم:",
                "چشم، دوباره می‌گم:",
            ])
            return "repeat_request", f"{intro} {previous_reply}", history

        return "repeat_request_no_previous", pick([
            "حتماً، لطفاً بفرمایید کدوم بخش رو دوباره توضیح بدم؟",
            "حتماً، کدوم قسمت رو دوست دارید دوباره تکرار کنم؟",
            "بله حتماً، لطفاً بفرمایید کدوم بخش رو دوباره بگم؟",
        ]), history

    if language == "en" and any(phrase in lower_text for phrase in repeat_request_en):
        previous_reply = last_assistant_message(history)

        if previous_reply:
            intro = pick([
                "Sure, I’ll repeat that:",
                "Of course, let me say that again:",
                "Sure, one more time:",
            ])
            return "repeat_request", f"{intro} {previous_reply}", history

        return "repeat_request_no_previous", pick([
            "Sure, which part would you like me to repeat?",
            "Of course. Which part should I say again?",
            "Sure, please tell me which part you’d like repeated.",
        ]), history

    if language == "fa" and any(phrase in text for phrase in audio_connection_issue_fa):
        set_pending_state(history, "waiting_for_audio_confirmation")
        set_audio_failure_count(history, 0)

        reply = pick([
            "الان صدای من رو دارید؟",
            "الان صدام واضح میاد؟",
            "الان صدا رو واضح دارید؟",
            "الان بهتر صدام رو می‌شنوید؟",
        ])

        return "audio_issue_check", reply, history

    if language == "en" and any(phrase in lower_text for phrase in audio_connection_issue_en):
        set_pending_state(history, "waiting_for_audio_confirmation")
        set_audio_failure_count(history, 0)

        reply = pick([
            "Can you hear me now?",
            "Is my voice clear now?",
            "Can you hear the audio clearly now?",
            "Is it better now?",
        ])

        return "audio_issue_check", reply, history

    return None, None, history


# -----------------------------
# Whisper STT
# -----------------------------

def transcribe_audio(audio_path):
    start_time = time.time()

    audio_path = preprocess_audio_for_stt(audio_path)

    def run_transcribe(forced_language=None):
        segments, info = stt_model.transcribe(
            audio_path,
            language=forced_language,
            task="transcribe",
            beam_size=2,
            best_of=2,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300
            ),
            initial_prompt=CALL_CENTER_STT_PROMPT,
            temperature=0.0
        )

        raw_parts = []

        for segment in segments:
            cleaned = segment.text.strip()
            if cleaned:
                raw_parts.append(cleaned)

        raw_text = " ".join(raw_parts).strip()
        whisper_language = getattr(info, "language", None)
        probability = round(getattr(info, "language_probability", 0), 2)

        return raw_text, whisper_language, probability

    raw_text, whisper_language, language_probability = run_transcribe(
        forced_language=None
    )

    latin_chars = sum(1 for ch in raw_text if "a" <= ch.lower() <= "z")
    persian_chars = sum(1 for ch in raw_text if "\u0600" <= ch <= "\u06FF")

    lower_raw_text = raw_text.lower()

    persian_hint_latin = any(
        hint in lower_raw_text
        for hint in [
            "salam",
            "selam",
            "janab",
            "canav",
            "khaste",
            "hasne",
            "hassne",
            "haste",
            "nabash",
            "nabish",
            "vaght",
            "bekheir",
        ]
    )

    should_retry_fa = (
        latin_chars > persian_chars
        and len(raw_text.strip()) <= 100
        and (
            language_probability < 0.72
            or persian_hint_latin
        )
    )

    if should_retry_fa:
        retry_raw_text, retry_language, retry_probability = run_transcribe(
            forced_language="fa"
        )

        if retry_raw_text:
            raw_text = retry_raw_text
            whisper_language = retry_language
            language_probability = retry_probability

    elapsed = round(time.time() - start_time, 2)

    normalized_text = normalize_text_light(raw_text)

    detected_language = detect_language_simple(
        normalized_text,
        whisper_language
    )

    normalized_text = clean_text_by_language(
        normalized_text,
        detected_language
    )

    return raw_text, normalized_text, elapsed, detected_language, language_probability

# -----------------------------
# Ollama reply
# -----------------------------

def generate_reply_with_ollama(user_text, language, history):
    start_time = time.time()

    history_text = build_history_text(history)

    if language == "fa":
        instruction = """
پاسخ را فارسی بده.
مثل اپراتور زنده و آموزش‌دیده جواب بده.
جواب باید محاوره‌ای، محترمانه، کوتاه و مناسب تماس تلفنی باشد.
اگر مناسب بود، جواب را با یک تأیید کوتاه شروع کن؛ مثل «بله، متوجه شدم»، «درسته»، «حتماً»، «بله، حتماً».
زیادی رسمی نباش.
زیادی خودمانی هم نباش.
از کلماتی مثل «داداش»، «عزیز»، «اوکی»، «شماره‌تو بده» استفاده نکن.
اگر کاربر فقط سلام کرده، حتماً جواب را با «سلام» شروع کن و درباره بلیت یا پرداخت حرف نزن.
اگر کاربر گفته «خسته نباشید»، آن را تکرار نکن؛ فقط تشکر کن.
اگر مشکل مبهم است، فقط یک سؤال روشن‌کننده بپرس.
اگر موضوع پرداخت، بلیت، پیامک، سفارش یا استرداد است، شماره موبایل یا کد رزرو را بخواه.
اگر کاربر خودش گفته پیامک، بلیت یا اطلاعات بلیت به دستش نرسیده، دوباره نپرس که آیا به دستش رسیده یا نه.
در این حالت فقط مشکل را تأیید کن و شماره موبایل یا کد رزرو را بخواه.
آخر پاسخ سؤال اضافه مثل «به دستتون رسیده؟»، «پیامکش رسیده؟» یا «نرسیده بود؟» نپرس.
اگر کاربر شماره یا کد داد، فقط دریافت اطلاعات را تأیید کن؛ ادعا نکن بررسی انجام شده.
اگر اطلاعات کافی نیست، حدس نزن.
فقط یک سؤال در هر پاسخ بپرس.
"""
    else:
        instruction = """
Reply in English.
Behave like a polite, trained, live call center agent.
Keep it short, warm, natural, and suitable for a phone call.
Start with a short acknowledgment when appropriate, such as "Sure", "I understand", or "Right, I see".
Do not sound robotic.
If the user only greets you, start with a greeting and do not mention tickets or payment.
If the issue is vague, ask one neutral clarifying question.
If the issue is about payment, ticket, SMS, order, or refund, ask for the phone number or reservation code.
If the user already said they did not receive the ticket, SMS, or ticket information, do not ask again whether it was received.
Only acknowledge the issue and ask for the phone number or reservation code.
Do not add extra confirmation questions like "did you receive it?" or "was it delivered?"
If the user provides a phone number or reservation code, acknowledge receiving it but do not claim that you checked anything.
Do not guess if information is missing.
Ask only one question in each reply.
"""

    prompt = f"""
Conversation history:
{history_text}

Current user message:
{user_text}

Instruction:
{instruction}

Assistant reply:
""".strip()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.15,
                "top_p": 0.8,
                "repeat_penalty": 1.15,
                "num_predict": 90
            }
        },
        timeout=120
    )

    response.raise_for_status()
    data = response.json()

    reply = data.get("response", "").strip()
    elapsed = round(time.time() - start_time, 2)

    reply = reply.split("\n")[0].strip()

    return reply, elapsed


# -----------------------------
# Guardrails
# -----------------------------

def apply_guardrails(reply, user_text, language, history):
    reply = reply.strip()

    fatigue_greeting_phrases = [
        "خسته نباشید",
        "خسته نباشی",
        "خسته نباشین",
    ]

    if language == "fa" and any(phrase in user_text for phrase in fatigue_greeting_phrases):
        issue_keywords = [
            "پرداخت",
            "بلیت",
            "بلیط",
            "پیامک",
            "استرداد",
            "کد رزرو",
            "سفارش",
            "پول",
            "تراکنش",
            "اپراتور",
            "مشکل",
            "نیومده",
            "نرسیده",
            "ارسال نشده",
        ]

        if not any(keyword in user_text for keyword in issue_keywords):
            return "سلام، ممنونم. بفرمایید چطور می‌تونم کمک‌تون کنم؟"

        reply = reply.replace("خسته نباشید", "").strip()
        reply = reply.replace("خسته نباشی", "").strip()
        reply = reply.replace("خسته نباشین", "").strip()

    unwanted_prefixes = [
        "Assistant:",
        "پاسخ:",
        "جواب:",
        "Call Center Reply:",
    ]

    for prefix in unwanted_prefixes:
        if reply.startswith(prefix):
            reply = reply[len(prefix):].strip()

    bad_patterns = [
        "A.",
        "B.",
        "C.",
        "D.",
        "Which of the following",
        "Answer:",
        "گزینه",
        "کدام یک",
    ]

    if any(pattern in reply for pattern in bad_patterns):
        if language == "en":
            return "I understand. Could you briefly tell me what the issue is about?"
        return "بله، متوجه شدم. لطفاً کوتاه بفرمایید مشکل درباره چه موضوعیه؟"

    forbidden_claims_fa = [
        "ارسال شد",
        "ارسال کردم",
        "استرداد انجام شد",
        "بررسی کردم",
        "پرداخت شما تأیید شد",
        "پرداخت شما تایید شد",
        "سفارش شما پیدا شد",
        "بلیت شما صادر شد",
        "بلیت شما ارسال شد",
    ]

    forbidden_claims_en = [
        "I sent",
        "has been sent",
        "refund is completed",
        "I checked",
        "payment is confirmed",
        "order was found",
        "ticket has been issued",
    ]

    if language == "fa" and any(claim in reply for claim in forbidden_claims_fa):
        return "بله، متوجه شدم. برای بررسی دقیق، لطفاً شماره موبایل یا کد رزرو رو بفرمایید."

    if language == "en" and any(claim.lower() in reply.lower() for claim in forbidden_claims_en):
        return "I understand. To check this properly, please provide your phone number or reservation code."

    if has_phone_or_tracking(user_text):
        if language == "en":
            return "Thanks, I received it. To continue, this needs to be checked in the order system."
        return "ممنون، اطلاعات رو دریافت کردم. برای ادامه باید این مورد در سیستم سفارش‌ها بررسی بشه."

    if language == "fa":
        tone_replacements = {
            "شماره موبایل شماره موبایلی": "شماره موبایلی",
            "شماره موبایل شماره موبایل": "شماره موبایل",
            "لطفاً شماره موبایل شماره موبایلی": "لطفاً شماره موبایلی",
            "خریدت رو بفرمایید": "شماره موبایلی که باهاش خرید انجام دادید یا کد رزرو رو بفرمایید",
            "شماره موبایلت رو": "شماره موبایلی که باهاش خرید انجام دادید",
            "بلیتت رو": "بلیت‌تون رو",
            "بلیطت رو": "بلیت‌تون رو",
            "کدت رو": "کدتون رو",
            "شماره‌تو": "شماره موبایل‌تون رو",
            "شمارتو": "شماره موبایل‌تون رو",
        }

        for wrong, correct in tone_replacements.items():
            reply = reply.replace(wrong, correct)

        reply = reply.replace(
            "شماره موبایلی که باهاش خرید انجام دادید و یا",
            "شماره موبایلی که باهاش خرید انجام دادید یا"
        )

    if language == "fa":
        already_not_received = any(phrase in user_text for phrase in [
            "پیامک نشده",
            "پیامک نشد",
            "پیامکش نیومده",
            "نرسیده",
            "نیومده",
            "ارسال نشده",
            "دریافت نکردم",
            "به دستم نرسیده",
        ])

        if already_not_received:
            redundant_questions = [
                "اطلاعات بلیت یا پیامکش به دستتون رسیده نبود؟",
                "اطلاعات بلیت یا پیامکش به دستتون رسیده؟",
                "بلیت یا پیامکش به دستتون رسیده نبود؟",
                "بلیت یا پیامکش به دستتون رسیده؟",
                "پیامکش به دستتون رسیده نبود؟",
                "پیامکش به دستتون رسیده؟",
                "نرسیده بود؟",
                "دریافت کردید؟",
            ]

            for question in redundant_questions:
                reply = reply.replace(question, "").strip()

            if reply.endswith("؟"):
                reply = reply[:-1].strip()

            reply = re.sub(r"\s+", " ", reply).strip()

    if language == "fa":
        greeting_words = ["سلام", "وقتتون بخیر", "عصرتون بخیر", "صبحتون بخیر", "شبتون بخیر"]
        user_greeted = any(word in user_text for word in greeting_words)

        if user_greeted and not reply.startswith("سلام"):
            non_issue_keywords = [
                "پرداخت",
                "بلیت",
                "پیامک",
                "پول",
                "تراکنش",
                "استرداد",
                "مشکل",
                "نیومده",
                "نرسیده",
            ]

            if not any(keyword in user_text for keyword in non_issue_keywords):
                reply = "سلام، " + reply

    if len(reply) > 260:
        if language == "en":
            reply = reply[:240].rsplit(".", 1)[0].strip() + "."
        else:
            reply = reply[:240].rsplit("،", 1)[0].strip() + "."

    if reply.count("?") + reply.count("؟") > 1:
        if language == "en":
            parts = re.split(r"(?<=[?])", reply)
        else:
            parts = re.split(r"(?<=[؟])", reply)

        reply = parts[0].strip()

    return reply


# -----------------------------
# Output helper
# -----------------------------

def build_output_text(
    raw_text,
    normalized_text,
    detected_language,
    language_probability,
    assistant_reply,
    source,
    stt_time,
    llm_time,
    total_time,
    history,
    detected_number=""
):
    number_section = ""

    if detected_number:
        number_section = (
            f"Detected Number:\n"
            f"{detected_number}\n\n"
        )

    return (
        f"Raw STT Text:\n"
        f"{raw_text}\n\n"
        f"Normalized Text:\n"
        f"{normalized_text}\n\n"
        f"{number_section}"
        f"Detected Language:\n"
        f"{detected_language} / probability: {language_probability}\n\n"
        f"Assistant Reply:\n"
        f"{assistant_reply}\n\n"
        f"Source:\n"
        f"{source}\n\n"
        f"STT Time: {stt_time} seconds\n"
        f"Ollama Time: {llm_time} seconds\n"
        f"Total Time: {total_time} seconds\n"
        f"STT Device: {DEVICE} / {COMPUTE_TYPE}\n"
        f"STT Model: {WHISPER_MODEL_NAME}\n"
        f"Ollama Model: {OLLAMA_MODEL}\n\n"
        f"Conversation Memory:\n"
        f"{build_history_text(history)}"
    )


# -----------------------------
# Gradio process
# -----------------------------

def fast_playbook_slot_flow(user_text, language, history, detected_number):
    """
    Generic slot-based flow from playbook.yaml.
    Example: cancellation request -> ask reservation code -> confirm -> complete.
    """

    playbook = load_playbook()
    slot_flows = playbook.get("slot_flows", {})

    if not slot_flows:
        return None, None, history

    text = normalize_text_light(user_text)
    lower_text = text.lower()
    pending_state = get_pending_state(history)

    positive_fa = [
        "بله", "آره", "اره", "درسته", "درست",
        "بله درسته", "تایید", "تأیید", "همینه", "صحیحه"
    ]

    negative_fa = [
        "نه", "خیر", "اشتباه", "اشتباهه", "درست نیست"
    ]

    positive_en = [
        "yes", "yeah", "correct", "that's right", "that is right",
        "confirmed", "right", "true", "evet", "doğru", "dogru",
        "kesinlikle doğru", "kesinlikle dogru"
    ]

    negative_en = [
        "no", "wrong", "incorrect", "not correct",
        "hayır", "hayir", "yanlış", "yanlis"
    ]

    def get_reply(flow_data, key, value=""):
        responses = flow_data.get(key, {})
        response_list = responses.get(language) or responses.get("fa") or []

        if isinstance(response_list, str):
            response_list = [response_list]

        if not response_list:
            return ""

        reply = pick(response_list)

        if value:
            reply = reply.replace("{value}", value)

        return reply

    if pending_state.startswith("slot_flow:"):
        parts = pending_state.split(":")

        if len(parts) >= 3:
            flow_name = parts[1]
            slot_name = parts[2]
            flow_data = slot_flows.get(flow_name, {})

            pending_value_key = f"pending_{flow_name}_{slot_name}"
            pending_value = get_state_value(history, pending_value_key, "")

            if detected_number:
                slot_data = flow_data.get("slot", {})
                expected_length = int(slot_data.get("length", 0))
                clean_value = re.sub(r"\D", "", detected_number)

                if expected_length and len(clean_value) != expected_length:
                    reply = get_reply(flow_data, "invalid")
                    return f"{flow_name}_invalid_slot", reply, history

                set_state_value(history, pending_value_key, clean_value)

                reply = get_reply(flow_data, "confirm", clean_value)
                return f"{flow_name}_confirm_slot", reply, history

            if pending_value:
                is_positive = (
                    any(x in text for x in positive_fa)
                    or any(x in lower_text for x in positive_en)
                )

                is_negative = (
                    any(x in text for x in negative_fa)
                    or any(x in lower_text for x in negative_en)
                )

                if is_negative:
                    clear_state_value(history, pending_value_key)
                    reply = get_reply(flow_data, "rejected")
                    return f"{flow_name}_rejected", reply, history

                if is_positive:
                    history[:] = clear_pending_states(history)
                    clear_state_value(history, pending_value_key)
                    reply = get_reply(flow_data, "completed", pending_value)
                    return f"{flow_name}_completed", reply, history

                reply = get_reply(flow_data, "unclear_confirmation", pending_value)
                return f"{flow_name}_unclear_confirmation", reply, history

            reply = get_reply(flow_data, "invalid")
            return f"{flow_name}_slot_not_clear", reply, history

    matches = []

    for flow_name, flow_data in slot_flows.items():
        triggers = flow_data.get("triggers", [])
        priority = int(flow_data.get("priority", 0))

        for trigger in triggers:
            trigger_text = normalize_text_light(str(trigger))
            trigger_lower = trigger_text.lower()

            if trigger_text and (trigger_text in text or trigger_lower in lower_text):
                matches.append((priority, flow_name, flow_data))
                break

    if not matches:
        return None, None, history

    matches.sort(key=lambda item: item[0], reverse=True)
    _, flow_name, flow_data = matches[0]

    slot_data = flow_data.get("slot", {})
    slot_name = slot_data.get("name", "slot")

    set_pending_state(history, f"slot_flow:{flow_name}:{slot_name}")

    reply = get_reply(flow_data, "ask")
    return flow_name, reply, history
def process_text(user_text, history=None):
    if history is None:
        history = []

    normalized_text = normalize_text_light(user_text)
    normalized_text, detected_number = enrich_text_with_detected_number(normalized_text)

    detected_language = detect_language_simple(normalized_text, None)

    number_intent, number_reply, history = fast_number_confirmation_flow(
        user_text=normalized_text,
        language=detected_language,
        history=history,
        detected_number=detected_number
    )

    if number_reply:
        return number_reply, f"Fast Number Flow / {number_intent}"

    courtesy_intent, courtesy_reply = fast_courtesy_reply(
        user_text=normalized_text,
        language=detected_language,
        history=history
    )

    if courtesy_reply:
        return courtesy_reply, f"Fast Courtesy Handler / {courtesy_intent}"

    cancel_intent, cancel_reply = fast_refund_cancellation_reply(
        user_text=normalized_text,
        language=detected_language,
        history=history
    )

    if cancel_reply:
        return cancel_reply, f"Fast Refund/Cancellation Handler / {cancel_intent}"

    domain_reply = get_playbook_domain_reply(
        user_text=normalized_text,
        language=detected_language
    )

    if domain_reply:
        return domain_reply, "Playbook Domain Gate / out_of_domain"

    assistant_reply, llm_time = generate_reply_with_ollama(
        user_text=normalized_text,
        language=detected_language,
        history=history
    )

    assistant_reply = clean_persian_tone(assistant_reply, detected_language)

    return assistant_reply, "Ollama-first + Guardrails + Tone Cleaner"
def process_voice(audio_path, history, stt_test_mode=False, presentation_mode=True):
    if history is None:
        history = []

    if audio_path is None:
        return "لطفاً یک صدا ضبط کنید یا فایل صوتی آپلود کنید.", history

    total_start = time.time()
    status_text = "Transcribing voice input..."
    raw_text, normalized_text, stt_time, detected_language, language_probability = transcribe_audio(audio_path)
    normalized_text, detected_number = enrich_text_with_detected_number(normalized_text)
    if stt_test_mode:
        


        return output_text, history

    def finish_reply(assistant_reply, source, llm_time=0):
        total_time = round(time.time() - total_start, 2)

        history.append({"role": "UserLanguage", "content": detected_language})
        history.append({"role": "User", "content": normalized_text})
        history.append({"role": "Assistant", "content": assistant_reply})

        if presentation_mode:
            pretty_source = source.replace("_", " ").replace("/", " / ")
            
            output_text = (
                f"CALL SUMMARY\n\n"
                f"Customer\n"
                f"{normalized_text}\n\n"
                f"Sira\n"
                f"{assistant_reply}\n\n"
                f"STATE\n"
                f"{pretty_source}\n\n"
                f"TIME\n"
                f"{total_time}s"
            )

            source_labels = {
                "Fast Courtesy Handler / greeting": "Greeting detected",
                "Fast Courtesy Handler / greeting_with_courtesy": "Greeting detected",
                "Fast Number Flow / reservation_code_detected_ask_confirmation": "Reservation code detected",
                "Fast Number Flow / identifier_confirmed": "Customer verification completed",
                "Fast Number Flow / invalid_identifier": "Invalid identifier",
                "Fast Number Flow / invalid_reservation_code": "Invalid reservation code",
                "Fast Number Flow / identifier_rejected": "Customer rejected detected number",
                "Fast Refund/Cancellation Handler / general_cancellation_request": "Cancellation request detected",
                "Playbook Slot Flow / cancellation_request": "Cancellation request detected",
                "Ollama-first + Guardrails + Tone Cleaner": "AI conversation response",
            }

            pretty_source = source_labels.get(
                source,
                source.replace("_", " ").replace("/", " / ")
            )

            output_text = (
                f"Customer said:\n"
                f"{normalized_text}\n\n"
                f"━━━━━━━━━━━━━━\n\n"
                f"Sirareplied:\n"
                f"{assistant_reply}\n\n"
                f"━━━━━━━━━━━━━━\n\n"
                f"Conversation state:\n"
                f"{pretty_source}\n\n"
                f"Latency:\n"
                f"{total_time} seconds"
            )

        else:
            output_text = build_output_text(
                raw_text=raw_text,
                normalized_text=normalized_text,
                detected_language=detected_language,
                language_probability=language_probability,
                assistant_reply=assistant_reply,
                source=source,
                stt_time=stt_time,
                llm_time=llm_time,
                total_time=total_time,
                history=history,
                detected_number=detected_number
            )

        return output_text, history

    recent_assistant_text = " ".join([
        item.get("content", "")
        for item in history[-6:]
        if item.get("role") == "Assistant"
    ])

    asked_for_reservation_code = any(phrase in recent_assistant_text for phrase in [
        "کد رزرو",
        "reservation code",
    ])

    asked_for_phone_number = any(phrase in recent_assistant_text for phrase in [
        "شماره موبایل",
        "شماره موبایلی",
        "phone number",
    ])

    asked_only_for_reservation_code = asked_for_reservation_code and not asked_for_phone_number

    pending_state = get_pending_state(history)

    # # -----------------------------
    # # Cancellation reservation code confirmation
    # # -----------------------------

    # if pending_state == "waiting_for_reservation_code_for_cancellation":
    #     pending_reservation_code = get_state_value(history, "pending_reservation_code", "")

    #     positive_fa = [
    #         "بله",
    #         "آره",
    #         "اره",
    #         "درسته",
    #         "درست",
    #         "بله درست",
    #         "بله درسته",
    #         "تایید",
    #         "تأیید",
    #         "همینه",
    #         "صحیحه",
    #     ]

    #     negative_fa = [
    #         "نه",
    #         "خیر",
    #         "اشتباه",
    #         "اشتباهه",
    #         "درست نیست",
    #         "نه درست نیست",
    #     ]

    #     positive_en = [
    #         "yes",
    #         "yeah",
    #         "correct",
    #         "that's right",
    #         "that is right",
    #         "right",
    #         "confirmed",
    #     ]

    #     negative_en = [
    #         "no",
    #         "wrong",
    #         "incorrect",
    #         "that's wrong",
    #         "that is wrong",
    #         "not correct",
    #     ]

    #     lower_normalized_text = normalized_text.lower()

    #     if pending_reservation_code and not detected_number:
    #         if detected_language == "fa":
    #             is_negative = any(phrase in normalized_text for phrase in negative_fa)
    #             is_positive = any(phrase in normalized_text for phrase in positive_fa)

    #             if is_negative:
    #                 clear_state_value(history, "pending_reservation_code")
    #                 assistant_reply = "ببخشید، پس لطفاً کد رزرو ۸ رقمی رو یک بار دیگه بفرمایید."
    #                 source = "Fast Cancellation Reservation Flow / reservation_code_rejected"
    #                 return finish_reply(assistant_reply, source)

    #             if is_positive:
    #                 history[:] = clear_pending_states(history)
    #                 clear_state_value(history, "pending_reservation_code")
    #                 assistant_reply = "ممنونم، کد رزرو ثبت شد. بعد از بررسی سفارش، وضعیت امکان کنسلی مشخص می‌شه."
    #                 source = "Fast Cancellation Reservation Flow / reservation_code_confirmed"
    #                 return finish_reply(assistant_reply, source)

    #         else:
    #             is_negative = any(phrase in lower_normalized_text for phrase in negative_en)
    #             is_positive = any(phrase in lower_normalized_text for phrase in positive_en)

    #             if is_negative:
    #                 clear_state_value(history, "pending_reservation_code")
    #                 assistant_reply = "Sorry about that. Please say the 8-digit reservation code one more time."
    #                 source = "Fast Cancellation Reservation Flow / reservation_code_rejected"
    #                 return finish_reply(assistant_reply, source)

    #             if is_positive:
    #                 history[:] = clear_pending_states(history)
    #                 clear_state_value(history, "pending_reservation_code")
    #                 assistant_reply = "Thanks, the reservation code is saved. After checking the order, the cancellation status can be confirmed."
    #                 source = "Fast Cancellation Reservation Flow / reservation_code_confirmed"
    #                 return finish_reply(assistant_reply, source)

    # # -----------------------------
    # # Direct reservation code handling
    # # -----------------------------

    # if detected_number and (
    #     pending_state == "waiting_for_reservation_code_for_cancellation"
    #     or asked_only_for_reservation_code
    # ):
    #     reservation_code = re.sub(r"\D", "", detected_number)

    #     if len(reservation_code) == 8:
    #         set_pending_state(history, "waiting_for_reservation_code_for_cancellation")
    #         set_state_value(history, "pending_reservation_code", reservation_code)

    #         if detected_language == "en":
    #             assistant_reply = f"Okay, so the reservation code is {reservation_code}. Is that right?"
    #         else:
    #             assistant_reply = f"بله، پس کد رزرو شد {reservation_code}. درسته؟"

    #         source = "Fast Cancellation Reservation Flow / reservation_code_detected_ask_confirmation"
    #         return finish_reply(assistant_reply, source)

    #     if detected_language == "en":
    #         assistant_reply = "The reservation code should be 8 digits. Please say the 8-digit reservation code one more time."
    #     else:
    #         assistant_reply = "کد رزرو باید ۸ رقمی باشه. لطفاً کد رزرو ۸ رقمی رو یک بار دیگه بفرمایید."

    #     source = "Fast Cancellation Reservation Flow / reservation_code_invalid_length"
    #     return finish_reply(assistant_reply, source)

    # if not detected_number and (
    #     pending_state == "waiting_for_reservation_code_for_cancellation"
    #     or asked_only_for_reservation_code
    # ):
    #     pending_reservation_code = get_state_value(history, "pending_reservation_code", "")

    #     if pending_reservation_code:
    #         if detected_language == "en":
    #             assistant_reply = f"Sorry, I didn’t catch that clearly. Is the reservation code {pending_reservation_code} correct?"
    #         else:
    #             assistant_reply = f"ببخشید، تأییدتون رو واضح متوجه نشدم. کد رزرو {pending_reservation_code} درسته؟"

    #         source = "Fast Cancellation Reservation Flow / confirmation_not_clear"
    #         return finish_reply(assistant_reply, source)

    #     if detected_language == "en":
    #         assistant_reply = "Sorry, I couldn’t get the reservation code clearly. Please say the 8-digit reservation code one more time."
    #     else:
    #         assistant_reply = "ببخشید، کد رزرو رو واضح نگرفتم. لطفاً کد رزرو ۸ رقمی رو یک بار دیگه بفرمایید."

    #     source = "Fast Cancellation Reservation Flow / reservation_code_not_clear"
    #     return finish_reply(assistant_reply, source)

    # # -----------------------------
    # # Number confirmation flow
    # # -----------------------------

    number_intent, number_reply, history = fast_number_confirmation_flow(
        user_text=normalized_text,
        language=detected_language,
        history=history,
        detected_number=detected_number
    )

    if number_reply:
        assistant_reply = clean_persian_tone(number_reply, detected_language)
        source = f"Fast Number Flow / {number_intent}"
        return finish_reply(assistant_reply, source)

    # -----------------------------
    # Audio / repeat flow
    # -----------------------------

    audio_intent, audio_reply, history = fast_audio_issue_flow(
        user_text=normalized_text,
        language=detected_language,
        history=history
    )

    if audio_reply:
        assistant_reply = clean_persian_tone(audio_reply, detected_language)
        source = f"Fast Audio Flow / {audio_intent}"
        return finish_reply(assistant_reply, source)
    
    # -----------------------------
    # Generic playbook slot flow
    # -----------------------------

    slot_intent, slot_reply, history = fast_playbook_slot_flow(
        user_text=normalized_text,
        language=detected_language,
        history=history,
        detected_number=detected_number
    )

    if slot_reply:
        assistant_reply = clean_persian_tone(slot_reply, detected_language)
        source = f"Playbook Slot Flow / {slot_intent}"
        return finish_reply(assistant_reply, source)
    
    # -----------------------------
    # Bad STT
    # -----------------------------

    if not normalized_text:
        reply_language = get_last_user_language(history)
        assistant_reply = unclear_stt_reply(reply_language)

        output_text = build_output_text(
            raw_text=raw_text,
            normalized_text=normalized_text,
            detected_language=detected_language,
            language_probability=language_probability,
            assistant_reply=assistant_reply,
            source="Bad STT / Empty Text",
            stt_time=stt_time,
            llm_time=0,
            total_time=round(time.time() - total_start, 2),
            history=history,
            detected_number=detected_number
        )

        return output_text, history

    if looks_like_bad_stt(normalized_text):
        reply_language = get_last_user_language(history)
        assistant_reply = unclear_stt_reply(reply_language)

        output_text = build_output_text(
            raw_text=raw_text,
            normalized_text=normalized_text,
            detected_language=detected_language,
            language_probability=language_probability,
            assistant_reply=assistant_reply,
            source="Bad STT Detector / Memory Not Updated",
            stt_time=stt_time,
            llm_time=0,
            total_time=round(time.time() - total_start, 2),
            history=history,
            detected_number=detected_number
        )

        return output_text, history

    # -----------------------------
    # Fast handlers
    # -----------------------------

    fast_intent, fast_reply = fast_courtesy_reply(
        user_text=normalized_text,
        language=detected_language,
        history=history
    )

    if fast_reply:
        assistant_reply = clean_persian_tone(fast_reply, detected_language)
        source = f"Fast Courtesy Handler / {fast_intent}"
        return finish_reply(assistant_reply, source)

    cancel_intent, cancel_reply = fast_refund_cancellation_reply(
        user_text=normalized_text,
        language=detected_language,
        history=history
    )

    if cancel_reply:
        assistant_reply = clean_persian_tone(cancel_reply, detected_language)
        source = f"Fast Refund/Cancellation Handler / {cancel_intent}"
        return finish_reply(assistant_reply, source)

    ticket_intent, ticket_reply = fast_ticket_sms_issue_reply(
        user_text=normalized_text,
        language=detected_language,
        history=history
    )

    if ticket_reply:
        assistant_reply = clean_persian_tone(ticket_reply, detected_language)
        source = f"Fast Ticket/SMS Handler / {ticket_intent}"
        return finish_reply(assistant_reply, source)

    # -----------------------------
    # Fast seat reserved handler
    # -----------------------------

    seat_intent, seat_reply = fast_seat_reserved_issue_reply(
        user_text=normalized_text,
        language=detected_language,
        history=history
    )

    if seat_reply:
        assistant_reply = clean_persian_tone(seat_reply, detected_language)
        source = f"Fast Seat Reserved Flow / {seat_intent}"
        return finish_reply(assistant_reply, source)

    # -----------------------------
    # Playbook generic flow
    # -----------------------------

    playbook_intent, playbook_reply = fast_playbook_reply(
        user_text=normalized_text,
        language=detected_language,
        history=history
    )

    if playbook_reply:
        assistant_reply = clean_persian_tone(playbook_reply, detected_language)
        source = f"Fast Playbook Flow / {playbook_intent}"
        return finish_reply(assistant_reply, source)
    
    # -----------------------------
    # Generic playbook domain gate
    # -----------------------------

    domain_reply = get_playbook_domain_reply(
        user_text=normalized_text,
        language=detected_language
    )

    if domain_reply:
        assistant_reply = clean_persian_tone(domain_reply, detected_language)
        source = "Playbook Domain Gate / out_of_domain"
        return finish_reply(assistant_reply, source)
    
    # -----------------------------
    # Ollama fallback
    # -----------------------------

    try:
        assistant_reply, llm_time = generate_reply_with_ollama(
            user_text=normalized_text,
            language=detected_language,
            history=history
        )

        assistant_reply = apply_guardrails(
            reply=assistant_reply,
            user_text=normalized_text,
            language=detected_language,
            history=history
        )

        assistant_reply = clean_persian_tone(assistant_reply, detected_language)

        source = "Ollama-first + Guardrails + Tone Cleaner"

    except Exception as e:
        llm_time = 0
        source = "Error"
        print("Ollama error:", e)

        if detected_language == "en":
            assistant_reply = "I understand. Could you briefly tell me what the issue is about?"
        else:
            assistant_reply = "ببخشید، کامل متوجه نشدم. جسارتاً یک بار دیگه تکرار می‌کنید؟"

    return finish_reply(assistant_reply, source, llm_time)
def reset_conversation():
    return [], "Conversation reset."


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800;900&display=swap');

* {
    font-family: 'Vazirmatn', Arial, sans-serif !important;
    box-sizing: border-box;
}

body {
    background:
        radial-gradient(circle at top right, rgba(239, 35, 60, 0.20), transparent 34%),
        radial-gradient(circle at bottom left, rgba(59, 130, 246, 0.16), transparent 32%),
        linear-gradient(135deg, #0b1020 0%, #111827 44%, #1f2937 100%) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    direction: rtl !important;
    text-align: right !important;
    padding-top: 28px !important;
    color: #f8fafc !important;
}

.gradio-container * {
    direction: rtl !important;
}

button {
    text-align: center !important;
    border-radius: 18px !important;
    font-weight: 800 !important;
    min-height: 46px !important;
    transition: all 0.22s ease !important;
}

button:hover {
    transform: translateY(-1px);
    filter: brightness(1.04);
}

textarea,
input {
    border-radius: 18px !important;
}

.cinema-shell {
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.14), rgba(255,255,255,0.06));
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 34px;
    padding: 34px;
    box-shadow: 0 30px 90px rgba(0,0,0,0.34);
    backdrop-filter: blur(18px);
    margin-bottom: 22px;
}

.cinema-shell::before {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    top: -90px;
    left: -80px;
    background: radial-gradient(circle, rgba(239,35,60,0.40), transparent 65%);
    filter: blur(4px);
    pointer-events: none;
}

.cinema-shell::after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    bottom: -90px;
    right: -70px;
    background: radial-gradient(circle, rgba(59,130,246,0.28), transparent 65%);
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-top {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: flex-start;
    margin-bottom: 24px;
}

.kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #fecdd3;
    background: rgba(239, 35, 60, 0.16);
    border: 1px solid rgba(248, 113, 113, 0.30);
    padding: 8px 13px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 14px;
}

.kicker-dot {
    width: 8px;
    height: 8px;
    border-radius: 99px;
    background: #fb7185;
    box-shadow: 0 0 20px rgba(251, 113, 133, 0.8);
}

.hero-title {
    color: #ffffff;
    font-size: 38px;
    line-height: 1.35;
    font-weight: 900;
    margin: 0 0 12px 0;
    letter-spacing: -0.7px;
}

.hero-desc {
    color: #dbeafe;
    font-size: 15px;
    line-height: 2.05;
    max-width: 780px;
    margin: 0;
}

.status-card {
    min-width: 190px;
    background: rgba(15, 23, 42, 0.54);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 24px;
    padding: 16px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
}

.status-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 8px;
}

.status-value {
    color: #ffffff;
    font-size: 18px;
    font-weight: 900;
}

.status-sub {
    color: #cbd5e1;
    font-size: 11px;
    margin-top: 6px;
}

.flow-row {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
    margin-top: 28px;
}

.flow-item {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 18px;
    padding: 13px 12px;
    color: #f8fafc;
}

.flow-number {
    width: 26px;
    height: 26px;
    border-radius: 9px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(239,35,60,0.92);
    color: #fff;
    font-size: 12px;
    font-weight: 900;
    margin-bottom: 8px;
}

.flow-text {
    font-size: 12px;
    font-weight: 700;
    color: #e5e7eb;
}

.glass-panel {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(255,255,255,0.64);
    border-radius: 28px;
    padding: 22px;
    box-shadow: 0 22px 70px rgba(0,0,0,0.22);
    color: #0f172a;
    height: 100%;
}

.panel-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 14px;
}

.panel-title {
    color: #0f172a;
    font-size: 18px;
    font-weight: 900;
    margin: 0;
}

.panel-badge {
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
}

.panel-desc {
    color: #64748b;
    font-size: 13px;
    line-height: 1.9;
    margin-bottom: 16px;
}

.primary-btn {
    background: linear-gradient(135deg, #ef233c 0%, #fb7185 100%) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 12px 28px rgba(239, 35, 60, 0.30) !important;
}

.secondary-btn {
    background: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid #1e293b !important;
}

.output-box textarea {
    direction: rtl !important;
    text-align: right !important;
    line-height: 1.95 !important;
    font-size: 14px !important;
    color: #0f172a !important;
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
    border: 1px solid #e2e8f0 !important;
}

label {
    color: #1e293b !important;
    font-weight: 800 !important;
}

.footer-dynamic {
    color: #cbd5e1;
    text-align: center;
    font-size: 12px;
    padding: 16px 0 4px;
}

.footer-dynamic span {
    color: #fb7185;
    font-weight: 800;
}

@media (max-width: 900px) {
    .hero-top {
        flex-direction: column;
    }

    .hero-title {
        font-size: 28px;
    }

    .flow-row {
        grid-template-columns: 1fr 1fr;
    }

    .cinema-shell {
        padding: 24px;
        border-radius: 26px;
    }
}
"""

with gr.Blocks(
    title=f"{PLATFORM_NAME} Voice AI",
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="red",
        neutral_hue="slate",
        font=["Vazirmatn", "Arial", "sans-serif"],
    ),
) as demo:

    gr.HTML("""
    <div class="cinema-shell">
        <div class="hero-content">
            <div class="hero-top">
                <div>
                    <div class="kicker">
                        <span class="kicker-dot"></span>
                        Sira· VOICE AI PROTOTYPE
                    </div>
                    <h1 class="hero-title">Sira؛ اپراتور صوتی هوشمند برای مکالمه‌های واقعی</h1>
                    <p class="hero-desc">
                       یک پروتوتایپ واقعی از AI Voice Agent؛ از شنیدن صدای مشتری تا تشخیص نیت，
اجرای قوانین هر کسب‌وکار و تولید پاسخ کوتاه، طبیعی و قابل اعتماد.
                    </p>
                </div>

                <div class="status-card">
                    <div class="status-label">Demo Status</div>
                    <div class="status-value">Live Prototype</div>
                    <div class="status-sub">STT · Guardrails · Playbook Engine</div>
                </div>
            </div>

            <div class="flow-row">
                <div class="flow-item">
                    <div class="flow-number">۱</div>
                    <div class="flow-text">دریافت صدای مشتری</div>
                </div>
                <div class="flow-item">
                    <div class="flow-number">۲</div>
                    <div class="flow-text">تبدیل گفتار به متن</div>
                </div>
                <div class="flow-item">
                    <div class="flow-number">۳</div>
                    <div class="flow-text">تشخیص نیت تماس</div>
                </div>
                <div class="flow-item">
                    <div class="flow-number">۴</div>
                    <div class="flow-text">اعمال قوانین پشتیبانی</div>
                </div>
                <div class="flow-item">
                    <div class="flow-number">۵</div>
                    <div class="flow-text">پاسخ اپراتوری کوتاه</div>
                </div>
            </div>
        </div>
    </div>
    """)

    state = gr.State([])

    with gr.Row(equal_height=True):

        with gr.Column(scale=4):

            with gr.Accordion("Audio Input", open=True):
                gr.Markdown("""
صدای مشتری را ضبط کنید یا فایل تماس را آپلود کنید.

سیستم بعد از تحلیل، مسیر تصمیم‌گیری و پاسخ مناسب را نمایش می‌دهد.
""")

            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label=""
            )

            with gr.Row():

                stt_test_mode = gr.Checkbox(
                    label="STT Test Mode",
                    value=False
                )

                presentation_mode = gr.Checkbox(
                    label="Presentation Mode",
                    value=True
                )

            with gr.Row():
                submit_btn = gr.Button(
                    "تحلیل تماس",
                    variant="primary",
                    elem_classes=["primary-btn"]
                )
                reset_btn = gr.Button(
                    "شروع مکالمه جدید",
                    elem_classes=["secondary-btn"]
                )

        with gr.Column(scale=6):

            with gr.Accordion("AI Result", open=True):
                gr.Markdown("""
متن خام، متن نرمال‌شده، شماره تشخیص‌داده‌شده، پاسخ دستیار، منبع تصمیم‌گیری و حافظه مکالمه اینجا نمایش داده می‌شود.
""")

            gr.Markdown("<div style='height: 54px'></div>")

            gr.HTML("""
            <div class="processing-status">
                <span class="status-dot"></span>
                <span>Ready to analyze voice input</span>
            </div>
            """)

            output_box = gr.Textbox(
                label="",
                lines=18,
                elem_id="conversation_output",
                container=False
            )

    gr.HTML("""
    <div class="footer-dynamic">
        Built as a <span>cinematic voice AI demo</span> for smarter customer support workflows
    </div>
    """)

    submit_btn.click(
        fn=process_voice,
        inputs=[audio_input, state, stt_test_mode, presentation_mode],
        outputs=[output_box, state]
    )

    reset_btn.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[state, output_box]
    )

demo.css = """
#conversation_output textarea {
    margin-top: 18px !important;

    background: linear-gradient(
        180deg,
        #111827 0%,
        #0F172A 100%
    ) !important;

    color: #F8FAFC !important;

    border: 1px solid rgba(255,255,255,0.08) !important;

    border-radius: 24px !important;

    padding: 28px !important;

    font-size: 15px !important;

    line-height: 2 !important;

    box-shadow:
        0 10px 40px rgba(0,0,0,0.35);

    font-family: Inter, sans-serif !important;
}

#conversation_output label {
    color: #CBD5E1 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}
button[aria-expanded] {

    background: linear-gradient(
        180deg,
        rgba(15,23,42,0.92),
        rgba(17,24,39,0.98)
    ) !important;

    border: 1px solid rgba(255,255,255,0.08) !important;

    border-radius: 18px !important;

    padding: 16px 20px !important;

    color: #F8FAFC !important;

    font-weight: 700 !important;

    transition: all 0.25s ease !important;

    box-shadow:
        0 8px 24px rgba(0,0,0,0.22) !important;
}

button[aria-expanded]:hover {

    border-color: rgba(255,255,255,0.18) !important;

    transform: translateY(-1px);

    background: linear-gradient(
        180deg,
        rgba(20,30,50,0.96),
        rgba(17,24,39,1)
    ) !important;
}
.gradio-audio {
    background: linear-gradient(
        180deg,
        rgba(15,23,42,0.92),
        rgba(17,24,39,0.96)
    ) !important;

    border: 1px solid rgba(255,255,255,0.08) !important;

    border-radius: 24px !important;

    padding: 18px !important;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.25) !important;
}

.gradio-audio:hover {
    border-color: rgba(255,255,255,0.16) !important;
}
.processing-status {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    padding: 10px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    color: #CBD5E1;
    font-size: 13px;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: #22C55E;
    box-shadow: 0 0 12px rgba(34,197,94,0.65);
    animation: pulseDot 1.4s ease-in-out infinite;
}

@keyframes pulseDot {
    0%, 100% {
        opacity: 0.45;
        transform: scale(0.9);
    }
    50% {
        opacity: 1;
        transform: scale(1.15);
    }
}

svg.lucide-chevron-down {
    display: none !important;
}
.block {
    gap: 22px !important;
}
"""
if __name__ == "__main__":
    demo.launch()