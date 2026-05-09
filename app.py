import os
import time
import re
import random

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

DEVICE = "cuda"
COMPUTE_TYPE = "float16"
WHISPER_MODEL_NAME = "large-v3-turbo"

OLLAMA_MODEL = "qwen-callcenter"
OLLAMA_URL = "http://localhost:11434/api/generate"

MAX_AUDIO_CONFIRMATION_FAILURES = 3

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
    "This is a bilingual Persian and English call center conversation. "
    "The topic may include payment, ticket, SMS, concert, cinema, theater, tracking code, order number, refund, support. "
    "این یک مکالمه فارسی و انگلیسی مرکز تماس است. "
    "کلمات پرتکرار: سلام، جناب، وقتتون بخیر، خسته نباشید، عصرتون بخیر باشه، صبحتون بخیر باشه، "
    "پرداخت، بلیت، بلیتم، پیامک، کد پیگیری، شماره موبایل، استرداد، پشتیبانی، اپراتور، "
    "شماره موبایل ممکن است اینطور گفته شود: صفر نهصد و دو دویست و یک چهل و دو هشتاد، "
    "یا نهصد و دوازده دویست و یک چهل و دو هشتاد. "
    "صداتون قطع شد، صداتون نیومد، دوباره می‌گید، تکرار می‌کنید."
)

# -----------------------------
# General helpers
# -----------------------------

def pick(options):
    return random.choice(options)


def detect_language_simple(text, whisper_language=None):
    if whisper_language in ["en", "fa"]:
        return whisper_language

    english_chars = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    persian_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")

    if english_chars > persian_chars:
        return "en"

    return "fa"


def normalize_text_light(text):
    """
    Light normalization only.
    This improves common STT mistakes without changing meaning too aggressively.
    """

    text = text.strip()

    replacements = {
        # Full greeting phrases first - order matters
        "عصر تون بخیر بشه": "عصرتون بخیر باشه",
        "عصرتون بخیر بشه": "عصرتون بخیر باشه",
        "عصر تون بخیر باشه": "عصرتون بخیر باشه",

        "صبح تون بخیر بشه": "صبحتون بخیر باشه",
        "صبحتون بخیر بشه": "صبحتون بخیر باشه",
        "صبح تون بخیر باشه": "صبحتون بخیر باشه",

        "شب تون بخیر بشه": "شبتون بخیر باشه",
        "شبتون بخیر بشه": "شبتون بخیر باشه",
        "شب تون بخیر باشه": "شبتون بخیر باشه",

        "وقت تون بخیر بشه": "وقتتون بخیر باشه",
        "وقتتون بخیر بشه": "وقتتون بخیر باشه",
        "وقت تون بخیر باشه": "وقتتون بخیر باشه",

        # Shorter greeting corrections
        "عصر تون": "عصرتون",
        "صبح تون": "صبحتون",
        "شب تون": "شبتون",
        "وقت تون": "وقتتون",

        # Greeting STT fixes
        "سلام جانب": "سلام جناب",
        "سلام جانبی": "سلام جناب",
        "سلام جنابی": "سلام جناب",
        "جنابی": "جناب",

        "بخیر بشه": "بخیر باشه",
        "به خیر بشه": "بخیر باشه",
        "به خیر": "بخیر",
        "بحیر": "بخیر",

        # Common courtesy corrections
        "خسته توشید": "خسته نباشید",
        "خسته نوشید": "خسته نباشید",
        "خسته باشید": "خسته نباشید",
        "خسته نبشید": "خسته نباشید",
        "خسته نباشید.": "خسته نباشید",

        # Common Persian STT corrections
        "وفتتون": "وقتتون",
        "وقتون": "وقتتون",
        "وقتون بخیر": "وقتتون بخیر",
        "اندوز": "هنوز",

        # Audio / repeat issue corrections
        "صدرزون": "صداتون",
        "صدر تون": "صداتون",
        "صدا تون": "صداتون",
        "صدارتون": "صداتون",
        "صداتون قطع وقت شد": "صداتون قطع شد",
        "صدا قطع وقت شد": "صدا قطع شد",
        "صدا تون قطع شد": "صداتون قطع شد",
        "صداتون نیومده": "صداتون نیومد",
        "صدا نیومده": "صدا نیومد",
        "دوباره میگید": "دوباره می‌گید",
        "دوباره بگید": "دوباره بفرمایید",
        "تکرار میکنید": "تکرار می‌کنید",
        "میشنوم": "می‌شنوم",

        # Number word STT corrections
        "سفر": "صفر",
        "سرف": "صفر",
        "صف ": "صفر ",
        "صفره": "صفر",
        "صفرش": "صفر",

        "نومصد": "نهصد",
        "نومسد": "نهصد",
        "نحصد": "نهصد",
        "نصد": "نهصد",
        "نوهصد": "نهصد",
        "نوصد": "نهصد",
        "نودصد": "نهصد",
        "نودسد": "نهصد",
        "نه سد": "نهصد",
        "نه صد": "نهصد",

        "دویصد": "دویست",
        "دیویست": "دویست",
        "دویسد": "دویست",
        "دودست": "دویست",

        "چارصد": "چهارصد",
        "چل": "چهل",
        "شست": "شصت",
        "شیش": "شش",
        "پونصد": "پانصد",

        # Ticket / payment / support
        "بلیط": "بلیت",
        "بلیطم": "بلیتم",
        "اس ام اس": "پیامک",
        "اسمس": "پیامک",
        "SMS": "پیامک",
        "پرداختنی": "پرداختی",

        # Payment colloquial / STT fixes
        "پرداختشم انجام دادم": "پرداخت انجام دادم",
        "پرداختشم": "پرداخت",
        "پرداختش رو انجام دادم": "پرداخت رو انجام دادم",
        "پرداختشو انجام دادم": "پرداخت رو انجام دادم",
        "پرداختش انجام دادم": "پرداخت انجام دادم",
        "پرداختم": "پرداختم",
        "بلیت خرید": "خرید بلیت",

        "کانسرت": "کنسرت",
        "پشتیبونی": "پشتیبانی",
        "اوپراتور": "اپراتور",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_text_by_language(text, language):
    if language == "en":
        replacements = {
            "پیامک": "SMS",
            "بلیت": "ticket",
            "بلیتم": "my ticket",
            "کد پیگیری": "tracking code",
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
# Number extraction
# -----------------------------

def extract_digits_from_spoken_persian_number(text):
    """
    Converts common spoken Persian number patterns into digits.
    Example:
    صفر نهصد و دو دویست و یک چهل و دو هشتاد
    -> 09022014280
    """

    text = normalize_text_light(text)

    ascii_text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))

    # If text already contains digits, combine all digit pieces:
    # 0902 201 42 80 -> 09022014280
    all_digits = "".join(re.findall(r"\d", ascii_text))

    if 4 <= len(all_digits) <= 20:
        return all_digits

    ones = {
        "صفر": 0,
        "یک": 1,
        "یه": 1,
        "دو": 2,
        "سه": 3,
        "چهار": 4,
        "چار": 4,
        "پنج": 5,
        "شش": 6,
        "شیش": 6,
        "هفت": 7,
        "هشت": 8,
        "نه": 9,
    }

    teens = {
        "ده": 10,
        "یازده": 11,
        "دوازده": 12,
        "سیزده": 13,
        "چهارده": 14,
        "پانزده": 15,
        "پونزده": 15,
        "شانزده": 16,
        "هفده": 17,
        "هجده": 18,
        "نوزده": 19,
    }

    tens = {
        "بیست": 20,
        "سی": 30,
        "چهل": 40,
        "پنجاه": 50,
        "شصت": 60,
        "هفتاد": 70,
        "هشتاد": 80,
        "نود": 90,
    }

    hundreds = {
        "صد": 100,
        "یکصد": 100,
        "دویست": 200,
        "سیصد": 300,
        "چهارصد": 400,
        "پانصد": 500,
        "ششصد": 600,
        "هفتصد": 700,
        "هشتصد": 800,
        "نهصد": 900,
    }

    valid_words = set(ones) | set(teens) | set(tens) | set(hundreds) | {"و"}

    tokens = re.findall(r"[\w\u0600-\u06FF]+", text)
    tokens = [token for token in tokens if token in valid_words]

    if not tokens:
        return ""

    # Case 1: digit-by-digit style
    digit_style_tokens = [token for token in tokens if token != "و"]

    if digit_style_tokens and all(token in ones for token in digit_style_tokens):
        digits = "".join(str(ones[token]) for token in digit_style_tokens)

        if 4 <= len(digits) <= 20:
            return digits

    # Case 2: grouped Persian numbers
    result = ""
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "و":
            i += 1
            continue

        if token == "صفر":
            result += "0"
            i += 1
            continue

        if token in hundreds:
            value = hundreds[token]
            i += 1

            if i < len(tokens) and tokens[i] == "و":
                i += 1

            if i < len(tokens):
                next_token = tokens[i]

                if next_token in teens:
                    value += teens[next_token]
                    i += 1
                elif next_token in tens:
                    value += tens[next_token]
                    i += 1

                    if i < len(tokens) and tokens[i] == "و":
                        i += 1

                    if i < len(tokens) and tokens[i] in ones:
                        value += ones[tokens[i]]
                        i += 1
                elif next_token in ones:
                    value += ones[next_token]
                    i += 1

            result += str(value).zfill(3)
            continue

        if token in tens:
            value = tens[token]
            i += 1

            if i < len(tokens) and tokens[i] == "و":
                i += 1

            if i < len(tokens) and tokens[i] in ones:
                value += ones[tokens[i]]
                i += 1

            result += str(value).zfill(2)
            continue

        if token in teens:
            result += str(teens[token]).zfill(2)
            i += 1
            continue

        if token in ones:
            result += str(ones[token])
            i += 1
            continue

        i += 1

    if 4 <= len(result) <= 20:
        return result

    return ""


def extract_identifier_candidate(text):
    """
    Handles mixed spoken/digit identifiers.
    Examples:
    صفر 902 202 4280 -> 09022024280
    912 201 4282 -> 9122014282
    صفر نهصد و دو دویست و یک چهل و دو هشتاد -> 09022014280
    """

    text = normalize_text_light(text)
    ascii_text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))

    grouped_number_words = [
        "صد",
        "نهصد",
        "دویست",
        "سیصد",
        "چهارصد",
        "پانصد",
        "ششصد",
        "هفتصد",
        "هشتصد",
        "بیست",
        "سی",
        "چهل",
        "پنجاه",
        "شصت",
        "هفتاد",
        "هشتاد",
        "نود",
        "ده",
        "یازده",
        "دوازده",
    ]

    # If grouped words exist, use the full Persian number parser.
    # This prevents "صفر نهصد و دو..." from being reduced to "02".
    if any(word in ascii_text for word in grouped_number_words):
        parsed = extract_digits_from_spoken_persian_number(text)
        if parsed:
            return parsed

    digit_words = {
        "صفر": "0",
        "سفر": "0",
        "سرف": "0",
        "صف": "0",
        "صفره": "0",
        "صفرش": "0",
        "یک": "1",
        "یه": "1",
        "دو": "2",
        "سه": "3",
        "چهار": "4",
        "چار": "4",
        "پنج": "5",
        "شش": "6",
        "شیش": "6",
        "هفت": "7",
        "هشت": "8",
        "نه": "9",
    }

    tokens = re.findall(r"\d+|[\u0600-\u06FF]+", ascii_text)

    pieces = []

    for token in tokens:
        if token.isdigit():
            pieces.append(token)
        elif token in digit_words:
            pieces.append(digit_words[token])

    mixed_candidate = "".join(pieces)

    if 4 <= len(mixed_candidate) <= 20:
        return mixed_candidate

    return extract_digits_from_spoken_persian_number(text)


def normalize_requested_identifier(digits):
    """
    Only used after the assistant has asked for purchase phone number or tracking code.

    Iranian mobile rule:
    - 11 digits -> phone
    - 10 digits starting with 9 -> add leading 0 and treat as phone
    - otherwise -> tracking code
    """

    clean_digits = re.sub(r"\D", "", digits)

    if len(clean_digits) == 10 and clean_digits.startswith("9"):
        return "0" + clean_digits, "phone"

    if len(clean_digits) == 11:
        return clean_digits, "phone"

    return clean_digits, "tracking"


def enrich_text_with_detected_number(text):
    """
    Adds detected digit form to the text so the rest of the system can use it.
    """

    digits = extract_identifier_candidate(text)

    if digits:
        return f"{text} شماره تشخیص داده شده: {digits}", digits

    return text, ""


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

        "شماره موبایل شماره موبایلی": "شماره موبایلی",
        "شماره موبایل شماره موبایل": "شماره موبایل",
        "لطفاً شماره موبایل شماره موبایلی": "لطفاً شماره موبایلی",

        "خریدت رو بفرمایید": "شماره موبایلی که باهاش خرید انجام دادید یا کد پیگیری رو بفرمایید",
        "شماره موبایل خریدت رو": "شماره موبایلی که باهاش خرید انجام دادید",
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
# State helpers
# -----------------------------

def last_assistant_message(history):
    for item in reversed(history):
        if item.get("role") == "Assistant":
            return item.get("content", "")
    return ""


def get_pending_state(history):
    for item in reversed(history):
        if item.get("role") == "SystemState" and item.get("key") == "pending_state":
            return item.get("content", "")
    return ""


def set_pending_state(history, state):
    history.append({
        "role": "SystemState",
        "key": "pending_state",
        "content": state
    })


def clear_pending_states(history):
    return [
        item for item in history
        if not (item.get("role") == "SystemState" and item.get("key") == "pending_state")
    ]


def get_state_value(history, key, default=""):
    for item in reversed(history):
        if item.get("role") == "SystemState" and item.get("key") == key:
            return item.get("content", default)
    return default


def set_state_value(history, key, value):
    history[:] = [
        item for item in history
        if not (item.get("role") == "SystemState" and item.get("key") == key)
    ]

    history.append({
        "role": "SystemState",
        "key": key,
        "content": str(value)
    })


def clear_state_value(history, key):
    history[:] = [
        item for item in history
        if not (item.get("role") == "SystemState" and item.get("key") == key)
    ]


def get_audio_failure_count(history):
    count = 0

    for item in reversed(history):
        if item.get("role") == "SystemState" and item.get("key") == "audio_failure_count":
            try:
                count = int(item.get("content", 0))
            except ValueError:
                count = 0
            break

    return count


def set_audio_failure_count(history, count):
    set_state_value(history, "audio_failure_count", count)


def reset_audio_failure_count(history):
    clear_state_value(history, "audio_failure_count")


# -----------------------------
# Fast courtesy handler
# -----------------------------

def fast_courtesy_reply(user_text, language, history):
    """
    Handles pure greetings and courtesy phrases without calling Ollama.
    """

    text = normalize_text_light(user_text)

    issue_keywords_fa = [
        "پرداخت",
        "بلیت",
        "بلیط",
        "پیامک",
        "استرداد",
        "کد پیگیری",
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
    ]

    issue_keywords_en = [
        "payment",
        "ticket",
        "sms",
        "refund",
        "tracking",
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

    greeting_en = [
        "hi",
        "hello",
        "good morning",
        "good evening",
        "good afternoon",
    ]

    if language == "fa":
        has_greeting = any(phrase in text for phrase in greeting_fa)
        has_issue = any(keyword in text for keyword in issue_keywords_fa)

        if has_greeting and not has_issue:
            return "greeting", pick([
                "سلام، ممنونم. بفرمایید چطور می‌تونم کمک‌تون کنم؟",
                "سلام، در خدمتم. بفرمایید.",
                "سلام، ممنونم. بفرمایید چه کمکی از دستم برمیاد؟",
                "سلام، وقتتون بخیر. بفرمایید چطور می‌تونم راهنمایی‌تون کنم؟",
            ])

    else:
        lower_text = text.lower()
        has_greeting = any(phrase in lower_text for phrase in greeting_en)
        has_issue = any(keyword in lower_text for keyword in issue_keywords_en)

        if has_greeting and not has_issue:
            return "greeting", pick([
                "Hello, how can I help you today?",
                "Hi, how can I help?",
                "Hello, please go ahead.",
                "Hi, what can I help you with?",
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
                "بله، متوجه شدم. لطفاً شماره موبایل خرید یا کد پیگیری رو بفرمایید.",
                "حتماً، برای بررسی لطفاً شماره موبایل خرید یا کد پیگیری رو بفرمایید.",
                "درسته، لطفاً شماره موبایل خرید یا کد پیگیری رو بفرمایید تا بررسی بشه.",
            ])

        if has_ticket and has_missing:
            return "ticket_sms_issue", pick([
                "بله، متوجه شدم. لطفاً شماره موبایل خرید یا کد پیگیری رو بفرمایید.",
                "حتماً، برای بررسی لطفاً شماره موبایل خرید یا کد پیگیری رو بفرمایید.",
            ])

    else:
        has_payment = any(x in lower_text for x in payment_en)
        has_ticket = any(x in lower_text for x in ticket_en)
        has_missing = any(x in lower_text for x in missing_en)

        if has_payment and has_ticket and has_missing:
            return "payment_ticket_sms_issue", pick([
                "I understand. Please provide your purchase phone number or tracking code so this can be checked.",
                "Sure, please provide your purchase phone number or tracking code.",
            ])

        if has_ticket and has_missing:
            return "ticket_sms_issue", pick([
                "I understand. Please provide your purchase phone number or tracking code.",
                "Sure, please provide your purchase phone number or tracking code so this can be checked.",
            ])

    return None, None


# -----------------------------
# Number confirmation flow
# -----------------------------

def fast_number_confirmation_flow(user_text, language, history, detected_number):
    """
    Handles phone/tracking number confirmation.
    Only active after the assistant has asked for purchase phone number or tracking code.
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
        pending_number = get_state_value(history, "pending_number", "")
        pending_number_type = get_state_value(history, "pending_number_type", "tracking")

        zero_prefix_correction_fa = [
            "اولش صفر",
            "اول صفر",
            "صفر اولش",
            "صفر داره",
            "با صفر شروع",
        ]

        zero_prefix_correction_en = [
            "starts with zero",
            "start with zero",
            "zero at the beginning",
            "there is a zero first",
            "it begins with zero",
            "it starts with 0",
        ]

        if language == "fa" and pending_number and any(phrase in text for phrase in zero_prefix_correction_fa):
            corrected_number = pending_number

            if not corrected_number.startswith("0"):
                corrected_number = "0" + corrected_number

            corrected_number, pending_number_type = normalize_requested_identifier(corrected_number)

            set_state_value(history, "pending_number", corrected_number)
            set_state_value(history, "pending_number_type", pending_number_type)

            if pending_number_type == "phone":
                return (
                    "phone_prefix_zero_corrected",
                    pick([
                        f"بله، درست می‌فرمایید. پس شماره شد {corrected_number}. درسته؟",
                        f"درسته، صفر اولش رو اضافه کردم؛ شماره شد {corrected_number}. تأیید می‌کنید؟",
                    ]),
                    history
                )

            return (
                "tracking_prefix_zero_corrected",
                pick([
                    f"بله، درست می‌فرمایید. پس کد پیگیری شد {corrected_number}. درسته؟",
                    f"درسته، صفر اولش رو اضافه کردم؛ کد شد {corrected_number}. تأیید می‌کنید؟",
                ]),
                history
            )

        if language == "en" and pending_number and any(phrase in lower_text for phrase in zero_prefix_correction_en):
            corrected_number = pending_number

            if not corrected_number.startswith("0"):
                corrected_number = "0" + corrected_number

            corrected_number, pending_number_type = normalize_requested_identifier(corrected_number)

            set_state_value(history, "pending_number", corrected_number)
            set_state_value(history, "pending_number_type", pending_number_type)

            if pending_number_type == "phone":
                return (
                    "phone_prefix_zero_corrected",
                    pick([
                        f"Right, I added the zero at the beginning. So the phone number is {corrected_number}. Is that right?",
                        f"Got it, so it starts with zero: {corrected_number}. Is that the correct phone number?",
                    ]),
                    history
                )

            return (
                "tracking_prefix_zero_corrected",
                pick([
                    f"Right, I added the zero at the beginning. So the tracking code is {corrected_number}. Is that right?",
                    f"Got it, so it starts with zero: {corrected_number}. Is that the correct tracking code?",
                ]),
                history
            )

        # If user gives a new number while we are waiting for confirmation,
        # treat it as a correction, not as a yes/no confirmation.
        if detected_number:
            corrected_number, corrected_number_type = normalize_requested_identifier(detected_number)

            set_state_value(history, "pending_number", corrected_number)
            set_state_value(history, "pending_number_type", corrected_number_type)

            if language == "en":
                if corrected_number_type == "phone":
                    return (
                        "phone_corrected_ask_confirmation",
                        pick([
                            f"Okay, so the phone number is {corrected_number}. Is that right?",
                            f"Okay, so {corrected_number}; is that the correct phone number?",
                        ]),
                        history
                    )

                return (
                    "tracking_corrected_ask_confirmation",
                    pick([
                        f"Okay, so the tracking code is {corrected_number}. Is that right?",
                        f"Okay, so {corrected_number}; is that the correct tracking code?",
                    ]),
                    history
                )

            if corrected_number_type == "phone":
                return (
                    "phone_corrected_ask_confirmation",
                    pick([
                        f"بله، پس شماره شد {corrected_number}. درسته؟",
                        f"بله، پس {corrected_number}؛ همین شماره درسته؟",
                    ]),
                    history
                )

            return (
                "tracking_corrected_ask_confirmation",
                pick([
                    f"بله، پس کد پیگیری شد {corrected_number}. درسته؟",
                    f"بله، پس {corrected_number}؛ همین کد درسته؟",
                ]),
                history
            )

        if language == "fa":
            is_negative = any(phrase in text for phrase in negative_fa)
            is_positive = any(phrase in text for phrase in positive_fa)

            if is_negative:
                history[:] = clear_pending_states(history)
                clear_state_value(history, "pending_number")
                clear_state_value(history, "pending_number_type")

                return (
                    "identifier_rejected",
                    "ببخشید، پس لطفاً شماره موبایل خرید یا کد پیگیری رو یک بار دیگه بفرمایید.",
                    history
                )

            if is_positive:
                history[:] = clear_pending_states(history)
                clear_state_value(history, "pending_number")
                clear_state_value(history, "pending_number_type")

                if pending_number_type == "phone":
                    reply = "ممنونم، شماره موبایل ثبت شد. برای ادامه باید این مورد در سیستم سفارش‌ها بررسی بشه."
                else:
                    reply = "ممنونم، کد پیگیری ثبت شد. برای ادامه باید این مورد در سیستم سفارش‌ها بررسی بشه."

                return (
                    "identifier_confirmed",
                    reply,
                    history
                )

        else:
            is_negative = any(phrase in lower_text for phrase in negative_en)
            is_positive = any(phrase in lower_text for phrase in positive_en)

            if is_negative:
                history[:] = clear_pending_states(history)
                clear_state_value(history, "pending_number")
                clear_state_value(history, "pending_number_type")

                return (
                    "identifier_rejected",
                    "Sorry about that. Please say the phone number or tracking code one more time.",
                    history
                )

            if is_positive:
                history[:] = clear_pending_states(history)
                clear_state_value(history, "pending_number")
                clear_state_value(history, "pending_number_type")

                if pending_number_type == "phone":
                    reply = "Thanks, the phone number is saved. To continue, this needs to be checked in the order system."
                else:
                    reply = "Thanks, the tracking code is saved. To continue, this needs to be checked in the order system."

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
            "شماره رو",
            "شماره را",
            "کد پیگیری",
            "phone number",
            "tracking code",
        ])

        if asked_for_number:
            normalized_identifier, identifier_type = normalize_requested_identifier(detected_number)

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
                    "tracking_detected_ask_confirmation",
                    pick([
                        f"Okay, so the tracking code is {normalized_identifier}. Is that right?",
                        f"Okay, so {normalized_identifier}; is that the correct tracking code?",
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
                "tracking_detected_ask_confirmation",
                pick([
                    f"بله، پس کد پیگیری شد {normalized_identifier}. درسته؟",
                    f"بله، پس {normalized_identifier}؛ همین کد درسته؟",
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

    segments, info = stt_model.transcribe(
        audio_path,
        language=None,
        task="transcribe",
        beam_size=1,
        best_of=1,
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

    elapsed = round(time.time() - start_time, 2)

    raw_text = " ".join(raw_parts).strip()
    normalized_text = normalize_text_light(raw_text)

    detected_language = detect_language_simple(normalized_text, getattr(info, "language", None))
    language_probability = round(getattr(info, "language_probability", 0), 2)

    normalized_text = clean_text_by_language(normalized_text, detected_language)

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
اگر موضوع پرداخت، بلیت، پیامک، سفارش یا استرداد است، شماره موبایل خرید یا کد پیگیری را بخواه.
اگر کاربر خودش گفته پیامک، بلیت یا اطلاعات بلیت به دستش نرسیده، دوباره نپرس که آیا به دستش رسیده یا نه.
در این حالت فقط مشکل را تأیید کن و شماره موبایل خرید یا کد پیگیری را بخواه.
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
If the issue is about payment, ticket, SMS, order, or refund, ask for the purchase phone number or tracking code.
If the user already said they did not receive the ticket, SMS, or ticket information, do not ask again whether it was received.
Only acknowledge the issue and ask for the purchase phone number or tracking code.
Do not add extra confirmation questions like "did you receive it?" or "was it delivered?"
If the user provides a phone number or tracking code, acknowledge receiving it but do not claim that you checked anything.
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
            "کد پیگیری",
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
        return "بله، متوجه شدم. برای بررسی دقیق، لطفاً شماره موبایل خرید یا کد پیگیری رو بفرمایید."

    if language == "en" and any(claim.lower() in reply.lower() for claim in forbidden_claims_en):
        return "I understand. To check this properly, please provide your purchase phone number or tracking code."

    if has_phone_or_tracking(user_text):
        if language == "en":
            return "Thanks, I received it. To continue, this needs to be checked in the order system."
        return "ممنون، اطلاعات رو دریافت کردم. برای ادامه باید این مورد در سیستم سفارش‌ها بررسی بشه."

    if language == "fa":
        tone_replacements = {
            "شماره موبایل شماره موبایلی": "شماره موبایلی",
            "شماره موبایل شماره موبایل": "شماره موبایل",
            "لطفاً شماره موبایل شماره موبایلی": "لطفاً شماره موبایلی",
            "خریدت رو بفرمایید": "شماره موبایلی که باهاش خرید انجام دادید یا کد پیگیری رو بفرمایید",
            "شماره موبایل خریدت رو": "شماره موبایلی که باهاش خرید انجام دادید",
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

def process_voice(audio_path, history):
    if history is None:
        history = []

    if audio_path is None:
        return "لطفاً یک صدا ضبط کنید یا فایل صوتی آپلود کنید.", history

    total_start = time.time()

    raw_text, normalized_text, stt_time, detected_language, language_probability = transcribe_audio(audio_path)

    normalized_text, detected_number = enrich_text_with_detected_number(normalized_text)

    number_intent, number_reply, history = fast_number_confirmation_flow(
        user_text=normalized_text,
        language=detected_language,
        history=history,
        detected_number=detected_number
    )

    if number_reply:
        assistant_reply = clean_persian_tone(number_reply, detected_language)
        llm_time = 0
        source = f"Fast Number Flow / {number_intent}"

        history.append({"role": "UserLanguage", "content": detected_language})
        history.append({"role": "User", "content": normalized_text})
        history.append({"role": "Assistant", "content": assistant_reply})

        total_time = round(time.time() - total_start, 2)

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

    audio_intent, audio_reply, history = fast_audio_issue_flow(
        user_text=normalized_text,
        language=detected_language,
        history=history
    )

    if audio_reply:
        assistant_reply = clean_persian_tone(audio_reply, detected_language)
        llm_time = 0
        source = f"Fast Audio Flow / {audio_intent}"

        history.append({"role": "UserLanguage", "content": detected_language})
        history.append({"role": "User", "content": normalized_text})
        history.append({"role": "Assistant", "content": assistant_reply})

        total_time = round(time.time() - total_start, 2)

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

    if not normalized_text:
        reply_language = get_last_user_language(history)
        assistant_reply = unclear_stt_reply(reply_language)
        total_time = round(time.time() - total_start, 2)

        output_text = build_output_text(
            raw_text=raw_text,
            normalized_text=normalized_text,
            detected_language=detected_language,
            language_probability=language_probability,
            assistant_reply=assistant_reply,
            source="Bad STT / Empty Text",
            stt_time=stt_time,
            llm_time=0,
            total_time=total_time,
            history=history,
            detected_number=detected_number
        )

        return output_text, history

    if looks_like_bad_stt(normalized_text):
        reply_language = get_last_user_language(history)
        assistant_reply = unclear_stt_reply(reply_language)
        total_time = round(time.time() - total_start, 2)

        output_text = build_output_text(
            raw_text=raw_text,
            normalized_text=normalized_text,
            detected_language=detected_language,
            language_probability=language_probability,
            assistant_reply=assistant_reply,
            source="Bad STT Detector / Memory Not Updated",
            stt_time=stt_time,
            llm_time=0,
            total_time=total_time,
            history=history,
            detected_number=detected_number
        )

        return output_text, history

    fast_intent, fast_reply = fast_courtesy_reply(
        user_text=normalized_text,
        language=detected_language,
        history=history
    )

    if fast_reply:
        assistant_reply = clean_persian_tone(fast_reply, detected_language)
        llm_time = 0
        source = f"Fast Courtesy Handler / {fast_intent}"

    else:
        ticket_intent, ticket_reply = fast_ticket_sms_issue_reply(
            user_text=normalized_text,
            language=detected_language,
            history=history
        )

        if ticket_reply:
            assistant_reply = clean_persian_tone(ticket_reply, detected_language)
            llm_time = 0
            source = f"Fast Ticket/SMS Handler / {ticket_intent}"

        else:
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

    history.append({"role": "UserLanguage", "content": detected_language})
    history.append({"role": "User", "content": normalized_text})
    history.append({"role": "Assistant", "content": assistant_reply})

    total_time = round(time.time() - total_start, 2)

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


def reset_conversation():
    return [], "Conversation reset."


with gr.Blocks(title="Bilingual AI Call Center Demo") as demo:
    gr.Markdown("# Bilingual AI Call Center Demo")
    gr.Markdown("Voice → Whisper STT → Number Confirmation → Bad STT Filter → Fast Ticket/SMS Handler → Ollama-first reasoning → Guardrails")

    state = gr.State([])

    audio_input = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Customer Voice"
    )

    output_box = gr.Textbox(
        label="Full Output",
        lines=24
    )

    with gr.Row():
        submit_btn = gr.Button("Submit")
        reset_btn = gr.Button("Reset Conversation")

    submit_btn.click(
        fn=process_voice,
        inputs=[audio_input, state],
        outputs=[output_box, state]
    )

    reset_btn.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[state, output_box]
    )

demo.launch()