from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from core.normalization import normalize_text_light


DEFAULT_TIMEZONE = "Asia/Tehran"


PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


WEEKDAYS_FA = {
    "سه شنبه": 1,
    "سه‌شنبه": 1,
    "چهارشنبه": 2,
    "پنجشنبه": 3,
    "یکشنبه": 6,
    "دوشنبه": 0,
    "شنبه": 5,
    "جمعه": 4,
}


PERIOD_HINTS = {
    "صبح": {
        "label": "morning",
        "default_hour": 10,
    },
    "ظهر": {
        "label": "noon",
        "default_hour": 12,
    },
    "بعدازظهر": {
        "label": "afternoon",
        "default_hour": 16,
    },
    "بعد از ظهر": {
        "label": "afternoon",
        "default_hour": 16,
    },
    "عصر": {
        "label": "evening",
        "default_hour": 17,
    },
    "شب": {
        "label": "night",
        "default_hour": 20,
    },
}


PERSIAN_NUMBER_WORDS = {
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
    "ده": 10,
    "یازده": 11,
    "دوازده": 12,
}


def _now(timezone_name: str = DEFAULT_TIMEZONE) -> datetime:
    return datetime.now(ZoneInfo(timezone_name))


def _clean_text(text: str) -> str:
    text = normalize_text_light(text or "")
    text = text.translate(PERSIAN_DIGITS)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _next_weekday(base_date: datetime, target_weekday: int) -> datetime:
    days_ahead = target_weekday - base_date.weekday()

    if days_ahead <= 0:
        days_ahead += 7

    return base_date + timedelta(days=days_ahead)


def _extract_date(text: str, base_datetime: datetime) -> tuple[datetime | None, str]:
    if "پس فردا" in text:
        return base_datetime + timedelta(days=2), "پس فردا"

    if "فردا" in text:
        return base_datetime + timedelta(days=1), "فردا"

    if "امروز" in text:
        return base_datetime, "امروز"

    for weekday_text, weekday_index in WEEKDAYS_FA.items():
        if re.search(rf"(?<!\S){re.escape(weekday_text)}(?!\S)", text):
            return _next_weekday(base_datetime, weekday_index), weekday_text

    return None, ""


def _word_to_hour(text: str) -> int | None:
    for word, value in PERSIAN_NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return value

    return None


def _extract_period(text: str) -> dict[str, Any] | None:
    for phrase, period in PERIOD_HINTS.items():
        if phrase in text:
            return {
                "text": phrase,
                "label": period["label"],
                "default_hour": period["default_hour"],
            }

    return None


def _extract_time(text: str) -> tuple[int | None, int, str, bool]:
    minute = 0
    is_ambiguous = False

    half_markers = [
        "و نیم",
        "نیم",
        "و سی",
        "و 30",
    ]

    if any(marker in text for marker in half_markers):
        minute = 30

    numeric_patterns = [
        r"ساعت\s+(\d{1,2})(?::(\d{1,2}))?",
        r"\b(\d{1,2}):(\d{1,2})\b",
    ]

    for pattern in numeric_patterns:
        match = re.search(pattern, text)

        if match:
            hour = int(match.group(1))

            if match.lastindex and match.lastindex >= 2 and match.group(2):
                minute = int(match.group(2))

            if 1 <= hour <= 12:
                is_ambiguous = True

            return hour, minute, match.group(0), is_ambiguous

    if "ساعت" in text:
        after_hour_word = text.split("ساعت", 1)[1]
        hour = _word_to_hour(after_hour_word)

        if hour is not None:
            if 1 <= hour <= 12:
                is_ambiguous = True

            return hour, minute, f"ساعت {after_hour_word.strip()}", is_ambiguous

    return None, minute, "", False


def _resolve_hour_with_period(hour: int, period: dict[str, Any] | None) -> tuple[int, bool]:
    if hour > 12:
        return hour, False

    if not period:
        return hour, True

    label = period["label"]

    if label in {"afternoon", "evening", "night"} and hour < 12:
        return hour + 12, False

    if label in {"morning", "noon"}:
        return hour, False

    return hour, True


def parse_datetime_hint(
    text: str,
    base_datetime: datetime | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    base_datetime = base_datetime or _now(timezone_name)
    cleaned_text = _clean_text(text)

    date_value, date_text = _extract_date(cleaned_text, base_datetime)
    period = _extract_period(cleaned_text)
    hour, minute, time_text, time_is_ambiguous = _extract_time(cleaned_text)

    if hour is None and period:
        hour = int(period["default_hour"])
        minute = 0
        time_text = period["text"]
        time_is_ambiguous = False

    resolved_hour = None
    is_ambiguous = False
    clarification_question = ""

    if hour is not None:
        resolved_hour, is_ambiguous = _resolve_hour_with_period(hour, period)

    if time_is_ambiguous and not period:
        is_ambiguous = True
        clarification_question = "منظورتون صبحه یا عصر؟"

    has_date_hint = date_value is not None
    has_time_hint = resolved_hour is not None
    has_datetime_hint = has_date_hint or has_time_hint or period is not None

    start_datetime = None

    if date_value is not None and resolved_hour is not None and not is_ambiguous:
        start_datetime = date_value.replace(
            hour=resolved_hour,
            minute=minute,
            second=0,
            microsecond=0,
        )

    return {
        "has_datetime_hint": has_datetime_hint,
        "has_date_hint": has_date_hint,
        "has_time_hint": has_time_hint,
        "date_text": date_text,
        "time_text": time_text,
        "period_text": period["text"] if period else "",
        "period_label": period["label"] if period else "",
        "hour": resolved_hour,
        "minute": minute if resolved_hour is not None else None,
        "timezone": timezone_name,
        "start_datetime": start_datetime.isoformat() if start_datetime else "",
        "is_ambiguous": is_ambiguous,
        "clarification_needed": bool(is_ambiguous),
        "clarification_question": clarification_question,
        "normalized_text": cleaned_text,
    }
