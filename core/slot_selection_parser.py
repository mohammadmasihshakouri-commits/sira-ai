from __future__ import annotations

import re
from typing import Any

from core.normalization import normalize_text_light


PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


ORDINAL_WORDS = {
    "اول": 1,
    "اولی": 1,
    "یکم": 1,
    "اولین": 1,
    "دوم": 2,
    "دومی": 2,
    "دومین": 2,
    "سوم": 3,
    "سومی": 3,
    "سومین": 3,
    "چهارم": 4,
    "چهارمی": 4,
    "پنجم": 5,
    "پنجمی": 5,
}


NUMBER_WORDS = {
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


def _clean_text(text: str) -> str:
    text = normalize_text_light(text or "")
    text = text.translate(PERSIAN_DIGITS)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_hour_from_slot(slot: dict[str, Any]) -> int | None:
    start_datetime = str(slot.get("start_datetime", "") or "")

    match = re.search(r"T(\d{2}):", start_datetime)
    if match:
        return int(match.group(1))

    display = str(slot.get("display", "") or "")
    display = _clean_text(display)

    match = re.search(r"\b(\d{1,2})\b", display)
    if match:
        hour = int(match.group(1))

        if "عصر" in display or "شب" in display or "بعدازظهر" in display or "بعد از ظهر" in display:
            if hour < 12:
                hour += 12

        return hour

    for word, value in NUMBER_WORDS.items():
        if re.search(rf"(?<!\S){re.escape(word)}(?!\S)", display):
            if "عصر" in display or "شب" in display or "بعدازظهر" in display or "بعد از ظهر" in display:
                if value < 12:
                    return value + 12

            return value

    return None


def _extract_requested_index(text: str) -> int | None:
    for word, index in ORDINAL_WORDS.items():
        if re.search(rf"(?<!\S){re.escape(word)}(?!\S)", text):
            return index

    index_patterns = [
        r"گزینه\s+(\d+)",
        r"شماره\s+(\d+)",
        r"مورد\s+(\d+)",
        r"\b(\d+)\b",
    ]

    for pattern in index_patterns:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))

            if 1 <= value <= 5:
                return value

    return None


def _extract_requested_hour(text: str) -> int | None:
    hour_patterns = [
        r"ساعت\s+(\d{1,2})",
        r"\b(\d{1,2})\b",
    ]

    for pattern in hour_patterns:
        match = re.search(pattern, text)
        if match:
            hour = int(match.group(1))

            if "صبح" in text:
                return hour

            if "عصر" in text or "شب" in text or "بعدازظهر" in text or "بعد از ظهر" in text:
                if hour < 12:
                    return hour + 12

            return hour

    for word, value in NUMBER_WORDS.items():
        if re.search(rf"(?<!\S){re.escape(word)}(?!\S)", text):
            if "صبح" in text:
                return value

            if "عصر" in text or "شب" in text or "بعدازظهر" in text or "بعد از ظهر" in text:
                if value < 12:
                    return value + 12

            return value

    return None


def parse_slot_selection(
    user_text: str,
    offered_slots: list[dict[str, Any]],
) -> dict[str, Any]:
    text = _clean_text(user_text)

    if not offered_slots:
        return {
            "selected": False,
            "selected_slot": None,
            "selection_method": "",
            "clarification_needed": True,
            "clarification_question": "لطفاً بفرمایید کدوم تایم رو انتخاب می‌کنید؟",
        }

    requested_index = _extract_requested_index(text)

    if requested_index is not None:
        for slot in offered_slots:
            slot_index = int(slot.get("index", 0) or 0)

            if slot_index == requested_index:
                return {
                    "selected": True,
                    "selected_slot": slot,
                    "selection_method": "index",
                    "clarification_needed": False,
                    "clarification_question": "",
                }

    requested_hour = _extract_requested_hour(text)

    if requested_hour is not None:
        matches = []

        for slot in offered_slots:
            slot_hour = _extract_hour_from_slot(slot)

            if slot_hour == requested_hour:
                matches.append(slot)

            elif requested_hour < 12 and slot_hour == requested_hour + 12:
                matches.append(slot)

        if len(matches) == 1:
            return {
                "selected": True,
                "selected_slot": matches[0],
                "selection_method": "hour",
                "clarification_needed": False,
                "clarification_question": "",
            }

        if len(matches) > 1:
            return {
                "selected": False,
                "selected_slot": None,
                "selection_method": "hour_ambiguous",
                "clarification_needed": True,
                "clarification_question": "چند تایم با این ساعت پیدا کردم. لطفاً شماره گزینه رو بفرمایید.",
            }

    return {
        "selected": False,
        "selected_slot": None,
        "selection_method": "",
        "clarification_needed": True,
        "clarification_question": "لطفاً یکی از تایم‌های پیشنهادی رو انتخاب کنید.",
    }
