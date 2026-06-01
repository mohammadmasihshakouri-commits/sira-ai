from __future__ import annotations

import re
from typing import Any

from core.normalization import normalize_text_light
from core.number_parser import (
    extract_identifier_candidate,
    normalize_requested_identifier,
)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_name_text(text: str, phone: str) -> str:
    cleaned = normalize_text_light(text)

    if phone:
        phone_digits = re.sub(r"\D", "", phone)
        cleaned = cleaned.replace(phone, " ")

        if phone_digits:
            cleaned = cleaned.replace(phone_digits, " ")

    cleaned = re.sub(r"[0-9۰-۹٠-٩]+", " ", cleaned)

    removable_phrases = [
        "اسمم",
        "اسم من",
        "نامم",
        "نام من",
        "من",
        "هستم",
        "شماره من",
        "شماره موبایل من",
        "موبایلم",
        "تلفنم",
        "شماره",
        "موبایل",
        "تلفن",
        "اینم",
        "این",
    ]

    for phrase in removable_phrases:
        cleaned = cleaned.replace(phrase, " ")

    cleaned = re.sub(r"[،,.]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def parse_customer_identity(user_text: str) -> dict[str, Any]:
    text = _safe_text(user_text)

    detected = extract_identifier_candidate(text)
    phone = ""

    if detected:
        normalized_identifier, identifier_type = normalize_requested_identifier(detected)

        if identifier_type == "phone":
            phone = normalized_identifier

    full_name = _clean_name_text(text, phone)

    return {
        "full_name": full_name,
        "phone": phone,
        "has_full_name": bool(full_name),
        "has_phone": bool(phone),
        "is_complete": bool(full_name and phone),
    }
