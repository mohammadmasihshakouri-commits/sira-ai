from __future__ import annotations

import os


def build_voice_opening_greeting() -> str:
    brand_name = os.getenv("SIRA_VOICE_BRAND_NAME", "سینماتیکت").strip()
    return f"سلام، {brand_name}، بفرمایید."