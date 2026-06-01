from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import edge_tts


DEFAULT_VOICE = "fa-IR-FaridNeural"


async def _synthesize_to_file_async(
    text: str,
    output_path: str,
    voice: str = DEFAULT_VOICE,
) -> str:
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)
    return output_path


def synthesize_speech_to_file(
    text: str,
    voice: str = DEFAULT_VOICE,
) -> str:
    clean_text = str(text or "").strip()

    if not clean_text:
        return ""

    output_path = str(
        Path(tempfile.gettempdir())
        / f"sira_tts_{int(time.time() * 1000)}.mp3"
    )

    try:
        asyncio.run(
            _synthesize_to_file_async(
                text=clean_text,
                output_path=output_path,
                voice=voice,
            )
        )
    except Exception as error:
        print("TTS error:", error)
        return ""

    return output_path
