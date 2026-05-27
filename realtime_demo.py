import os
import json
import asyncio
import tempfile

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
import numpy as np
import uvicorn

from app import process_text
from core.runtime_config import (
    get_platform_name,
    get_agent_label,
)

app = FastAPI()
PLATFORM_NAME = get_platform_name()
AGENT_LABEL = get_agent_label()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WHISPER_MODEL_PATH = os.getenv(
    "WHISPER_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "faster-whisper-large-v3"),
)

STT_DEVICE = os.getenv("STT_DEVICE", "cuda")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16")

model = WhisperModel(
    WHISPER_MODEL_PATH,
    device=STT_DEVICE,
    compute_type=STT_COMPUTE_TYPE,
)

@app.get("/")
async def home():
    page = """
<!DOCTYPE html>
<html>
<head>
    <title>__PLATFORM_NAME__ Realtime Voice Demo</title>
</head>
<body>
    <h1>__PLATFORM_NAME__ Realtime Voice Demo</h1>
    <p>Current agent: <strong>__AGENT_LABEL__</strong></p>

    <button id="startBtn">Start</button>
    <button id="stopBtn">Stop</button>

    <h3>Debug Log</h3>
    <pre id="log" style="background:#111827;color:white;padding:16px;border-radius:12px;min-height:140px;"></pre>

    <h3>Transcript</h3>
    <div id="transcript"></div>

    <script>
        const log = document.getElementById("log");
        const transcript = document.getElementById("transcript");

        const assistantLabel = "__AGENT_LABEL__";

        let socket = null;
        let mediaRecorder = null;
        let mediaStream = null;

        function write(message) {
            log.textContent += message + "\\n";
        }

        function writeTranscript(text, speaker = "customer") {
            const bubble = document.createElement("div");

            bubble.style.background = speaker === "customer" ? "#111827" : "#1E293B";
            bubble.style.color = "white";
            bubble.style.padding = "18px";
            bubble.style.borderRadius = "18px";
            bubble.style.marginTop = "16px";
            bubble.style.fontSize = "18px";
            bubble.style.lineHeight = "1.8";

            bubble.innerHTML = `
                <strong>${speaker === "customer" ? "Customer" : assistantLabel}</strong>
                <br><br>
                ${text}
            `;

            transcript.appendChild(bubble);
            transcript.scrollTop = transcript.scrollHeight;
        }

        document.getElementById("startBtn").onclick = async () => {
            try {
                write("Start clicked.");

                socket = new WebSocket("ws://127.0.0.1:8000/ws");

                socket.onopen = () => {
                    write("WebSocket connected.");
                };

                socket.onerror = (error) => {
                    write("WebSocket error.");
                    console.error(error);
                };

                socket.onclose = () => {
                    write("WebSocket closed.");
                };

                socket.onmessage = (event) => {
                    write("Server: " + event.data);

                    try {
                        const data = JSON.parse(event.data);

                        if (data.type === "transcript") {
                            writeTranscript(data.text, "customer");
                        }

                        if (data.type === "assistant") {
                            writeTranscript(data.text, "assistant");
                        }

                        if (data.type === "error") {
                            writeTranscript("ERROR: " + data.message, "assistant");
                        }
                    } catch (e) {
                        write("Could not parse server message.");
                    }
                };

                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                write("Microphone permission granted.");

                mediaRecorder = new MediaRecorder(mediaStream, {
                    mimeType: "audio/webm"
                });

                mediaRecorder.ondataavailable = async (event) => {
                    if (event.data.size > 0 && socket && socket.readyState === WebSocket.OPEN) {
                        const buffer = await event.data.arrayBuffer();
                        socket.send(buffer);
                        write("Sent audio chunk: " + event.data.size + " bytes");
                    }
                };

                mediaRecorder.start(1000);
                write("Recording started.");
            } catch (error) {
                write("ERROR: " + error.message);
                console.error(error);
            }
        };

        document.getElementById("stopBtn").onclick = () => {
            write("Stop clicked.");

            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
                write("Recording stopped.");
            }

            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                write("Microphone stopped.");
            }

            if (socket) {
                socket.close();
                write("WebSocket closed by user.");
            }
        };
    </script>
</body>
</html>
"""
    page = page.replace("__PLATFORM_NAME__", PLATFORM_NAME)
    page = page.replace("__AGENT_LABEL__", AGENT_LABEL)

    return HTMLResponse(page)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(
    f"Connected to {PLATFORM_NAME} realtime backend. Current agent: {AGENT_LABEL}."
)

    chunk_count = 0
    audio_buffer = bytearray()

    try:
        while True:

            audio_chunk = await websocket.receive_bytes()

            chunk_count += 1

            audio_buffer.extend(audio_chunk)

            if len(audio_buffer) > 30000:

                temp_audio = tempfile.NamedTemporaryFile(
                    suffix=".webm",
                    delete=False
                )

                temp_audio.write(audio_buffer)
                temp_audio.close()

                try:
                    segments, info = model.transcribe(
                        temp_audio.name,
                        beam_size=1
                    )

                    transcript = " ".join(
                        segment.text for segment in segments
                    ).strip()

                    if transcript and transcript != last_transcript:

                        last_transcript = transcript

                        await websocket.send_text(
                            json.dumps({
                                "type": "transcript",
                                "text": transcript
                            })
                        )
                        assistant_reply, source = process_text(
                            transcript
                        )

                        await websocket.send_text(
                            json.dumps({
                                "type": "assistant",
                                "text": assistant_reply
                            })
                        )
                except Exception as e:

                    await websocket.send_text(
                        json.dumps({
                            "type": "error",
                            "message": str(e)
                        })
                    )

                finally:
                    os.remove(temp_audio.name)

                audio_buffer = bytearray()
        last_transcript = ""
    except Exception:
        pass


if __name__ == "__main__":
    uvicorn.run(
        "realtime_demo:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )