import os
import json
import asyncio
import tempfile
from faster_whisper import WhisperModel
import numpy as np
from app import process_text
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
from app import process_text


app = FastAPI()
model = WhisperModel(
    r"C:\Users\Admin\Desktop\voice-demo\models\faster-whisper-large-v3",
    device="cuda",
    compute_type="float16"
)

@app.get("/")
async def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Allo Realtime Demo</title>
</head>
<body>
    <h1>Allo Realtime Voice Demo</h1>

    <button id="startBtn">Start</button>
    <button id="stopBtn">Stop</button>

    <div id="transcript"></div>

    <script>
        const log = document.getElementById("log");
        let socket;
        let mediaRecorder;

function writeTranscript(text, speaker="customer") {

    const transcript = document.getElementById("transcript");

    const bubble = document.createElement("div");

    bubble.style.background =
        speaker === "customer"
        ? "#111827"
        : "#1E293B";

    bubble.style.color = "white";
    bubble.style.padding = "18px";
    bubble.style.borderRadius = "18px";
    bubble.style.marginTop = "16px";
    bubble.style.fontSize = "18px";
    bubble.style.lineHeight = "1.8";

    bubble.innerHTML = `
        <strong>
            ${speaker === "customer" ? "Customer" : "Allo"}
        </strong>
        <br><br>
        ${text}
    `;

    transcript.appendChild(bubble);

    transcript.scrollTop = transcript.scrollHeight;
}

        document.getElementById("startBtn").onclick = async () => {
            write("Starting realtime session...");

            socket = new WebSocket("ws://127.0.0.1:8000/ws");

            socket.onopen = () => {
                write("WebSocket connected.");
            };

socket.onmessage = (event) => {

    const data = JSON.parse(event.data);

if (data.type === "transcript") {
    writeTranscript(data.text, "customer");
}
if (data.type === "assistant") {
    writeTranscript(data.text, "assistant");
}
    if (data.type === "error") {
        writeTranscript("ERROR: " + data.message);
    }
};

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            mediaRecorder = new MediaRecorder(stream, {
                mimeType: "audio/webm"
            });

            mediaRecorder.ondataavailable = async (event) => {
                if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {
                    const buffer = await event.data.arrayBuffer();
                    socket.send(buffer);
                }
            };

            mediaRecorder.start(1000);
            write("Recording started.");
        };

        document.getElementById("stopBtn").onclick = () => {
            if (mediaRecorder) {
                mediaRecorder.stop();
                write("Recording stopped.");
            }

            if (socket) {
                socket.close();
                write("WebSocket closed.");
            }
        };
    </script>
</body>
</html>
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Connected to Allo realtime backend.")

    chunk_count = 0
    audio_buffer = bytearray()

    try:
        while True:

            audio_chunk = await websocket.receive_bytes()

            chunk_count += 1

            audio_buffer.extend(audio_chunk)

            if len(audio_buffer) > 120000:

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
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(
        "realtime_demo:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )