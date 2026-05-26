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
    <title>Sira Realtime Voice Demo</title>
</head>
<body>
    <h1>Sira Realtime Voice Demo</h1>

    <button id="startBtn">Start</button>
    <button id="stopBtn">Stop</button>

    <h3>Debug Log</h3>
    <pre id="log" style="background:#111827;color:white;padding:16px;border-radius:12px;min-height:140px;"></pre>

    <h3>Transcript</h3>
    <div id="transcript"></div>

    <script>
        const log = document.getElementById("log");
        const transcript = document.getElementById("transcript");

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
                <strong>${speaker === "customer" ? "Customer" : "Sira"}</strong>
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
""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Connected to Sira realtime backend.")

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