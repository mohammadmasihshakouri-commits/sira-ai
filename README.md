\# Allo Realtime Voice Agent



Allo is a realtime multilingual AI voice agent prototype for natural business phone conversations.



It combines live browser audio streaming, speech-to-text, intent routing, playbook-based business logic, guardrails, and conversational AI responses.



\## What it does



\- Streams microphone audio from the browser through WebSocket

\- Transcribes speech using faster-whisper

\- Detects user intent and routes conversations

\- Follows business-specific playbooks and rules

\- Handles reservation codes, phone numbers, confirmations, and retries

\- Supports Persian and English, with multilingual expansion in mind

\- Provides a polished Gradio demo UI and a realtime WebSocket demo



\## Core Features



\- Realtime audio streaming

\- Live transcript rendering

\- AI response bubbles

\- Persian STT normalization layer

\- Number and reservation-code parsing

\- Guardrails for out-of-domain questions

\- Playbook-driven conversation flows

\- Modular backend structure



\## Architecture



```text

Browser Microphone

&nbsp;     ↓

WebSocket Audio Stream

&nbsp;     ↓

Realtime STT Pipeline

&nbsp;     ↓

Normalization Layer

&nbsp;     ↓

Intent / Playbook Engine

&nbsp;     ↓

AI Response Orchestration

&nbsp;     ↓

Live Conversation UI

