from __future__ import annotations

import os
import ssl
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CREDENTIALS_DIR = PROJECT_ROOT / "credentials"

GOOGLE_CREDENTIALS_PATH = CREDENTIALS_DIR / "google_calendar_credentials.json"
GOOGLE_TOKEN_PATH = CREDENTIALS_DIR / "google_oauth_token.json"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/calendar",
]

CLIENTS_SHEET = "Clients"
APPOINTMENTS_SHEET = "Appointments"
CALL_LOG_SHEET = "CallLog"


load_dotenv(PROJECT_ROOT / ".env")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_phone(phone: str) -> str:
    digits = "".join(char for char in str(phone) if char.isdigit())

    if len(digits) == 10 and digits.startswith("9"):
        return f"0{digits}"

    return digits


def _get_spreadsheet_id() -> str:
    spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")

    if not spreadsheet_id:
        raise RuntimeError("GOOGLE_SHEETS_SPREADSHEET_ID is missing in .env")

    return spreadsheet_id


def _get_credentials() -> Credentials:
    creds = None

    if GOOGLE_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(GOOGLE_TOKEN_PATH), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not GOOGLE_CREDENTIALS_PATH.exists():
            raise FileNotFoundError(
                f"Google credentials file not found: {GOOGLE_CREDENTIALS_PATH}"
            )

        flow = InstalledAppFlow.from_client_secrets_file(
            str(GOOGLE_CREDENTIALS_PATH),
            SCOPES,
        )
        creds = flow.run_local_server(port=0)

    CREDENTIALS_DIR.mkdir(exist_ok=True)
    GOOGLE_TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    return creds


def _get_sheets_service():
    creds = _get_credentials()
    return build("sheets", "v4", credentials=creds)


def _execute_with_retry(request, retries: int = 5, delay_seconds: float = 2.0):
    last_error = None

    retryable_errors = (
        ssl.SSLEOFError,
        ConnectionResetError,
        TimeoutError,
    )

    for attempt in range(1, retries + 1):
        try:
            return request.execute()
        except retryable_errors as error:
            last_error = error

            if attempt == retries:
                break

            time.sleep(delay_seconds)

    raise last_error


def _read_range(range_name: str) -> list[list[str]]:
    service = _get_sheets_service()
    spreadsheet_id = _get_spreadsheet_id()

    request = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_name)
    )

    result = _execute_with_retry(request)

    return result.get("values", [])


def _append_row(sheet_name: str, row: list[Any]) -> dict[str, Any]:
    service = _get_sheets_service()
    spreadsheet_id = _get_spreadsheet_id()

    request = (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=spreadsheet_id,
            range=f"{sheet_name}!A:Z",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": [row]},
        )
    )

    result = _execute_with_retry(request)

    return result


def get_sheet_headers(sheet_name: str) -> list[str]:
    rows = _read_range(f"{sheet_name}!1:1")

    if not rows:
        return []

    return rows[0]


def find_client_by_phone(phone: str) -> dict[str, Any] | None:
    normalized_phone = normalize_phone(phone)
    rows = _read_range(f"{CLIENTS_SHEET}!A:G")

    if not rows:
        return None

    headers = rows[0]
    data_rows = rows[1:]

    for index, row in enumerate(data_rows, start=2):
        padded_row = row + [""] * (len(headers) - len(row))
        record = dict(zip(headers, padded_row))

        if normalize_phone(record.get("phone", "")) == normalized_phone:
            record["_row_number"] = index
            return record

    return None


def create_client(
    full_name: str,
    phone: str,
    source_channel: str = "manual_test",
    notes: str = "",
) -> dict[str, Any]:
    phone = normalize_phone(phone)
    existing_client = find_client_by_phone(phone)

    if existing_client:
        return {
            "created": False,
            "client": existing_client,
        }

    now = _now_iso()
    client_id = f"client_{uuid.uuid4().hex[:12]}"

    row = [
        client_id,
        full_name,
        phone,
        now,
        now,
        source_channel,
        notes,
    ]

    _append_row(CLIENTS_SHEET, row)

    return {
        "created": True,
        "client": {
            "client_id": client_id,
            "full_name": full_name,
            "phone": phone,
            "first_seen_at": now,
            "last_seen_at": now,
            "source_channel": source_channel,
            "notes": notes,
        },
    }


def create_call_log(
    channel: str,
    user_id: str,
    agent_key: str,
    agent_type: str,
    intent: str,
    topic: str,
    summary: str,
    outcome: str,
    client_id: str = "",
    phone: str = "",
    tool_calls: str = "",
) -> dict[str, Any]:
    phone = normalize_phone(phone) if phone else ""

    now = _now_iso()
    log_id = f"log_{uuid.uuid4().hex[:12]}"

    row = [
        log_id,
        now,
        channel,
        user_id,
        agent_key,
        agent_type,
        intent,
        topic,
        client_id,
        phone,
        summary,
        outcome,
        tool_calls,
    ]

    _append_row(CALL_LOG_SHEET, row)

    return {
        "created": True,
        "log_id": log_id,
        "timestamp": now,
    }

def find_active_appointments_by_phone(phone: str) -> list[dict[str, Any]]:
    rows = _read_range(f"{APPOINTMENTS_SHEET}!A:K")
    normalized_phone = normalize_phone(phone)

    if not rows:
        return []

    headers = rows[0]
    data_rows = rows[1:]

    active_statuses = {"scheduled", "confirmed", "pending"}

    appointments = []

    for index, row in enumerate(data_rows, start=2):
        padded_row = row + [""] * (len(headers) - len(row))
        record = dict(zip(headers, padded_row))
        record["_row_number"] = index

        record_phone = normalize_phone(record.get("phone", ""))
        status = str(record.get("status", "") or "").strip().lower()

        if record_phone == normalized_phone and status in active_statuses:
            appointments.append(record)

    return appointments


def create_appointment(
    client_id: str,
    full_name: str,
    phone: str,
    service: str,
    start_time: str,
    end_time: str,
    status: str = "scheduled",
    calendar_event_id: str = "",
    notes: str = "",
) -> dict[str, Any]:
    now = _now_iso()
    appointment_id = f"appt_{uuid.uuid4().hex[:12]}"

    normalized_phone = normalize_phone(phone)

    row = [
        appointment_id,
        client_id,
        full_name,
        normalized_phone,
        service,
        start_time,
        end_time,
        status,
        calendar_event_id,
        now,
        notes,
    ]

    _append_row(APPOINTMENTS_SHEET, row)

    return {
        "created": True,
        "appointment": {
            "appointment_id": appointment_id,
            "client_id": client_id,
            "full_name": full_name,
            "phone": normalized_phone,
            "service": service,
            "start_time": start_time,
            "end_time": end_time,
            "status": status,
            "calendar_event_id": calendar_event_id,
            "created_at": now,
            "notes": notes,
        },
    }
