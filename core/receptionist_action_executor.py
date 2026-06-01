from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from core.tools.google_calendar_tools import create_calendar_event
from core.tools.google_sheets_tools import (
    create_appointment,
    create_client,
    find_client_by_phone,
)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _calculate_end_time(start_time: str, duration_minutes: int = 60) -> str:
    start_dt = datetime.fromisoformat(start_time)
    end_dt = start_dt + timedelta(minutes=duration_minutes)
    return end_dt.isoformat()


def execute_receptionist_action(
    plan: dict[str, Any],
    conversation_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conversation_state = conversation_state or {}
    action = plan.get("action", "")

    if action != "create_appointment_from_selected_slot":
        return {
            "executed": False,
            "action": action,
            "reason": "No executable action for this plan.",
        }

    selected_slot = plan.get("selected_slot") or {}
    start_time = _safe_text(selected_slot.get("start_datetime"))

    if not start_time:
        return {
            "executed": False,
            "action": action,
            "reason": "Selected slot has no start_datetime.",
        }

    full_name = _safe_text(conversation_state.get("full_name"))
    phone = _safe_text(conversation_state.get("phone"))
    service = _safe_text(conversation_state.get("pending_service"))

    if not full_name or not phone:
        return {
            "executed": False,
            "action": action,
            "reason": "Missing customer identity.",
        }

    if not service:
        service = _safe_text(plan.get("intent", {}).get("service")) or "وقت مراجعه"

    existing_client = find_client_by_phone(phone)

    if existing_client:
        client = existing_client
        client_created = False
    else:
        client_result = create_client(
            full_name=full_name,
            phone=phone,
            source_channel="voice_receptionist",
        )
        client = client_result["client"]
        client_created = bool(client_result.get("created"))

    end_time = _safe_text(selected_slot.get("end_datetime"))

    if not end_time:
        end_time = _calculate_end_time(start_time)

    calendar_event = create_calendar_event(
        summary=f"{service} - {full_name}",
        start_time=start_time,
        end_time=end_time,
        description=(
            f"Client: {full_name}\n"
            f"Phone: {phone}\n"
            f"Service: {service}\n"
            f"Created by Sira receptionist."
        ),
    )

    calendar_event_id = str(calendar_event.get("id", "") or "")

    appointment_result = create_appointment(
        client_id=client["client_id"],
        full_name=full_name,
        phone=phone,
        service=service,
        start_time=start_time,
        end_time=end_time,
        calendar_event_id=calendar_event_id,
        notes="created by receptionist action executor",
    )

    return {
        "executed": True,
        "action": action,
        "client_created": client_created,
        "client": client,
        "calendar_event": calendar_event,
        "appointment": appointment_result["appointment"],
    }