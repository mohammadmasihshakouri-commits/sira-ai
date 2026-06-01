from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from core.tools.google_calendar_tools import list_calendar_events


DEFAULT_TIMEZONE = "Asia/Tehran"
DEFAULT_BUSINESS_START_HOUR = 10
DEFAULT_BUSINESS_END_HOUR = 21
DEFAULT_SLOT_MINUTES = 60


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(str(value).strip())


def _event_start_end(event: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    start_data = event.get("start", {}) or {}
    end_data = event.get("end", {}) or {}

    start_value = start_data.get("dateTime") or start_data.get("date")
    end_value = end_data.get("dateTime") or end_data.get("date")

    if not start_value or not end_value:
        return None, None

    try:
        return _parse_datetime(start_value), _parse_datetime(end_value)
    except ValueError:
        return None, None


def _events_overlap(
    slot_start: datetime,
    slot_end: datetime,
    event_start: datetime,
    event_end: datetime,
) -> bool:
    return slot_start < event_end and event_start < slot_end


def _format_slot_display(slot_start: datetime) -> str:
    hour = slot_start.hour

    if hour == 0:
        return "۱۲ شب"

    if hour < 12:
        return f"{hour} صبح"

    if hour == 12:
        return "۱۲ ظهر"

    return f"{hour - 12} عصر"


def check_requested_slot_availability(
    start_time: str,
    end_time: str,
) -> dict[str, Any]:
    slot_start = _parse_datetime(start_time)
    slot_end = _parse_datetime(end_time)

    events = list_calendar_events(
        start_time=start_time,
        end_time=end_time,
    )

    blocking_events = []

    for event in events:
        if event.get("status") == "cancelled":
            continue

        event_start, event_end = _event_start_end(event)

        if not event_start or not event_end:
            continue

        if _events_overlap(slot_start, slot_end, event_start, event_end):
            blocking_events.append(event)

    return {
        "is_available": len(blocking_events) == 0,
        "blocking_events": blocking_events,
        "start_time": start_time,
        "end_time": end_time,
    }


def find_available_slots_for_day(
    date_text: str,
    business_start_hour: int = DEFAULT_BUSINESS_START_HOUR,
    business_end_hour: int = DEFAULT_BUSINESS_END_HOUR,
    slot_minutes: int = DEFAULT_SLOT_MINUTES,
    max_slots: int = 6,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    tz = ZoneInfo(timezone)
    day = datetime.fromisoformat(date_text).date()

    day_start = datetime.combine(
        day,
        time(hour=business_start_hour, minute=0),
        tzinfo=tz,
    )
    day_end = datetime.combine(
        day,
        time(hour=business_end_hour, minute=0),
        tzinfo=tz,
    )

    events = list_calendar_events(
        start_time=day_start.isoformat(),
        end_time=day_end.isoformat(),
        timezone=timezone,
        max_results=50,
    )

    active_events = []

    for event in events:
        if event.get("status") == "cancelled":
            continue

        event_start, event_end = _event_start_end(event)

        if event_start and event_end:
            active_events.append(
                {
                    "event": event,
                    "start": event_start,
                    "end": event_end,
                }
            )

    available_slots = []
    current_start = day_start
    index = 1

    while current_start + timedelta(minutes=slot_minutes) <= day_end:
        current_end = current_start + timedelta(minutes=slot_minutes)

        is_busy = any(
            _events_overlap(
                current_start,
                current_end,
                item["start"],
                item["end"],
            )
            for item in active_events
        )

        if not is_busy:
            available_slots.append(
                {
                    "index": index,
                    "start_datetime": current_start.isoformat(),
                    "end_datetime": current_end.isoformat(),
                    "display": _format_slot_display(current_start),
                }
            )
            index += 1

            if len(available_slots) >= max_slots:
                break

        current_start = current_end

    return {
        "date": date_text,
        "timezone": timezone,
        "business_start_hour": business_start_hour,
        "business_end_hour": business_end_hour,
        "slot_minutes": slot_minutes,
        "available_slots": available_slots,
        "busy_events_count": len(active_events),
    }
