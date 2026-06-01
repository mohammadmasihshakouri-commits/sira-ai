from __future__ import annotations

from typing import Any

from core.tools.google_sheets_tools import _execute_with_retry, _get_credentials


DEFAULT_CALENDAR_ID = "primary"
DEFAULT_TIMEZONE = "Asia/Tehran"


def _get_calendar_service():
    from googleapiclient.discovery import build

    creds = _get_credentials()
    return build("calendar", "v3", credentials=creds)


def list_calendar_events(
    start_time: str,
    end_time: str,
    calendar_id: str = DEFAULT_CALENDAR_ID,
    timezone: str = DEFAULT_TIMEZONE,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    service = _get_calendar_service()

    request = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=start_time,
            timeMax=end_time,
            timeZone=timezone,
            singleEvents=True,
            orderBy="startTime",
            maxResults=max_results,
        )
    )

    result = _execute_with_retry(request)

    return result.get("items", [])


def create_calendar_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    calendar_id: str = DEFAULT_CALENDAR_ID,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    service = _get_calendar_service()

    event_body = {
        "summary": summary,
        "description": description,
        "start": {
            "dateTime": start_time,
            "timeZone": timezone,
        },
        "end": {
            "dateTime": end_time,
            "timeZone": timezone,
        },
    }

    request = (
        service.events()
        .insert(
            calendarId=calendar_id,
            body=event_body,
        )
    )

    return _execute_with_retry(request)

