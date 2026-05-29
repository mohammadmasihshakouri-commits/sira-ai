import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from core.channel_adapters import telegram_adapter
from core.sira_runtime import handle_message
from core.business_profile_loader import get_business_agent
from core.session_manager import list_sessions


business_agent = get_business_agent("cinematicket", "support")
enabled_playbooks = business_agent["agent_config"].get("playbooks", [])


messages = [
    "من می‌خوام بلیتم رو کنسل کنم",
    "کد رزرو من ۳۳۵۲۴۷۵۳ هست",
    "بله درسته",
]


for text in messages:
    raw_update = {
        "user_id": 1001,
        "chat_id": 2002,
        "username": "telegram_test_user",
        "text": text,
    }

    channel_message = telegram_adapter(raw_update)

    result = handle_message(
        channel_message=channel_message,
        conversation_state=None,
        enabled_playbooks=enabled_playbooks,
    )

    print("\nUSER:", text)
    print("SIRA:", result["reply"])
    print("STATE:", result["state"])
    print("TOOL:", result["tool_call"])


print("\nSESSIONS:")
print(list_sessions())