from core.channel_message import create_channel_message


def telegram_adapter(raw_update):
    return create_channel_message(
        channel="telegram",
        user_id=str(raw_update.get("user_id")),
        message_text=raw_update.get("text", ""),
        metadata={
            "chat_id": raw_update.get("chat_id"),
            "username": raw_update.get("username"),
        },
    )


def website_chat_adapter(raw_message):
    return create_channel_message(
        channel="website",
        user_id=str(raw_message.get("session_id")),
        message_text=raw_message.get("text", ""),
        metadata={
            "page_url": raw_message.get("page_url"),
            "browser": raw_message.get("browser"),
        },
    )