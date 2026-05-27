from datetime import datetime


def create_channel_message(
    channel,
    user_id,
    message_text,
    metadata=None,
):
    return {
        "channel": channel,
        "user_id": user_id,
        "message_text": message_text,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat(),
    }


def create_channel_response(
    response_text,
    channel,
    metadata=None,
):
    return {
        "channel": channel,
        "response_text": response_text,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat(),
    }