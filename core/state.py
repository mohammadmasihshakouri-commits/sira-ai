def last_assistant_message(history):
    for item in reversed(history):
        if item.get("role") == "Assistant":
            return item.get("content", "")
    return ""


def get_pending_state(history):
    for item in reversed(history):
        if item.get("role") == "SystemState" and item.get("key") == "pending_state":
            return item.get("content", "")
    return ""


def set_pending_state(history, state):
    history.append({
        "role": "SystemState",
        "key": "pending_state",
        "content": state
    })


def clear_pending_states(history):
    return [
        item for item in history
        if not (item.get("role") == "SystemState" and item.get("key") == "pending_state")
    ]


def get_state_value(history, key, default=""):
    for item in reversed(history):
        if item.get("role") == "SystemState" and item.get("key") == key:
            return item.get("content", default)
    return default


def set_state_value(history, key, value):
    history[:] = [
        item for item in history
        if not (item.get("role") == "SystemState" and item.get("key") == key)
    ]

    history.append({
        "role": "SystemState",
        "key": key,
        "content": str(value)
    })


def clear_state_value(history, key):
    history[:] = [
        item for item in history
        if not (item.get("role") == "SystemState" and item.get("key") == key)
    ]

def get_audio_failure_count(history):
    count = 0

    for item in reversed(history):
        if item.get("role") == "SystemState" and item.get("key") == "audio_failure_count":
            try:
                count = int(item.get("content", 0))
            except ValueError:
                count = 0
            break

    return count


def set_audio_failure_count(history, count):
    set_state_value(history, "audio_failure_count", count)


def reset_audio_failure_count(history):
    clear_state_value(history, "audio_failure_count")