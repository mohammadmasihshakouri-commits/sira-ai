CALL_STARTED = "call_started"
WAITING_FOR_USER_REQUEST = "waiting_for_user_request"
WAITING_FOR_RESERVATION_CODE = "waiting_for_reservation_code"
WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
RESOLVED = "resolved"
HANDOFF_NEEDED = "handoff_needed"


def create_initial_state():
    return {
        "state": CALL_STARTED,
        "last_intent": None,
        "pending_value": None,
        "pending_value_type": None,
        "turn_count": 0,
    }


def update_state(current_state, intent=None, pending_value=None, pending_value_type=None):
    if current_state is None:
        current_state = create_initial_state()

    current_state["turn_count"] = current_state.get("turn_count", 0) + 1

    if intent:
        current_state["last_intent"] = intent

    if pending_value is not None:
        current_state["pending_value"] = pending_value

    if pending_value_type is not None:
        current_state["pending_value_type"] = pending_value_type

    return current_state


def set_state(current_state, new_state):
    if current_state is None:
        current_state = create_initial_state()

    current_state["state"] = new_state
    return current_state