from app import process_text
from core.playbook_runtime import run_playbook_turn
from core.conversation_state import update_state
from core.session_manager import (
    get_session,
    save_session,
)
from core.conversation_state import (
    create_initial_state,
    update_state,
)

def handle_message(
    channel_message,
    conversation_state,
    enabled_playbooks,
):
    user_id = channel_message["user_id"]
    user_text = channel_message["message_text"]

    if conversation_state is None:
        conversation_state = get_session(user_id)

    if conversation_state is None:
        conversation_state = create_initial_state()

    playbook_result = run_playbook_turn(
        user_text=user_text,
        conversation_state=conversation_state,
        enabled_playbooks=enabled_playbooks
    )

    tool_call = None

    if playbook_result["handled"]:
        assistant_reply = playbook_result["reply"]
        source = playbook_result["source"]
        conversation_state = playbook_result["state"]
        tool_call = playbook_result.get("tool_call")
    else:
        assistant_reply, source = process_text(user_text)

    conversation_state = update_state(
        conversation_state,
        intent=source
    )

    save_session(user_id, conversation_state)

    return {
        "reply": assistant_reply,
        "source": source,
        "state": conversation_state,
        "tool_call": tool_call,
    }