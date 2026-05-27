from datetime import datetime


def create_call_trace(business_name, agent_key):
    return {
        "business_name": business_name,
        "agent_key": agent_key,
        "started_at": datetime.utcnow().isoformat(),
        "turns": [],
        "tool_calls": [],
        "final_state": None,
    }


def add_turn(trace, customer_message, assistant_reply, source, state):
    trace["turns"].append({
        "customer_message": customer_message,
        "assistant_reply": assistant_reply,
        "source": source,
        "state": state.copy() if isinstance(state, dict) else state,
        "timestamp": datetime.utcnow().isoformat(),
    })

    trace["final_state"] = state.copy() if isinstance(state, dict) else state

    return trace


def add_tool_call(trace, tool_name, input_data, output_data):
    trace["tool_calls"].append({
        "tool_name": tool_name,
        "input": input_data,
        "output": output_data,
        "timestamp": datetime.utcnow().isoformat(),
    })

    return trace