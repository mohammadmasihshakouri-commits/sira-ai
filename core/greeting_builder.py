def build_greeting(business_agent):
    business_name = business_agent.get("business_name", "این مجموعه")

    agent_config = business_agent.get("agent_config", {})
    identity = business_agent.get(
        "agent_identity",
        agent_config.get("agent_identity", {}),
    )
    call_settings = business_agent.get(
        "call_settings",
        agent_config.get("call_settings", {}),
    )

    if not call_settings.get("greeting_enabled", True):
        return None

    agent_name = identity.get("agent_name", "دستیار هوشمند")

    return (
        f"سلام، وقتتون بخیر. "
        f"من {agent_name} هستم، دستیار هوشمند پشتیبانی {business_name}. "
        f"بفرمایید، چطور می‌تونم راهنمایی‌تون کنم؟"
    )