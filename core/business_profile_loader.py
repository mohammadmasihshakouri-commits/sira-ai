from pathlib import Path
import yaml

from core.agent_type_loader import load_agent_type


BASE_DIR = Path(__file__).resolve().parents[1]
BUSINESS_PROFILES_DIR = BASE_DIR / "config" / "business_profiles"


def load_business_profile(profile_name):
    file_path = BUSINESS_PROFILES_DIR / f"{profile_name}.yaml"

    if not file_path.exists():
        raise FileNotFoundError(f"Business profile not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def list_business_profiles():
    return [
        file.stem
        for file in BUSINESS_PROFILES_DIR.glob("*.yaml")
    ]


def get_business_agent(profile_name, agent_key):
    profile = load_business_profile(profile_name)

    agents = profile.get("agents", {})

    if agent_key not in agents:
        raise KeyError(f"Agent '{agent_key}' not found in business profile '{profile_name}'")

    agent_config = agents[agent_key]

    if not agent_config.get("enabled", False):
        raise ValueError(f"Agent '{agent_key}' is disabled for business profile '{profile_name}'")

    return {
        "business_name": profile.get("business_name"),
        "industry": profile.get("industry"),
        "language": profile.get("language"),
        "agent_key": agent_key,
        "agent_config": agent_config,
    }


def get_resolved_business_agent(profile_name, agent_key):
    business_agent = get_business_agent(profile_name, agent_key)
    agent_config = business_agent["agent_config"]

    agent_type_name = agent_config.get("agent_type")

    if not agent_type_name:
        raise ValueError(
            f"Agent '{agent_key}' in business profile '{profile_name}' does not define agent_type"
        )

    agent_type_config = load_agent_type(agent_type_name)

    return {
        "business_name": business_agent.get("business_name"),
        "industry": business_agent.get("industry"),
        "language": business_agent.get("language"),
        "agent_key": business_agent.get("agent_key"),
        "agent_type": agent_type_name,
        "purpose": agent_type_config.get("purpose"),
        "default_start": agent_type_config.get("default_start"),
        "identity_required": agent_type_config.get("identity_required", {}),
        "core_flows": agent_type_config.get("core_flows", []),
        "agent_identity": agent_config.get("agent_identity", {}),
        "enabled_tools": agent_config.get("enabled_tools", agent_type_config.get("tools", [])),
        "available_tools": agent_type_config.get("tools", []),
        "enabled_playbooks": agent_config.get("playbooks", []),
        "call_settings": agent_config.get("call_settings", {}),
        "raw_agent_config": agent_config,
        "raw_agent_type_config": agent_type_config,
    }

