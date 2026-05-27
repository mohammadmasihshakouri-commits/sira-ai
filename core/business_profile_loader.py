from pathlib import Path
import yaml


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