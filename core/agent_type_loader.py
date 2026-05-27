from pathlib import Path
import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
AGENT_TYPES_DIR = BASE_DIR / "config" / "agent_types"


def load_agent_type(agent_type_name):
    file_path = AGENT_TYPES_DIR / f"{agent_type_name}.yaml"

    if not file_path.exists():
        raise FileNotFoundError(f"Agent type not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def list_agent_types():
    return [
        file.stem
        for file in AGENT_TYPES_DIR.glob("*.yaml")
    ]