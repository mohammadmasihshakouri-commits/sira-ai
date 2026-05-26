import os
import yaml


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_WORKSPACE_ID = os.getenv("SIRA_WORKSPACE_ID", "cinematicket")

DEFAULT_BUSINESS_DIR = os.path.join(
    BASE_DIR,
    "businesses",
    DEFAULT_WORKSPACE_ID,
)

DEFAULT_AGENT_CONFIG_PATH = os.getenv(
    "SIRA_AGENT_CONFIG_PATH",
    os.path.join(DEFAULT_BUSINESS_DIR, "agent.yaml"),
)

DEFAULT_PLAYBOOK_PATH = os.getenv(
    "SIRA_PLAYBOOK_PATH",
    os.path.join(DEFAULT_BUSINESS_DIR, "playbook.yaml"),
)

LEGACY_PLAYBOOK_PATH = os.path.join(
    BASE_DIR,
    "config",
    "cinematicket_playbook.yaml",
)


_AGENT_CONFIG_CACHE = None


def load_yaml_file(path, default=None):
    if default is None:
        default = {}

    if not path or not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or default


def load_agent_config(force_reload=False):
    global _AGENT_CONFIG_CACHE

    if _AGENT_CONFIG_CACHE is not None and not force_reload:
        return _AGENT_CONFIG_CACHE

    _AGENT_CONFIG_CACHE = load_yaml_file(DEFAULT_AGENT_CONFIG_PATH, default={})
    return _AGENT_CONFIG_CACHE


def get_platform_name():
    config = load_agent_config()
    return config.get("platform", {}).get("name") or "Sira"


def get_workspace_name():
    config = load_agent_config()
    return config.get("workspace", {}).get("name") or "Demo Workspace"


def get_agent_name():
    config = load_agent_config()
    name = config.get("agent", {}).get("name")

    if name:
        return name

    return None


def get_agent_label():
    agent_name = get_agent_name()

    if agent_name:
        return agent_name

    return "Agent not configured"


def get_agent_greeting(language="fa"):
    config = load_agent_config()
    greeting = config.get("agent", {}).get("greeting", {})

    if isinstance(greeting, dict):
        value = greeting.get(language) or greeting.get("fa")
        return value or ""

    if isinstance(greeting, str):
        return greeting

    return ""


def get_playbook_path():
    if os.path.exists(DEFAULT_PLAYBOOK_PATH):
        return DEFAULT_PLAYBOOK_PATH

    return LEGACY_PLAYBOOK_PATH