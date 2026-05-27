import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from core.runtime_config import (
    DEFAULT_AGENT_CONFIG_PATH,
    DEFAULT_PLAYBOOK_PATH,
    get_agent_label,
    get_platform_name,
    get_playbook_path,
    get_workspace_name,
)


def main():
    print("Sira runtime config check")
    print("-" * 32)
    print(f"Platform: {get_platform_name()}")
    print(f"Workspace: {get_workspace_name()}")
    print(f"Agent: {get_agent_label()}")
    print(f"Agent config path: {DEFAULT_AGENT_CONFIG_PATH}")
    print(f"Default playbook path: {DEFAULT_PLAYBOOK_PATH}")
    print(f"Active playbook path: {get_playbook_path()}")


if __name__ == "__main__":
    main()