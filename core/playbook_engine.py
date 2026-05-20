import os
import yaml


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYBOOK_PATH = os.path.join(BASE_DIR, "config", "cinematicket_playbook.yaml")
PLAYBOOK_CACHE = None


def load_playbook():
    global PLAYBOOK_CACHE

    if PLAYBOOK_CACHE is not None:
        return PLAYBOOK_CACHE

    if not os.path.exists(PLAYBOOK_PATH):
        PLAYBOOK_CACHE = {}
        return PLAYBOOK_CACHE

    with open(PLAYBOOK_PATH, "r", encoding="utf-8") as file:
        PLAYBOOK_CACHE = yaml.safe_load(file) or {}

    return PLAYBOOK_CACHE