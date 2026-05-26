import os
import yaml

from core.runtime_config import get_playbook_path


PLAYBOOK_CACHE = None
PLAYBOOK_CACHE_PATH = None


def load_playbook(force_reload=False):
    global PLAYBOOK_CACHE
    global PLAYBOOK_CACHE_PATH

    playbook_path = get_playbook_path()

    if (
        PLAYBOOK_CACHE is not None
        and PLAYBOOK_CACHE_PATH == playbook_path
        and not force_reload
    ):
        return PLAYBOOK_CACHE

    if not os.path.exists(playbook_path):
        PLAYBOOK_CACHE = {}
        PLAYBOOK_CACHE_PATH = playbook_path
        return PLAYBOOK_CACHE

    with open(playbook_path, "r", encoding="utf-8") as file:
        PLAYBOOK_CACHE = yaml.safe_load(file) or {}
        PLAYBOOK_CACHE_PATH = playbook_path
        return PLAYBOOK_CACHE