_sessions = {}


def get_session(user_id):
    return _sessions.get(user_id)


def save_session(user_id, state):
    _sessions[user_id] = state


def clear_session(user_id):
    if user_id in _sessions:
        del _sessions[user_id]


def list_sessions():
    return _sessions