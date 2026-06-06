"""Auth helpers. Note the deliberate cycle: validate_token calls
refresh_token, which on certain branches calls validate_token again.
"""
import time


def validate_token(token):
    if not token:
        return False
    if _is_expired(token):
        token = refresh_token(token)
    return parse(token) is not None


def refresh_token(token):
    new = parse(token) or "stub"
    # Cycle: refresh_token re-validates after rotation.
    return new if validate_token(new) else None


def parse(token):
    """Lightweight token parser. Note: also defined in utils/format.py
    so find_symbol('parse') should return two definitions."""
    if not token or len(token) < 4:
        return None
    return {"raw": token, "issued": time.time()}


def _is_expired(token):
    return token.startswith("expired_")
