"""Logging helpers."""
import sys


def log_info(msg):
    print(f"[info] {msg}", file=sys.stderr)


def log_error(msg):
    print(f"[err]  {msg}", file=sys.stderr)
