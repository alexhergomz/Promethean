"""Shared HTTP helper with timeout + retry. stdlib only."""
from __future__ import annotations

import json as _json
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_UA = "Promethean-Research/1.0 (+https://github.com/alexhergomz/Promethean)"
DEFAULT_TIMEOUT = 10.0
DEFAULT_RETRIES = 2


class HttpError(Exception):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url
        self.body = body


def get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    as_json: bool = True,
):
    """GET with retry on transient failures (timeouts, 5xx, 429).

    Returns parsed JSON if as_json=True (default), otherwise raw bytes.
    Raises HttpError on 4xx (non-429), and the last exception after retries.
    """
    if params:
        qs = urllib.parse.urlencode(params, doseq=True)
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{qs}"

    hdrs = {"User-Agent": DEFAULT_UA, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if as_json:
                    return _json.loads(raw.decode("utf-8", errors="replace"))
                return raw
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(0.5 * (2 ** attempt))
                last_exc = HttpError(e.code, url, body)
                continue
            raise HttpError(e.code, url, body) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise
    if last_exc:
        raise last_exc


def post_json(
    url: str,
    payload: dict,
    headers: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
):
    """POST JSON with retry. Returns parsed JSON response."""
    hdrs = {
        "User-Agent": DEFAULT_UA,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if headers:
        hdrs.update(headers)

    body = _json.dumps(payload).encode("utf-8")

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return _json.loads(raw.decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(0.5 * (2 ** attempt))
                last_exc = HttpError(e.code, url, body_text)
                continue
            raise HttpError(e.code, url, body_text) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise
    if last_exc:
        raise last_exc
