"""Recover tool calls that a model emitted as text instead of native ``tool_calls``.

Many local models (Qwen2.5-Coder and friends, served via llama.cpp or Ollama)
do not populate the OpenAI ``message.tool_calls`` array. Instead they write the
call into the assistant *content* as JSON or XML — a fenced ```json block, a
Qwen ``<tool_call>`` wrapper, an ``<function …>`` tag, or a bare top-level JSON
object. The harness dispatches only native ``tool_calls``, so without this
module those turns silently no-op (the run exits 0 having done nothing).

``recover_tool_calls`` scans the final assistant text for tool-shaped blocks and
returns any whose ``name`` matches a *registered* tool. Matching against the
live tool set is the safety gate: it recovers real calls in the wrong wire
format while ignoring prose that merely looks JSON-ish, and it surfaces
tool-shaped blocks that name an *unknown* tool separately so the caller can warn
instead of silently dropping them (the "hallucinated tool" case).

The parser is deliberately provider-agnostic and pure — it takes text plus the
set of valid names and returns data. All the observed wire formats are covered
by unit tests in tests/test_tool_call_recovery.py.
"""

from __future__ import annotations

import json
import re

# Keys a model might use for the arguments payload, in priority order.
_ARG_KEYS = ("arguments", "parameters", "input", "args", "params")
# Keys a model might use for the tool name.
_NAME_KEYS = ("name", "tool", "tool_name", "function", "recipient_name")


def _coerce_input(raw) -> dict:
    """Normalize an arguments payload to a dict. Strings are JSON-parsed."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except (json.JSONDecodeError, ValueError):
            return {"_raw": raw}
    if raw is None:
        return {}
    return {"value": raw}


def _candidate_from_obj(obj) -> tuple[str, dict] | None:
    """Return (name, input) if a decoded JSON object looks like a tool call."""
    if not isinstance(obj, dict):
        return None
    name = None
    for k in _NAME_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            name = v.strip()
            break
    if not name:
        return None
    args = None
    for k in _ARG_KEYS:
        if k in obj:
            args = obj[k]
            break
    # Some models put the args inline as sibling keys (everything that isn't
    # the name/arg wrapper). Fall back to that only when there's no explicit
    # arguments key, so a normal {"name","arguments"} shape isn't polluted.
    if args is None:
        inline = {k: v for k, v in obj.items()
                  if k not in _NAME_KEYS and k not in _ARG_KEYS}
        args = inline if inline else {}
    return name, _coerce_input(args)


def _iter_json_objects(text: str):
    """Yield decoded top-level JSON objects embedded anywhere in ``text``.

    Brace-matched scan that respects string literals and escapes, so JSON that
    is surrounded by prose (or concatenated) is still found. Non-object values
    are skipped.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, in_str, esc, j = 0, False, False, i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if depth == 0 and j < n:
            chunk = text[i:j + 1]
            try:
                yield json.loads(chunk)
            except (json.JSONDecodeError, ValueError):
                pass
            i = j + 1
        else:
            # Unbalanced from here on — nothing more to find.
            break


# <function=NAME>{json}</function>  and  <function name="NAME"> … </function>
_FUNC_TAG_RE = re.compile(
    r"<function(?:\s*=\s*|\s+name\s*=\s*)[\"']?([A-Za-z_][\w\-]*)[\"']?\s*"
    r"(/?)>(.*?)(?:</function>|$)",
    re.DOTALL,
)
# <parameter name="x">value</parameter>  (Qwen/Hermes XML arg style)
_PARAM_RE = re.compile(
    r"<parameter\s+name\s*=\s*[\"']?([\w\-]+)[\"']?\s*>(.*?)</parameter>",
    re.DOTALL,
)


def _candidates_from_function_tags(text: str):
    """Yield (name, input) for <function …> XML tags."""
    for m in _FUNC_TAG_RE.finditer(text):
        name, body = m.group(1), m.group(3).strip()
        params = dict(_PARAM_RE.findall(body))
        if params:
            yield name, {k: v.strip() for k, v in params.items()}
            continue
        # Body (if any) is usually a JSON args object for this tag's name.
        obj = next(_iter_json_objects(body), None) if body else None
        yield name, _coerce_input(obj) if obj is not None else {}


# ```json … ```  /  ```xml … ```  /  ``` … ```  fenced blocks.
_FENCE_RE = re.compile(r"```[ \t]*[\w+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
# <tool_call> … </tool_call>  (Qwen standard wrapper; may be unterminated).
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


def _raw_candidates(text: str):
    """Yield every (name, input, span_text) tool-shaped candidate in ``text``.

    span_text is the substring to strip from the visible content when the
    candidate is accepted.
    """
    seen_spans: list[str] = []

    def _mark(s):
        seen_spans.append(s)

    # 1. <tool_call> wrappers.
    for m in _TOOLCALL_RE.finditer(text):
        inner = m.group(1)
        for obj in _iter_json_objects(inner):
            cand = _candidate_from_obj(obj)
            if cand:
                yield cand[0], cand[1], m.group(0)
                _mark(m.group(0))
                break

    # 2. Fenced code blocks (```json / ```xml / ```).
    for m in _FENCE_RE.finditer(text):
        inner = m.group(1)
        got = False
        for name, inp in _candidates_from_function_tags(inner):
            yield name, inp, m.group(0)
            got = True
        if got:
            continue
        for obj in _iter_json_objects(inner):
            cand = _candidate_from_obj(obj)
            if cand:
                yield cand[0], cand[1], m.group(0)
                got = True
        if got:
            continue

    # 3. Bare <function …> tags outside fences.
    for m in _FUNC_TAG_RE.finditer(text):
        for name, inp in _candidates_from_function_tags(m.group(0)):
            yield name, inp, m.group(0)

    # 4. Bare top-level JSON objects (the Ollama / plain-JSON case).
    for obj in _iter_json_objects(text):
        cand = _candidate_from_obj(obj)
        if cand:
            yield cand[0], cand[1], json.dumps(obj)


def recover_tool_calls(text: str, valid_names) -> tuple[list[dict], str, list[str]]:
    """Extract tool calls a model wrote as text instead of native tool_calls.

    Args:
        text:        the final assistant content.
        valid_names: iterable of registered tool names (the dispatch gate).

    Returns (recovered, cleaned_text, unknown):
        recovered    — list of {"id","name","input"} for tool-shaped blocks
                       whose name is a registered tool, deduped, in order.
        cleaned_text — ``text`` with the recovered blocks removed (so history
                       and any re-render don't carry the raw JSON/XML).
        unknown      — names of tool-shaped blocks that did NOT match a
                       registered tool (likely hallucinated); for warnings.
    """
    if not text or "{" not in text and "<" not in text:
        return [], text, []

    valid = set(valid_names or ())
    recovered: list[dict] = []
    unknown: list[str] = []
    strip_spans: list[str] = []
    seen: set[tuple] = set()

    for name, inp, span in _raw_candidates(text):
        if name in valid:
            key = (name, json.dumps(inp, sort_keys=True, default=str))
            if key in seen:
                continue
            seen.add(key)
            recovered.append({
                "id": f"recovered_{len(recovered)}",
                "name": name,
                "input": inp,
            })
            if span:
                strip_spans.append(span)
        else:
            if name not in unknown:
                unknown.append(name)

    cleaned = text
    for span in strip_spans:
        cleaned = cleaned.replace(span, "", 1)
    cleaned = cleaned.strip()

    return recovered, cleaned, unknown
