"""Tests for tool_call_recovery — the parse-from-text tool-call fallback.

Each case mirrors a wire format observed in real local-model runs (see the
macOS review §8.1 matrix): fenced JSON, fenced XML <function>, Qwen
<tool_call>, and bare top-level JSON.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_call_recovery import recover_tool_calls  # noqa: E402

VALID = {"Write", "Edit", "Read", "Bash", "MemorySearch", "AskUserQuestion"}


def test_fenced_json_bare_object():
    # Qwen2.5-Coder-3B via llama.cpp: fenced ```json {"name","arguments"}
    text = 'Sure, I will create it.\n```json\n{"name": "Write", "arguments": {"file_path": "hello.py", "content": "print(1)"}}\n```'
    calls, cleaned, unknown = recover_tool_calls(text, VALID)
    assert len(calls) == 1
    assert calls[0]["name"] == "Write"
    assert calls[0]["input"]["file_path"] == "hello.py"
    assert "```" not in cleaned
    assert unknown == []


def test_fenced_xml_function_selfish():
    # Qwen2.5-Coder-7B via llama.cpp: fenced ```xml <function name="Write" …>
    text = '```xml\n<function name="Write">\n<parameter name="file_path">a.txt</parameter>\n<parameter name="content">hi</parameter>\n</function>\n```'
    calls, cleaned, unknown = recover_tool_calls(text, VALID)
    assert len(calls) == 1
    assert calls[0]["name"] == "Write"
    assert calls[0]["input"] == {"file_path": "a.txt", "content": "hi"}


def test_function_equals_syntax():
    # <function=Write>{json}</function>
    text = '<function=Write>{"file_path": "b.txt", "content": "x"}</function>'
    calls, _, _ = recover_tool_calls(text, VALID)
    assert len(calls) == 1
    assert calls[0]["name"] == "Write"
    assert calls[0]["input"]["file_path"] == "b.txt"


def test_qwen_tool_call_wrapper():
    # Qwen standard: <tool_call>{"name","arguments"}</tool_call>
    text = '<tool_call>\n{"name": "Bash", "arguments": {"command": "ls"}}\n</tool_call>'
    calls, cleaned, _ = recover_tool_calls(text, VALID)
    assert len(calls) == 1
    assert calls[0]["name"] == "Bash"
    assert calls[0]["input"]["command"] == "ls"
    assert "<tool_call>" not in cleaned


def test_bare_top_level_json():
    # Ollama 7B: bare JSON object as the whole content.
    text = '{"name": "MemorySearch", "arguments": {"query": "model information"}}'
    calls, cleaned, _ = recover_tool_calls(text, VALID)
    assert len(calls) == 1
    assert calls[0]["name"] == "MemorySearch"
    assert calls[0]["input"]["query"] == "model information"


def test_bare_json_with_string_arguments():
    # arguments delivered as a JSON *string* (some templates double-encode).
    text = '{"name": "Bash", "arguments": "{\\"command\\": \\"pwd\\"}"}'
    calls, _, _ = recover_tool_calls(text, VALID)
    assert len(calls) == 1
    assert calls[0]["input"]["command"] == "pwd"


def test_unregistered_tool_is_flagged_not_dispatched():
    # Hallucinated tool name → not recovered, reported in `unknown`.
    text = '{"name": "Note", "arguments": {"text": "remember this"}}'
    calls, _, unknown = recover_tool_calls(text, VALID)
    assert calls == []
    assert "Note" in unknown


def test_plain_prose_untouched():
    text = "Here is how a dict works: {\"key\": \"value\"} is a mapping."
    calls, cleaned, unknown = recover_tool_calls(text, VALID)
    assert calls == []
    assert cleaned == text
    assert unknown == []


def test_native_tool_call_json_in_prose_not_falsely_matched():
    # A JSON object that isn't tool-shaped (no name key) must not match.
    text = 'The config is {"file_path": "x", "content": "y"} for reference.'
    calls, _, _ = recover_tool_calls(text, VALID)
    assert calls == []


def test_multiple_calls_recovered_and_deduped():
    text = (
        '```json\n{"name": "Read", "arguments": {"file_path": "a"}}\n```\n'
        '```json\n{"name": "Read", "arguments": {"file_path": "b"}}\n```\n'
        '```json\n{"name": "Read", "arguments": {"file_path": "a"}}\n```'
    )
    calls, _, _ = recover_tool_calls(text, VALID)
    names_inputs = [(c["name"], c["input"]["file_path"]) for c in calls]
    assert ("Read", "a") in names_inputs
    assert ("Read", "b") in names_inputs
    # Duplicate (Read, a) collapsed.
    assert len(calls) == 2


def test_ids_are_unique():
    text = (
        '{"name": "Read", "arguments": {"file_path": "a"}}\n'
        '{"name": "Read", "arguments": {"file_path": "b"}}'
    )
    calls, _, _ = recover_tool_calls(text, VALID)
    ids = [c["id"] for c in calls]
    assert len(ids) == len(set(ids))


def test_inline_args_without_wrapper():
    # {"name": "Write", "file_path": "...", "content": "..."} — no arguments key.
    text = '{"name": "Write", "file_path": "c.txt", "content": "hi"}'
    calls, _, _ = recover_tool_calls(text, VALID)
    assert len(calls) == 1
    assert calls[0]["input"] == {"file_path": "c.txt", "content": "hi"}


def test_empty_and_none_safe():
    assert recover_tool_calls("", VALID) == ([], "", [])
    assert recover_tool_calls("just text", VALID) == ([], "just text", [])
    assert recover_tool_calls("no braces here", set()) == ([], "no braces here", [])


def test_cleaned_text_preserves_surrounding_prose():
    text = 'Let me write that file.\n```json\n{"name": "Write", "arguments": {"file_path": "z"}}\n```\nDone.'
    calls, cleaned, _ = recover_tool_calls(text, VALID)
    assert len(calls) == 1
    assert "Let me write that file." in cleaned
    assert "Done." in cleaned
    assert "Write" not in cleaned or "```" not in cleaned


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
