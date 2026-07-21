# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Edit recovery for weak models.** When `Edit`'s `old_string` has no verbatim
  match, a unique target is recovered by stripping Read line-number gutters,
  ignoring indentation, then a high-similarity block match. Only unambiguous
  spans are applied, and the diff is annotated. Toggle `fuzzy_edit`.
- **Auto verify-after-edit.** After a successful `Edit`/`Write` on a source
  file, its checker runs (pyright/mypy/flake8/py_compile for Python,
  shellcheck/bash for shell) and problems are appended to the tool result.
  Silent when clean or no checker is installed. Toggle `verify_after_edit`.
- **AGENTS.md** is loaded alongside `CLAUDE.md` (project and `~/.claude/`).
- **Repo-aware `/init`** pre-fills `CLAUDE.md` from a scan of the project
  (languages, manifests, entry points, inferred test command).
- Windows hardware detection for `/model recommend` (RAM via GlobalMemoryStatusEx,
  VRAM via the driver registry / nvidia-smi).
- **Tab file-path completion** in the REPL input, alongside slash-command
  completion. Path-like tokens complete as you type; Tab completes any token.

### Changed
- **The harness is now llama.cpp-only.** Promethean targets a single backend:
  llama.cpp (llama-server) over the OpenAI-compatible protocol (the `custom`
  provider), which also drives any other OpenAI-compatible server (Ollama's
  `/v1`, LM Studio, vLLM). This replaces the previous multi-provider engine and
  removes a large maintenance surface. `/model` and the setup wizard are
  local-first; `/model recommend` is unchanged.

### Removed
- Hosted cloud providers (Anthropic, OpenAI, Gemini, Kimi, Qwen/DashScope,
  Zhipu, DeepSeek, MiniMax) and the native Ollama transport, along with their
  cost tables, prefix routing, optillm integration, and the `anthropic`
  dependency. Point `custom_base_url` at an OpenAI-compatible proxy to use a
  hosted model.

## [3.05.76]

Baseline release from which this changelog is tracked. History prior to this
entry predates the changelog and is available in the Git commit log.

[Unreleased]: https://github.com/alexhergomz/Promethean/compare/v3.05.76...HEAD
[3.05.76]: https://github.com/alexhergomz/Promethean/releases/tag/v3.05.76
