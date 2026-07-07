# Choosing and installing a model

Promethean is model-agnostic — it speaks OpenAI-compatible HTTP to whatever
serves the model. This page is the practical "what do I download and how do I
point Promethean at it" guide, separate from the tuned flagship stack.

> **Shortcut:** run **`/model recommend`** in the REPL. It detects your
> RAM/VRAM, pulls the live quantization lists from Hugging Face, and shows only
> the quants that actually fit — with the context each leaves and a copy-paste
> download + wiring command. Pass a budget to override detection
> (`/model recommend 12`) or a family (`/model recommend gemma`). The rest of
> this page is the manual version.

- **Flagship / tuned target:** a ~9B model on the bundled TurboQuant + TriAttention
  `llama.cpp` forks, for 224K context on an 8 GB AMD GPU. That's what the numbers
  in the README are measured against.
- **Everything else (this page):** any GGUF on a stock `llama-server`, or any
  Ollama / LM Studio / cloud model. No custom build required.

If you already run Ollama or LM Studio, you don't need to download anything
separately — skip to [Point Promethean at it](#point-promethean-at-it).

---

## Quick pick

| Hardware | Suggested starting model | Why |
|---|---|---|
| 8–16 GB RAM/VRAM, want coding | **Qwen2.5-Coder-7B-Instruct** (Q4_K_M, ~4.7 GB) | Strong open coding model that fits comfortably |
| Tight on memory (≤8 GB) | **Qwen2.5-Coder-3B-Instruct** (Q4_K_M, ~1.9 GB) | Snappy; weaker reasoning |
| Agentic tool use is the priority | **Qwen2.5-7B-Instruct** or **Llama-3.1-8B-Instruct** | General-instruct models tend to emit native tool calls more reliably than the *coder* variants (see [Tool calling](#tool-calling)) |

Quantization rule of thumb: **Q4_K_M** is the usual sweet spot (quality vs.
size). Go higher (Q5_K_M / Q6_K) if you have the memory and want more accuracy;
lower (Q3) only when you must.

---

## Download a GGUF (stock llama.cpp path)

GGUF files live on Hugging Face. `bartowski` and the model authors publish
quantized repos; pick the `Q4_K_M` file. Convention: keep them in
`~/.promethean/models/`.

```bash
mkdir -p ~/.promethean/models && cd ~/.promethean/models

# Example: Qwen2.5-Coder-7B-Instruct, Q4_K_M
curl -L -o qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
```

Model repo pages (browse for other sizes/quants):

- Qwen2.5-Coder: <https://huggingface.co/Qwen> · GGUFs at
  <https://huggingface.co/bartowski?search=Qwen2.5-Coder>
- Qwen2.5-Instruct (general): <https://huggingface.co/bartowski?search=Qwen2.5-7B-Instruct>
- Llama-3.1-8B-Instruct: <https://huggingface.co/bartowski?search=Llama-3.1-8B-Instruct>

You also need a `llama-server` binary. Either build the flagship fork (see the
README's inference section) or grab a stock prebuilt release from
<https://github.com/ggml-org/llama.cpp/releases> (pick your platform's asset;
macOS/Apple-Silicon uses the `macos-arm64` build, which serves GGUF on Metal).

---

## Point Promethean at it

### A. Let Promethean auto-start `llama-server` (custom provider)

Set these in `~/.promethean/config.json` (or via `/config key=value` in the REPL):

```json
{
  "model": "custom/qwen2.5-coder-7b-instruct",
  "custom_base_url": "http://127.0.0.1:8080/v1",
  "custom_api_key": "sk-local-no-key-needed",
  "llama_model_path": "/home/you/.promethean/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
  "llama_autostart": true
}
```

When the base URL is loopback, Promethean starts `llama-server` for you if it's
down (`server_autostart.py`). The auto-start args include `--jinja` by default
(needed for native tool calls). Override them with
`/config llama_server_args=…` if you run a multi-slot or custom setup.

### B. Point at a server you run yourself

Start it manually and just set the URL — same `custom` provider, `llama_autostart: false`:

```bash
llama-server -m ~/.promethean/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  -ngl 99 -fa on --jinja -c 32768
```

`--jinja` applies the model's chat template so it can emit tool calls; `-ngl 99`
offloads layers to the GPU; `-c` sets the context window.

### C. Ollama / LM Studio / cloud

No download step here — point at the running server:

```bash
promethean -m ollama/qwen2.5-coder:7b        # Ollama
promethean -m lmstudio/qwen2.5-coder-7b       # LM Studio
promethean -m claude-sonnet-4-5               # cloud (needs an API key)
```

Switch any time with `/model <name>`. Named profiles (`/model qwen`) bundle
model + provider + base URL into one alias.

---

## Tool calling

An agent needs the model to *call tools*. Two things make this robust with
local models — both are handled for you, but worth understanding:

1. **Native tool calls** need the chat template applied — pass `--jinja` to
   `llama-server` (on by default in auto-start). Without it, `llama.cpp` can't
   emit the structured call at all.
2. **Text-format calls** — many models (the Qwen-**Coder** variants especially)
   write the call as JSON or XML in the message *text* instead of the native
   field. Promethean recovers those automatically
   (`providers._recover_text_tool_calls`) and dispatches them, and warns if a
   model calls a tool that doesn't exist. Toggle with
   `/config recover_text_tool_calls=false`.

If tool use still misbehaves, a **general-instruct** model
(`Qwen2.5-7B-Instruct`, `Llama-3.1-8B-Instruct`) is usually more reliable at
native tool calls than a *coder* model of the same size.

---

## Context window

Promethean compacts the conversation as it approaches the model's context
limit. For a local `custom` server it auto-detects the served window from
`llama-server`'s `/props` endpoint at startup (`sync_context_limit`), so
compaction triggers at the right point even when the model runs at, say,
`n_ctx=32768` rather than the 128k provider default. If you run a **multi-slot**
server (`-np > 1`), each slot only gets `n_ctx / n_parallel` tokens — set
`/config context_limit=<per-slot tokens>` explicitly; that value always wins
over auto-detection.
