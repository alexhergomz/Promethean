"""Hardware-matched local-model recommendations.

`/model recommend` detects the machine's memory budget, then — instead of
dumping the full ~20-file quantization list from each Hugging Face repo — shows
only the quants that actually fit, with the context window each one leaves, and
a single recommended pick per model.

Curated toward **KV-cache-efficient** families that make good local agents:
  * Qwen3.5 (primary) — GQA, small KV, the tuned flagship family.
  * Gemma 4 (secondary) — interleaved local/global attention, KV-friendly.
  * Nemotron (tertiary) — a couple of small, capable options.

Quant availability + sizes are fetched live from the HF API so the list stays
current; the fit math and ranking are pure functions (unit-tested). Network is
best-effort — a failed fetch degrades to "couldn't reach HF" for that repo, it
never raises.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field

# ── Catalog ─────────────────────────────────────────────────────────────────
# kv_gb_per_16k: rough full-precision KV-cache GB per 16K tokens, anchored to
# the measured Qwen3.5-9B figure (~1.0 GB / 16K) and scaled by architecture.
# With the flagship TurboQuant Q4 KV cache the real cost is ~4× smaller.


@dataclass
class ModelSpec:
    key: str                 # short selector, e.g. "qwen3.5-9b"
    family: str
    repo: str                # HF GGUF repo id
    params_b: float          # total parameters (billions)
    active_b: float          # active params (== params_b for dense; less for MoE)
    kv_gb_per_16k: float
    tier: int                # 1 = primary recommend, 2 = secondary, 3 = niche
    tags: list = field(default_factory=list)
    note: str = ""
    max_ctx_k: int = 0       # native max context (K tokens); 0 → family default

    # Family defaults for native max context. The recommendation sizes the
    # quant so this whole window fits alongside the weights.
    _FAMILY_MAX_CTX = {"Qwen3.5": 256, "Gemma 4": 128, "Nemotron": 128}

    def __post_init__(self):
        if not self.max_ctx_k:
            self.max_ctx_k = self._FAMILY_MAX_CTX.get(self.family, 128)

    @property
    def is_moe(self) -> bool:
        return self.active_b < self.params_b


CATALOG: list[ModelSpec] = [
    # ── Qwen3.5 (primary; KV-efficient GQA) ─────────────────────────────────
    ModelSpec("qwen3.5-0.8b", "Qwen3.5", "unsloth/Qwen3.5-0.8B-GGUF", 0.8, 0.8, 0.12, 1,
              ["tiny"], "Ultra-light; phones / CPU-only."),
    ModelSpec("qwen3.5-2b", "Qwen3.5", "unsloth/Qwen3.5-2B-GGUF", 2.0, 2.0, 0.28, 1,
              [], "Small but coherent for simple agent loops."),
    ModelSpec("qwen3.5-4b", "Qwen3.5", "unsloth/Qwen3.5-4B-GGUF", 4.0, 4.0, 0.5, 1,
              [], "Good balance on ≤8 GB."),
    ModelSpec("qwen3.5-9b", "Qwen3.5", "unsloth/Qwen3.5-9B-GGUF", 9.0, 9.0, 1.0, 1,
              ["flagship"], "The tuned flagship target — best local agent on 8 GB."),
    ModelSpec("qwen3.5-27b", "Qwen3.5", "unsloth/Qwen3.5-27B-GGUF", 27.0, 27.0, 2.4, 1,
              ["large"], "Needs a big GPU or lots of RAM (CPU offload)."),
    ModelSpec("qwen3.5-35b-a3b", "Qwen3.5", "unsloth/Qwen3.5-35B-A3B-GGUF", 35.0, 3.0, 1.6, 2,
              ["moe"], "MoE: only ~3B active (fast), but all experts must fit in memory."),
    # ── Gemma 4 (secondary; KV-friendly) ────────────────────────────────────
    ModelSpec("gemma4-e4b", "Gemma 4", "unsloth/gemma-4-E4B-it-GGUF", 4.0, 4.0, 0.45, 2,
              [], "Efficient 4B-class; strong general chat."),
    ModelSpec("gemma4-12b", "Gemma 4", "unsloth/gemma-4-12b-it-GGUF", 12.0, 12.0, 1.3, 2,
              [], "12B; needs ~10 GB+ at 4-bit."),
    # ── Nemotron (tertiary; a couple of small options) ──────────────────────
    ModelSpec("nemotron-nano-4b", "Nemotron", "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF", 4.0, 4.0, 0.5, 3,
              [], "NVIDIA Nemotron Nano; compact."),
    ModelSpec("nemotron-nano-8b", "Nemotron", "unsloth/Llama-3.1-Nemotron-Nano-8B-v1-GGUF", 8.0, 8.0, 0.95, 3,
              ["tool-calling"], "Llama-3.1 based; reliable native tool calls."),
]

CATALOG_BY_KEY = {m.key: m for m in CATALOG}

# ── Quant quality scoring ───────────────────────────────────────────────────
# Higher score = better fidelity. Extracted from the quant label; a small bonus
# for Unsloth-Dynamic (UD-*) quants, which are higher quality at a given size.
_QUANT_RE = re.compile(r"(UD-)?((?:IQ|Q)\d+(?:_[A-Z0-9]+)*|BF16|F16|F32)", re.IGNORECASE)
_EXCLUDE = ("bf16", "f16", "f32")   # too big / full precision; not agent quants


def parse_quant_label(filename: str) -> str | None:
    """Extract a quant label (e.g. 'Q4_K_M', 'UD-Q4_K_XL', 'IQ4_XS') from a GGUF filename."""
    m = _QUANT_RE.search(filename)
    if not m:
        return None
    ud = m.group(1) or ""
    return f"{ud}{m.group(2)}".upper()


def _quant_bits(label: str) -> int:
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if m else 4


def quant_quality(label: str) -> float:
    """Heuristic fidelity score for ranking quants (higher = better)."""
    lab = label.upper()
    digits = re.search(r"(\d+)", lab)
    bits = int(digits.group(1)) if digits else 4
    score = bits * 10.0
    if "_K_M" in lab or "_K_XL" in lab:
        score += 3
    elif "_K_S" in lab:
        score += 1
    elif "_K" in lab:
        score += 2
    if lab.startswith("IQ"):
        score -= 1          # i-quants slightly below same-bit K-quants for agents
    if lab.startswith("UD-"):
        score += 1.5        # Unsloth dynamic: better quality per byte
    return score


# ── HF quant fetch ──────────────────────────────────────────────────────────
@dataclass
class Quant:
    label: str
    size_gb: float
    filename: str = ""       # representative file (shard 1 for sharded quants)
    shards: int = 1


def _parse_tree(tree: list) -> list[Quant]:
    """Turn an HF /tree/main JSON listing into a deduped list of Quant.

    Groups sharded files by quant label (sums their sizes), skips mmproj
    projectors and full-precision weights.
    """
    agg: dict[str, dict] = {}
    for entry in tree:
        path = entry.get("path", "")
        if not path.endswith(".gguf"):
            continue
        base = path.rsplit("/", 1)[-1]
        if base.lower().startswith("mmproj"):
            continue
        label = parse_quant_label(base)
        if not label or label.lower() in _EXCLUDE:
            continue
        size = entry.get("size") or (entry.get("lfs") or {}).get("size") or 0
        a = agg.setdefault(label, {"gb": 0.0, "file": base, "shards": 0})
        a["gb"] += float(size) / 1e9
        a["shards"] += 1
        # Keep the lexicographically-first path as the representative filename
        # (shard 00001-of-… sorts first).
        if base < a["file"]:
            a["file"] = base
    return [Quant(lbl, round(v["gb"], 2), v["file"], v["shards"])
            for lbl, v in agg.items() if v["gb"] > 0]


def download_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def verify_download(url: str, timeout: float = 15.0) -> tuple[bool, int | None]:
    """Check a download URL is live without downloading the file.

    Issues a 1-byte ranged GET (``Range: bytes=0-0``), following HF's redirect
    to the CDN. Returns (ok, total_size_bytes). ok is True on 206/200; the total
    size is read from the Content-Range (``bytes 0-0/TOTAL``) or Content-Length
    header. Never raises.
    """
    req = urllib.request.Request(
        url, method="GET",
        headers={"Range": "bytes=0-0", "User-Agent": "promethean-model-recommend"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            total = None
            cr = resp.headers.get("Content-Range")
            if cr and "/" in cr:
                tail = cr.rsplit("/", 1)[-1].strip()
                if tail.isdigit():
                    total = int(tail)
            if total is None:
                cl = resp.headers.get("Content-Length")
                if cl and cl.strip().isdigit():
                    total = int(cl)
            return status in (200, 206), total
    except Exception:
        return False, None


def fetch_quants(repo: str, timeout: float = 8.0) -> list[Quant] | None:
    """Fetch available GGUF quants for a repo from the HF API. None on failure."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "promethean-model-recommend"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tree = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(tree, list):
        return None
    quants = _parse_tree(tree)
    quants.sort(key=lambda q: q.size_gb)
    return quants


# ── Hardware detection ──────────────────────────────────────────────────────
@dataclass
class Hardware:
    ram_gb: float | None
    vram_gb: float | None

    @property
    def budget_gb(self) -> float | None:
        """Practical memory budget for a fully GPU-resident model.

        VRAM when a discrete GPU is found; otherwise system RAM (CPU / unified
        memory). None if nothing could be detected.
        """
        if self.vram_gb:
            return self.vram_gb
        return self.ram_gb


def _detect_ram_gb() -> float | None:
    # Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    # macOS / BSD
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip()) / 1e9
    except Exception:
        pass
    # Fallback
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return None


def _detect_vram_gb() -> float | None:
    import glob
    # Linux sysfs (AMD/Intel) — most reliable, no tools needed.
    best = 0
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
        try:
            with open(p) as f:
                best = max(best, int(f.read().strip()))
        except Exception:
            continue
    if best > 0:
        return best / 1e9
    # NVIDIA
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        if out.returncode == 0:
            mb = max(int(x) for x in out.stdout.split() if x.strip().isdigit())
            return mb / 1024
    except Exception:
        pass
    return None


def detect_hardware() -> Hardware:
    return Hardware(ram_gb=_detect_ram_gb(), vram_gb=_detect_vram_gb())


# ── Fit math (KV-cache aware) ───────────────────────────────────────────────
RUNTIME_OVERHEAD_GB = 0.8      # compute/activation buffers
MIN_CTX_K = 8                  # a quant that can't hold 8K ctx isn't viable
DEFAULT_KV_DIV = 4.0           # assume Q4 KV cache (TurboQuant / llama.cpp -ctk q4_0)


def kv_gb(model: ModelSpec, ctx_k: float, kv_quant_div: float) -> float:
    """KV-cache footprint (GB) for ``ctx_k`` thousand tokens at a compression."""
    return model.kv_gb_per_16k * (ctx_k / 16.0) / max(kv_quant_div, 0.01)


def max_context_k(budget_gb: float, model: ModelSpec, quant_gb: float,
                  kv_quant_div: float = 1.0) -> float:
    """Largest context (K tokens) that fits alongside the weights + overhead."""
    free = budget_gb - quant_gb - RUNTIME_OVERHEAD_GB
    if free <= 0:
        return 0.0
    per_16k = model.kv_gb_per_16k / max(kv_quant_div, 0.01)
    if per_16k <= 0:
        return 999.0
    return (free / per_16k) * 16.0


@dataclass
class Pick:
    quant: Quant
    ctx_k: float               # context this quant reaches at the chosen KV div
    fits_target: bool          # does it reach the full target context?


def _pick(budget_gb, model, q, kv_div, target_ctx_k) -> Pick:
    c = max_context_k(budget_gb, model, q.size_gb, kv_div)
    return Pick(q, round(min(c, 9_999)), c >= target_ctx_k)


def recommend_for_model(budget_gb: float, model: ModelSpec, quants: list[Quant],
                        target_ctx_k: float | None = None,
                        kv_quant_div: float = DEFAULT_KV_DIV) -> dict | None:
    """Recommend a quant sized so the model's *full* context fits with the KV cache.

    Targets ``target_ctx_k`` (default: the model's native max context) and
    accounts for the KV-cache footprint at that length under ``kv_quant_div``
    compression. The recommended quant is the highest-fidelity one whose
    weights + full-context KV + overhead still fit the budget — which naturally
    steers toward a lower-precision or smaller quant on tight hardware. If no
    quant reaches the full target, it recommends the one that reaches the *most*
    context, and always offers the fidelity trade-off as an alternative.
    """
    target = target_ctx_k if target_ctx_k else float(model.max_ctx_k)
    # Every quant that can at least hold MIN_CTX_K at this compression.
    scored = [(q, max_context_k(budget_gb, model, q.size_gb, kv_quant_div))
              for q in quants]
    scored = [(q, c) for (q, c) in scored if c >= MIN_CTX_K]
    if not scored:
        return None

    fits = [(q, c) for (q, c) in scored if c >= target]
    if fits:
        # Best fidelity that still covers the full target context.
        q, _ = max(fits, key=lambda t: quant_quality(t[0].label))
    else:
        # Can't cover the full window at any quant → maximize reachable context
        # (i.e. the lowest-precision / smallest quant), honoring the KV budget.
        q, _ = max(scored, key=lambda t: (t[1], quant_quality(t[0].label)))
    recommended = _pick(budget_gb, model, q, kv_quant_div, target)

    # Alternatives, framed as the fidelity↔context trade-off.
    rq = quant_quality(q.label)
    higher = [(qq, c) for (qq, c) in scored if quant_quality(qq.label) > rq]
    lower = [(qq, c) for (qq, c) in scored if quant_quality(qq.label) < rq]
    alt_quality = None
    if higher:  # the best fidelity available, and the context it costs
        qq, _ = max(higher, key=lambda t: quant_quality(t[0].label))
        alt_quality = _pick(budget_gb, model, qq, kv_quant_div, target)
    alt_context = None
    if not recommended.fits_target and lower:  # squeeze even more context
        qq, _ = max(lower, key=lambda t: t[1])
        alt_context = _pick(budget_gb, model, qq, kv_quant_div, target)

    return {"model": model, "recommended": recommended,
            "target_ctx_k": round(target), "kv_div": kv_quant_div,
            "alt_quality": alt_quality, "alt_context": alt_context}


def _min_fit_gb(model: ModelSpec) -> float:
    """Rough smallest-quant footprint (~3-bit) for pre-filtering candidates."""
    return model.params_b * 0.42 + RUNTIME_OVERHEAD_GB


def candidate_models(budget_gb: float, families: list[str] | None = None) -> list[ModelSpec]:
    """Catalog entries that could plausibly fit the budget, best-first.

    Pre-filters by a coarse footprint estimate so we don't fetch repos that
    obviously can't fit (e.g. 27B on 8 GB). Ordered tier, then larger-first.
    """
    cands = [m for m in CATALOG if _min_fit_gb(m) <= budget_gb * 1.05]
    if families:
        fam = {f.lower() for f in families}
        cands = [m for m in cands if m.family.lower() in fam]
    cands.sort(key=lambda m: (m.tier, -m.params_b))
    return cands


def build_recommendations(budget_gb: float, families: list[str] | None = None,
                          max_models: int = 6, target_ctx_k: float | None = None,
                          kv_quant_div: float = DEFAULT_KV_DIV) -> list[dict]:
    """Detect candidates, fetch their quants concurrently, and narrow each.

    Each model is sized so its full context (``target_ctx_k`` or the model's
    native max) fits alongside the weights at ``kv_quant_div`` KV compression.
    Returns recommend_for_model() dicts ranked best-first; a ``fetch_failed``
    key flags repos that couldn't be reached.
    """
    import concurrent.futures as cf

    cands = candidate_models(budget_gb, families)[:max_models]
    if not cands:
        return []

    results: dict[str, list[Quant] | None] = {}
    with cf.ThreadPoolExecutor(max_workers=min(len(cands), 8)) as ex:
        futs = {ex.submit(fetch_quants, m.repo): m for m in cands}
        for fut in cf.as_completed(futs):
            m = futs[fut]
            try:
                results[m.key] = fut.result()
            except Exception:
                results[m.key] = None

    out = []
    for m in cands:
        quants = results.get(m.key)
        if quants is None:
            out.append({"model": m, "fetch_failed": True})
            continue
        rec = recommend_for_model(budget_gb, m, quants,
                                  target_ctx_k=target_ctx_k, kv_quant_div=kv_quant_div)
        if rec:
            out.append(rec)
    # Rank: real recommendations first (tier, then larger-params), failures last.
    out.sort(key=lambda r: (0 if "recommended" in r else 1,
                            r["model"].tier, -r["model"].params_b))
    return out
