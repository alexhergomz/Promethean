"""Tests for the hardware-matched model recommender (pure functions, no network)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_recommend as M  # noqa: E402

# A realistic slice of the unsloth/Qwen3.5-9B-GGUF /tree/main listing.
FAKE_TREE = [
    {"path": "Qwen3.5-9B-Q4_K_M.gguf", "size": 5680522464},
    {"path": "Qwen3.5-9B-Q5_K_M.gguf", "size": 6577841376},
    {"path": "Qwen3.5-9B-Q8_0.gguf", "size": 9527502048},
    {"path": "Qwen3.5-9B-IQ4_XS.gguf", "size": 5168653536},
    {"path": "Qwen3.5-9B-UD-Q4_K_XL.gguf", "size": 5966095584},
    {"path": "Qwen3.5-9B-UD-IQ2_XXS.gguf", "size": 3190613216},
    {"path": "Qwen3.5-9B-BF16.gguf", "size": 17920697312},   # excluded
    {"path": "mmproj-F16.gguf", "size": 918166080},          # excluded
    {"path": "README.md", "size": 1000},                     # excluded
]


def test_parse_quant_label():
    assert M.parse_quant_label("Qwen3.5-9B-Q4_K_M.gguf") == "Q4_K_M"
    assert M.parse_quant_label("Qwen3.5-9B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert M.parse_quant_label("Qwen3.5-9B-IQ4_XS.gguf") == "IQ4_XS"
    assert M.parse_quant_label("random.txt") is None


def test_parse_tree_excludes_and_dedupes():
    quants = M._parse_tree(FAKE_TREE)
    labels = {q.label for q in quants}
    assert "BF16" not in labels
    assert not any(q.label.startswith("MMPROJ") for q in quants)
    assert "Q4_K_M" in labels and "UD-Q4_K_XL" in labels
    q4 = next(q for q in quants if q.label == "Q4_K_M")
    assert abs(q4.size_gb - 5.68) < 0.05
    assert q4.filename == "Qwen3.5-9B-Q4_K_M.gguf"


def test_parse_tree_sums_shards():
    tree = [
        {"path": "M-Q4_K_M-00001-of-00002.gguf", "size": 3_000_000_000},
        {"path": "M-Q4_K_M-00002-of-00002.gguf", "size": 2_000_000_000},
    ]
    quants = M._parse_tree(tree)
    assert len(quants) == 1
    assert abs(quants[0].size_gb - 5.0) < 0.01
    assert quants[0].shards == 2
    assert "00001" in quants[0].filename   # first shard is the representative


def test_quant_quality_ordering():
    q = M.quant_quality
    assert q("Q8_0") > q("Q6_K") > q("Q5_K_M") > q("Q4_K_M") > q("Q3_K_M")
    assert q("Q4_K_M") > q("Q4_K_S")
    assert q("UD-Q4_K_XL") > q("Q4_K_S")
    assert q("Q4_K_M") > q("IQ4_XS")


def test_max_context_scales_and_zeroes():
    m = M.CATALOG_BY_KEY["qwen3.5-9b"]
    # Weights alone exceed a 4 GB budget → no room.
    assert M.max_context_k(4.0, m, 5.68) == 0.0
    # More budget → more context; Q4 KV (÷4) → ~4× more.
    fp16 = M.max_context_k(8.0, m, 5.68, kv_quant_div=1.0)
    q4 = M.max_context_k(8.0, m, 5.68, kv_quant_div=4.0)
    assert fp16 > 0
    assert 3.5 < q4 / fp16 < 4.5


def test_kv_gb_scales_with_context_and_compression():
    m = M.CATALOG_BY_KEY["qwen3.5-9b"]  # ~1.0 GB / 16K fp16
    assert abs(M.kv_gb(m, 16, 1.0) - 1.0) < 0.01
    assert abs(M.kv_gb(m, 256, 1.0) - 16.0) < 0.1     # 256K ≈ 16 GB fp16
    assert abs(M.kv_gb(m, 256, 4.0) - 4.0) < 0.1      # ÷4 with Q4 KV


def test_recommend_targets_full_context_picks_lower_precision():
    # Sizing for the full 256K window under Q4 KV on 8.6 GB forces a low quant:
    # weights must be < 8.6 - kv(256,÷4)=4.0 - overhead 0.8 = 3.8 GB.
    m = M.CATALOG_BY_KEY["qwen3.5-9b"]
    quants = M._parse_tree(FAKE_TREE)
    rec = M.recommend_for_model(8.6, m, quants, target_ctx_k=256, kv_quant_div=4.0)
    assert rec is not None
    r = rec["recommended"]
    assert r.quant.size_gb <= 3.8, r.quant.label
    assert r.fits_target, "recommended quant should reach the full 256K target"
    # And it offers the fidelity trade-off (a higher-quality, shorter-ctx quant).
    assert rec["alt_quality"] is not None
    assert not rec["alt_quality"].fits_target


def test_recommend_shorter_context_allows_higher_precision():
    # Targeting only 32K leaves room for a much better quant on the same budget.
    m = M.CATALOG_BY_KEY["qwen3.5-9b"]
    quants = M._parse_tree(FAKE_TREE)
    rec = M.recommend_for_model(8.6, m, quants, target_ctx_k=32, kv_quant_div=4.0)
    assert rec["recommended"].fits_target
    # Higher fidelity than the 256K-target case above.
    big = M.recommend_for_model(8.6, m, quants, target_ctx_k=256, kv_quant_div=4.0)
    assert (M.quant_quality(rec["recommended"].quant.label)
            > M.quant_quality(big["recommended"].quant.label))


def test_recommend_degrades_when_target_unreachable():
    # Full context impossible even at the smallest quant → recommend max reach,
    # flagged as not covering the target, still returning something usable.
    m = M.CATALOG_BY_KEY["qwen3.5-9b"]
    quants = M._parse_tree(FAKE_TREE)
    rec = M.recommend_for_model(6.0, m, quants, target_ctx_k=256, kv_quant_div=4.0)
    assert rec is not None
    assert not rec["recommended"].fits_target
    assert rec["recommended"].ctx_k >= M.MIN_CTX_K


def test_recommend_none_when_nothing_fits():
    m = M.CATALOG_BY_KEY["qwen3.5-9b"]
    quants = M._parse_tree(FAKE_TREE)
    # 3 GB can't hold a 9B at even the smallest quant + minimal KV.
    assert M.recommend_for_model(3.0, m, quants, kv_quant_div=4.0) is None


def test_candidate_prefilter_by_budget():
    # 8 GB should not surface the 27B; should surface the 9B and smaller.
    keys = {m.key for m in M.candidate_models(8.0)}
    assert "qwen3.5-27b" not in keys
    assert "qwen3.5-9b" in keys and "qwen3.5-4b" in keys
    # A big budget surfaces the 27B.
    assert "qwen3.5-27b" in {m.key for m in M.candidate_models(24.0)}


def test_candidate_family_filter():
    keys = {m.key for m in M.candidate_models(16.0, families=["gemma 4"])}
    assert keys and all(M.CATALOG_BY_KEY[k].family == "Gemma 4" for k in keys)


def test_build_recommendations_offline(monkeypatch):
    # No network: feed canned quants; one repo "unreachable".
    def fake_fetch(repo, timeout=8.0):
        if "4B" in repo:
            return None                      # simulate a fetch failure
        return M._parse_tree(FAKE_TREE)
    monkeypatch.setattr(M, "fetch_quants", fake_fetch)
    recs = M.build_recommendations(8.6)
    assert recs, "should produce recommendations"
    real = [r for r in recs if "recommended" in r]
    assert real and real[0]["model"].tier == 1
    # Real recommendations rank before fetch failures.
    first_fail = next((i for i, r in enumerate(recs) if r.get("fetch_failed")), len(recs))
    last_real = max((i for i, r in enumerate(recs) if "recommended" in r), default=-1)
    assert last_real < first_fail


def test_detect_hardware_returns_struct():
    hw = M.detect_hardware()
    assert isinstance(hw, M.Hardware)
    # budget_gb is RAM or VRAM or None — never raises.
    _ = hw.budget_gb


def test_detection_never_raises_when_all_probes_fail(monkeypatch):
    # Mirrors a Windows box with no NVIDIA GPU and no psutil: every probe
    # comes up empty, but detection returns None rather than throwing.
    monkeypatch.setattr(M, "_detect_ram_gb", lambda: None)
    monkeypatch.setattr(M, "_detect_vram_gb", lambda: None)
    hw = M.detect_hardware()
    assert hw.budget_gb is None


def test_recommend_command_survives_no_hardware(monkeypatch):
    # With no detectable budget and no override, the command must degrade to
    # a hint and return cleanly (this is the case that failed on Windows).
    import commands.config_cmd as cc
    monkeypatch.setattr(M, "detect_hardware", lambda: M.Hardware(None, None))
    assert cc._cmd_model_recommend([], {}) is True


def test_download_url_shape():
    url = M.download_url("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf")
    assert url == ("https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/"
                   "resolve/main/Qwen3.5-9B-Q4_K_M.gguf")


class _FakeResp:
    def __init__(self, status, headers):
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_verify_download_parses_content_range(monkeypatch):
    # 206 with a Content-Range header → ok + parsed total size.
    def fake_urlopen(req, timeout=15.0):
        assert req.headers.get("Range") == "bytes=0-0"
        return _FakeResp(206, {"Content-Range": "bytes 0-0/5680522464"})
    monkeypatch.setattr(M.urllib.request, "urlopen", fake_urlopen)
    ok, total = M.verify_download("https://x/y.gguf")
    assert ok and total == 5680522464


def test_verify_download_falls_back_to_content_length(monkeypatch):
    def fake_urlopen(req, timeout=15.0):
        return _FakeResp(200, {"Content-Length": "123456"})
    monkeypatch.setattr(M.urllib.request, "urlopen", fake_urlopen)
    ok, total = M.verify_download("https://x/y.gguf")
    assert ok and total == 123456


def test_verify_download_handles_errors(monkeypatch):
    def boom(req, timeout=15.0):
        raise OSError("no network")
    monkeypatch.setattr(M.urllib.request, "urlopen", boom)
    ok, total = M.verify_download("https://x/y.gguf")
    assert ok is False and total is None


def test_recommended_links_resolve_live():
    """Network-gated: every recommended download link actually resolves and its
    size matches the fetched HF size (1-byte range, no real download)."""
    quants = M.fetch_quants("unsloth/Qwen3.5-9B-GGUF", timeout=8)
    if not quants:
        import pytest
        pytest.skip("HF unreachable (offline)")
    q = next((x for x in quants if x.label == "Q4_K_M"), quants[0])
    url = M.download_url("unsloth/Qwen3.5-9B-GGUF", q.filename)
    ok, total = M.verify_download(url)
    assert ok, f"recommended download link did not resolve: {url}"
    assert total and abs(total / 1e9 - q.size_gb) < 0.05
    # A bogus file must not falsely verify.
    bad_ok, _ = M.verify_download(M.download_url(
        "unsloth/Qwen3.5-9B-GGUF", "does-not-exist-xyz.gguf"))
    assert not bad_ok


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
