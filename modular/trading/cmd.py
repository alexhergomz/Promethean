"""
/trading slash command for Promethean.

Subcommands:
  /trading analyze <SYMBOL>     — full multi-agent analysis
  /trading backtest <strategy>  — backtest a strategy
  /trading price <SYMBOL>       — quick price check
  /trading indicators <SYMBOL>  — technical indicators
  /trading status               — show trading memory status
  /trading history              — view past decisions
  /trading memory [search|clear] — manage trading memory
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ui.render import info, ok, warn, err, clr


# ── History storage ────────────────────────────────────────────────────────

_HISTORY_DIR = Path.home() / ".promethean" / "trading" / "history"


def _save_decision(symbol: str, signal: str, details: str) -> None:
    """Save a trading decision to history."""
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "symbol": symbol,
        "signal": signal,
        "timestamp": timestamp,
        "details": details[:2000],
    }
    path = _HISTORY_DIR / f"{timestamp}_{symbol}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))


# ── Analysis orchestrator ──────────────────────────────────────────────────

def _run_analysis(symbol: str, state, config: dict) -> bool:
    """Run full multi-agent analysis pipeline.

    Flow:
      1. Data collection (technical, fundamental, news)
      2. Bull/Bear researcher debate
      3. Research judge recommendation
      4. Risk management panel debate
      5. Portfolio manager final decision

    The LLM invocations happen via the agent loop — this function
    generates the full analysis prompt and injects it as a message.
    """
    from .agents import analyst, researcher, risk_manager, portfolio_manager

    info(f"\n{'='*60}")
    info(f"  Trading Analysis: {clr(symbol, 'bold')}")
    info(f"  Date: {datetime.now().strftime('%Y-%m-%d')}")
    info(f"{'='*60}\n")

    # Phase 1: Data collection
    info(clr("Phase 1/5: Collecting market data...", "cyan"))
    reports = analyst.run_all_analyses(symbol)

    for name, report in reports.items():
        if "Error" in report:
            warn(f"  {name}: {report}")
        else:
            lines = report.count('\n')
            ok(f"  {name}: {lines} lines collected")

    trade_date = datetime.now().strftime("%Y-%m-%d")

    # Phase 2-5: Build the full multi-agent prompt and return it
    # so the REPL sends it to the AI for processing
    info(clr("\nPhase 2-5: Sending to AI for multi-agent analysis...", "cyan"))
    info(clr("  1. Bull vs Bear research debate", "dim"))
    info(clr("  2. Research judge recommendation", "dim"))
    info(clr("  3. Risk management panel (aggressive/conservative/neutral)", "dim"))
    info(clr("  4. Portfolio manager final decision", "dim"))
    info("")

    full_prompt = _build_analysis_prompt(symbol, trade_date, reports)

    # Return as __ssj_query__ so the REPL sends this to the AI
    return ("__ssj_query__", full_prompt)


def _build_analysis_prompt(
    symbol: str, trade_date: str, reports: dict[str, str]
) -> str:
    """Build the multi-agent analysis prompt.

    Designed to be concise enough for small models (gpt-5-nano) while
    still producing structured output across all 5 phases.
    """
    # Check if we have real data or just errors
    has_data = any(
        "Error" not in reports.get(k, "Error")
        for k in ("technical", "fundamental", "news")
    )

    if has_data:
        data_section = f"""## Market Data for {symbol} (Date: {trade_date})

### Technical
{reports.get('technical', 'N/A')}

### Fundamentals
{reports.get('fundamental', 'N/A')}

### News
{reports.get('news', 'N/A')}"""
    else:
        data_section = f"""## Note: Market data unavailable (yfinance not installed)
Use your general knowledge about {symbol} as of {trade_date}. State clearly when using general knowledge vs. data."""

    return f"""Analyze {symbol} using the 5-phase trading pipeline below. Write each phase ONCE, then move to the next. Do NOT repeat any phase.

{data_section}

---

## Phase 1: BULL CASE
Write 3-4 bullet points arguing FOR buying {symbol} (growth drivers, technical strength, positive catalysts).
End with: **BULL VERDICT: [Buy/Strong Buy]** — one sentence thesis.

## Phase 2: BEAR CASE
Write 3-4 bullet points arguing AGAINST buying {symbol} (risks, overvaluation, negative catalysts).
End with: **BEAR VERDICT: [Sell/Lean Sell]** — one sentence thesis.

## Phase 3: JUDGE DECISION
Which case is stronger? State: **DECISION: BUY / SELL / HOLD** with confidence (High/Medium/Low) and suggested position size (% of portfolio).

## Phase 4: RISK PANEL
- **Aggressive**: one sentence arguing for larger position
- **Conservative**: one sentence arguing for smaller position with tighter stops
- **Neutral**: one sentence with balanced recommendation

## Phase 5: FINAL RATING
**RATING: [BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL]**
**Summary**: 2 sentences on investment thesis.
**Plan**: Entry, Position Size %, Stop Loss %, Take Profit %, Time Horizon.
**Top 3 Risks**: numbered list.
**Conviction**: High / Medium / Low

IMPORTANT: Write each phase exactly once. Do not loop back to Phase 1 after starting Phase 2.
"""


# ── Command handler ────────────────────────────────────────────────────────

def _cmd_trading(args: str, state, config) -> bool:
    """Handle /trading and its subcommands."""
    parts = args.split() if args.strip() else []
    sub = parts[0].lower() if parts else ""
    rest = " ".join(parts[1:])

    if sub == "analyze" or sub == "analyse":
        if not rest:
            err("Usage: /trading analyze <SYMBOL>")
            err("  Example: /trading analyze AAPL")
            return True
        symbol = rest.split()[0].upper()
        return _run_analysis(symbol, state, config)

    elif sub == "backtest":
        return _cmd_backtest(rest, state, config)

    elif sub == "price":
        return _cmd_price(rest)

    elif sub == "indicators":
        return _cmd_indicators(rest)

    elif sub == "status":
        return _cmd_status()

    elif sub == "history":
        return _cmd_history()

    elif sub == "memory":
        return _cmd_memory(rest)

    else:
        _show_help()
        return True


def _cmd_backtest(args: str, state, config) -> bool:
    """Handle /trading backtest."""
    parts = args.split()
    if not parts:
        info("Available strategies:")
        info("  dual_ma          — Dual SMA (20/50) crossover")
        info("  rsi_mean_reversion — RSI 30/70 mean reversion")
        info("  bollinger_breakout — Bollinger Band breakout")
        info("  macd_crossover   — MACD histogram crossover")
        info("")
        info("Usage: /trading backtest <symbol> [strategy] [--capital N]")
        info("  Example: /trading backtest AAPL dual_ma")
        return True

    symbol = parts[0].upper()
    strategy = parts[1] if len(parts) > 1 else "dual_ma"
    capital = 100000

    # Parse --capital flag
    for i, p in enumerate(parts):
        if p == "--capital" and i + 1 < len(parts):
            try:
                capital = float(parts[i + 1])
            except ValueError:
                pass

    info(f"\nRunning backtest: {clr(strategy, 'bold')} on {clr(symbol, 'bold')}")
    info(f"  Capital: ${capital:,.0f}")
    info("")

    from .tools import _run_backtest
    result = _run_backtest(
        {"symbol": symbol, "strategy": strategy, "initial_capital": capital},
        config,
    )
    info(result)
    return True


def _cmd_price(args: str) -> bool:
    """Handle /trading price."""
    if not args:
        err("Usage: /trading price <SYMBOL>")
        return True

    symbol = args.split()[0].upper()
    from .tools import _get_price
    result = _get_price({"symbol": symbol}, {})
    info(result)
    return True


def _cmd_indicators(args: str) -> bool:
    """Handle /trading indicators."""
    if not args:
        err("Usage: /trading indicators <SYMBOL>")
        return True

    symbol = args.split()[0].upper()
    from .tools import _get_technical_indicators
    result = _get_technical_indicators({"symbol": symbol}, {})
    info(result)
    return True


def _cmd_status() -> bool:
    """Show trading memory status."""
    from .agents.memory import get_all_memories
    memories = get_all_memories()

    info(clr("\nTrading Agent Status", "bold"))
    info(f"{'='*40}")
    info(f"\n{'Component':<25} {'Memories':>8}")
    info(f"{'-'*25} {'-'*8}")
    total = 0
    for comp, mem in memories.items():
        count = len(mem)
        total += count
        info(f"  {comp:<23} {count:>6}")
    info(f"{'-'*25} {'-'*8}")
    info(f"  {'Total':<23} {total:>6}\n")

    # Check history
    if _HISTORY_DIR.exists():
        decisions = list(_HISTORY_DIR.glob("*.json"))
        info(f"Past decisions: {len(decisions)}")
    else:
        info("Past decisions: 0")
    return True


def _cmd_history() -> bool:
    """Show past trading decisions."""
    if not _HISTORY_DIR.exists():
        info("No trading history found.")
        return True

    decisions = sorted(_HISTORY_DIR.glob("*.json"), reverse=True)
    if not decisions:
        info("No trading history found.")
        return True

    info(clr("\nTrading Decision History", "bold"))
    info(f"{'='*60}")
    for path in decisions[:20]:
        try:
            record = json.loads(path.read_text())
            ts = record.get("timestamp", "")
            sym = record.get("symbol", "")
            sig = record.get("signal", "")
            sig_clr = "green" if sig in ("BUY", "OVERWEIGHT") else "red" if sig in ("SELL", "UNDERWEIGHT") else "yellow"
            info(f"  {ts}  {sym:<8}  {clr(sig, sig_clr)}")
        except Exception:
            pass
    if len(decisions) > 20:
        info(f"  ... and {len(decisions) - 20} more")
    return True


def _cmd_memory(args: str) -> bool:
    """Handle /trading memory subcommands."""
    parts = args.split() if args else []
    action = parts[0] if parts else "list"
    rest = " ".join(parts[1:])

    from .tools import _trading_memory
    result = _trading_memory(
        {"action": action, "component": rest.split()[0] if rest else "portfolio_manager",
         "query": " ".join(rest.split()[1:]) if len(rest.split()) > 1 else rest},
        {},
    )
    info(result)
    return True


def _show_help() -> bool:
    """Show /trading help."""
    info(clr("\nTrading Agent", "bold"))
    info(f"{'='*50}")
    info("")
    info("  /trading analyze <SYMBOL>     Full multi-agent analysis")
    info("                                  (Bull/Bear debate + Risk panel + PM decision)")
    info("  /trading backtest <SYM> [strat] Run backtest with strategy")
    info("  /trading price <SYMBOL>        Quick price check")
    info("  /trading indicators <SYMBOL>   Technical indicators report")
    info("  /trading status                Trading memory status")
    info("  /trading history               Past decision history")
    info("  /trading memory [action]       Manage trading memory")
    info("")
    info("  Strategies: dual_ma, rsi_mean_reversion, bollinger_breakout, macd_crossover")
    info("")
    return True


# ── Export ─────────────────────────────────────────────────────────────────

COMMAND_DEFS = {
    "trading": {
        "func": _cmd_trading,
        "help": (
            "AI trading agent — multi-agent analysis and backtesting",
            ["analyze", "backtest", "price", "indicators", "status", "history", "memory"],
        ),
        "aliases": ["trade"],
    },
}
