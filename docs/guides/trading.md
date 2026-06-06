# Trading Agent

<div align=center>
<img src="https://github.com/SafeRL-Lab/cheetahclaws/blob/main/docs/trading_demo.gif" width="850"/>
</div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Trading Agent: SSJ → multi-agent analysis (Bull/Bear debate + Risk panel + PM decision) → backtest → indicators</center>
</div>

Promethean includes a built-in AI-powered trading analysis and backtesting module. It combines multi-agent debate-based decision making, technical/fundamental analysis, strategy backtesting, and a BM25 memory system that learns from past trades.

## Quick start

```bash
# 1. Install trading dependencies
pip install "promethean[trading]"
# or individually:
pip install yfinance rank-bm25

# 2. Start Promethean
promethean

# 3. Run a quick analysis
[myproject] » /trading analyze NVDA

# Or use SSJ mode for a guided experience
[myproject] » /ssj
  ⚡ SSJ » 14       ← Trading
  📈 Trading » a    ← Quick Analyze
  Symbol: NVDA
```

---

## Features overview

| Feature | Description |
|---|---|
| Multi-agent analysis | Bull/Bear debate → Research Judge → Risk Panel → Portfolio Manager |
| Technical indicators | 11 indicators: SMA, EMA, MACD, RSI, Bollinger Bands, ATR, VWAP, OBV, ADX, Stochastic, WMA |
| Fundamental analysis | P/E, EPS, revenue, margins, ROE, debt/equity, beta, 52-week range |
| News analysis | Recent news articles with sentiment interpretation |
| Backtesting | 4 built-in strategies with SignalEngine contract, equity + crypto engines |
| BM25 memory | Learn from past decisions — retrieves similar situations for context |
| Reflection | Post-trade analysis feeds lessons back into memory |
| Multi-market | US stocks, HK stocks, A-shares, crypto (20+ coins) |
| Data fallback | Automatic source fallback chains (yfinance → coingecko → akshare) |

---

## Slash commands

| Command | Description |
|---|---|
| `/trading analyze <SYMBOL>` | Full multi-agent analysis pipeline |
| `/trading backtest <SYMBOL> [strategy]` | Backtest a strategy on historical data |
| `/trading price <SYMBOL>` | Quick current price check |
| `/trading indicators <SYMBOL>` | Technical indicators report |
| `/trading status` | Trading memory status across all agent components |
| `/trading history` | View past trading decisions |
| `/trading memory [search\|clear]` | Inspect or manage trading memory |

Alias: `/trade` works the same as `/trading`.

---

## AI tools (callable by the model)

The trading module registers 7 tools that the AI can invoke autonomously:

| Tool | Description | Read-only |
|---|---|---|
| `GetMarketData` | Fetch OHLCV data for any symbol (US/HK/A-share/crypto) | Yes |
| `GetPrice` | Current price and basic metrics | Yes |
| `GetTechnicalIndicators` | Compute 11 technical indicators with formatted report | Yes |
| `GetFundamentals` | P/E, EPS, revenue, margins, ROE, market cap, beta | Yes |
| `GetNews` | Recent news articles for a symbol | Yes |
| `RunBacktest` | Execute a backtest with a built-in strategy | Yes |
| `TradingMemory` | List, search, or clear trading agent memories | No |

---

## Multi-agent analysis pipeline

When you run `/trading analyze NVDA`, the system executes a 5-phase pipeline:

```
Phase 1: Data Collection
  ├── Technical Analysis  → SMA, EMA, MACD, RSI, Bollinger, ATR, OBV, ADX, ...
  ├── Fundamental Analysis → P/E, EPS, revenue, margins, ROE, debt
  └── News Analysis       → Recent articles, sentiment

Phase 2: Bull Researcher
  └── Builds bullish case citing specific data (growth catalysts, technical support)
      Verdict: Strong Buy / Buy / Lean Buy

Phase 3: Bear Researcher
  └── Builds bearish case citing specific data (risks, technical weakness)
      Verdict: Strong Sell / Sell / Lean Sell

Phase 4: Risk Management Panel (3-way debate)
  ├── Aggressive Analyst  → argues for larger position, cites upside
  ├── Conservative Analyst → argues for risk protection, cites downside
  └── Neutral Analyst     → balanced view, optimal sizing

Phase 5: Portfolio Manager (final decision)
  └── RATING: BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL
      + Executive summary, action plan, stop loss, take profit, key risks
```

This design is inspired by [TradingAgents](https://github.com/TauricResearch/TradingAgents), which models real-world trading firm dynamics with specialized roles debating investment decisions.

### BM25 memory integration

Each agent component maintains its own memory store:

- `bull_researcher` — past bullish analyses and outcomes
- `bear_researcher` — past bearish analyses and outcomes
- `trader` — past trade execution decisions
- `risk_judge` — past research arbitration decisions
- `portfolio_manager` — past portfolio decisions

When analyzing a new situation, each agent retrieves the most similar past decisions using BM25 similarity matching. This allows the system to learn from successes and mistakes without retraining or fine-tuning.

Memory is stored at `~/.promethean/trading/memory/` as JSON files.

---

## Backtesting

### Built-in strategies

| Strategy | Logic | Type |
|---|---|---|
| `dual_ma` | SMA(20) vs SMA(50) crossover | Trend following |
| `rsi_mean_reversion` | Buy RSI < 30, sell RSI > 70 | Mean reversion |
| `bollinger_breakout` | Price vs Bollinger Bands(20, 2σ) | Volatility breakout |
| `macd_crossover` | MACD histogram direction | Momentum |

### Usage

```bash
# Backtest a single strategy
/trading backtest AAPL dual_ma

# Compare all strategies (via SSJ)
/ssj → 14 → b → AAPL → 5 (all)

# Or ask the AI directly
> Backtest all 4 strategies on TSLA for the last 2 years and compare
```

### Performance metrics

Each backtest reports:

| Metric | Description |
|---|---|
| Total Return | Cumulative profit/loss percentage |
| Annualized Return | Annualized compound return |
| Sharpe Ratio | Risk-adjusted return (excess return / volatility) |
| Sortino Ratio | Downside risk-adjusted return |
| Calmar Ratio | Return / max drawdown |
| Max Drawdown | Largest peak-to-trough decline |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / gross loss |
| Avg Bars Held | Average holding period per trade |

### SignalEngine contract

The backtesting system uses a standard signal contract inspired by [Vibe-Trading](https://github.com/Vibe-Trading/Vibe-Trading):

```python
# Signal values: -1.0 to 1.0
#  1.0 = fully long  (100% of capital)
#  0.5 = half long   (50%)
#  0.0 = flat        (no position)
# -0.5 = half short  (50% short)
# -1.0 = fully short (100% short)
```

### Backtest engines

| Engine | Markets | Rules |
|---|---|---|
| `EquityEngine` | US stocks, HK stocks | T+0, fractional shares (US), lot-size rounding (HK), stamp tax (HK) |
| `CryptoEngine` | Crypto spot/perpetuals | 24/7 trading, maker/taker fees, funding fees, liquidation checks |

---

## Data sources

### Fallback chains

The data layer automatically tries multiple sources in order:

| Market | Fallback chain |
|---|---|
| US equity | yfinance |
| HK equity | yfinance |
| Crypto | coingecko → yfinance |
| A-share | akshare → yfinance |

### Symbol formats

| Market | Format | Examples |
|---|---|---|
| US stocks | Ticker | `AAPL`, `MSFT`, `NVDA` |
| HK stocks | Code.HK | `0700.HK`, `9988.HK` |
| A-shares | Code.SZ/SH | `000001.SZ`, `600519.SH` |
| Crypto | Symbol | `BTC`, `ETH`, `SOL`, `BTC-USDT` |

### Supported crypto

BTC, ETH, BNB, SOL, XRP, ADA, DOGE, DOT, AVAX, MATIC, LINK, UNI, LTC, ATOM, NEAR, ARB, OP, APT, SUI, SEI

---

## Reflection mechanism

After a trade outcome is known (profit or loss), the reflection system:

1. Analyzes what each agent component got right or wrong
2. Extracts a condensed lesson (~100 words)
3. Stores the lesson in the component's BM25 memory
4. Future analyses retrieve these lessons when facing similar situations

This creates a continuous learning loop without model retraining.

---

## SSJ integration

The trading module is accessible via SSJ Developer Mode (`/ssj` → option 14):

```
╭─ 📈 Trading Agent ━━━━━━━━━━━━━━━━━━━━━━━━━
│
│  a. 🔍  Quick Analyze — Full multi-agent analysis
│  b. 📊  Backtest     — Test a strategy on historical data
│  c. 💰  Price Check  — Current price & key metrics
│  d. 📉  Indicators   — Technical indicators report
│  e. 🤖  Trading Bot  — Launch autonomous trading agent
│  f. 📜  History      — Past trading decisions
│  g. 🧠  Memory       — Trading memory status
│  0. ↩   Back to SSJ
╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Trading Bot** (option e) runs a multi-symbol autonomous analysis. Enter a comma-separated watchlist (default: `AAPL,MSFT,GOOGL,NVDA,BTC,ETH`) and the agent analyzes each symbol through the full pipeline, producing a summary table with ratings.

---

## Autonomous trading agent

Launch via `/agent start trading_agent` or SSJ → 15 → Agent → custom template:

```bash
/agent start trading_agent AAPL,MSFT,GOOGL,NVDA,BTC,ETH
```

The agent iterates through the watchlist, running the full analysis pipeline for each symbol. It maintains a `trading_log.md` with decisions and ratings.

---

## Skills

Three trading skills are available as prompt templates:

| Skill | Trigger | Description |
|---|---|---|
| `trading-analyze` | `/trading-analyze <SYMBOL>` | Full multi-agent analysis |
| `trading-strategy` | `/trading-strategy <desc>` | Generate and backtest a strategy |
| `trading-backtest` | `/trading-backtest <SYMBOL>` | Backtest with comparison table |

---

## Architecture

```
modular/trading/
├── cmd.py                 # /trading command + SSJ sub-menu
├── tools.py               # 7 AI tools (TOOL_DEFS)
├── data/
│   ├── fetchers.py        # Data sources with fallback chains
│   └── indicators.py      # 11 technical indicators (pure Python)
├── engines/
│   ├── base.py            # SignalEngine contract + backtest engine + metrics
│   ├── equity.py          # US/HK equity engine
│   └── crypto.py          # Crypto engine (spot + perpetual)
├── agents/
│   ├── memory.py          # BM25 memory system
│   ├── analyst.py         # Technical / fundamental / news / sentiment
│   ├── researcher.py      # Bull/Bear debate + research judge
│   ├── risk_manager.py    # Aggressive / conservative / neutral panel
│   ├── portfolio_manager.py  # Final decision + signal extraction
│   └── reflection.py      # Post-trade reflection → memory
├── skills/                # 3 markdown skill templates
└── agent_templates/       # Autonomous trading agent template
```

---

## Configuration

Trading data and state are stored at:

| Path | Contents |
|---|---|
| `~/.promethean/trading/memory/` | BM25 memory JSON files (per agent component) |
| `~/.promethean/trading/history/` | Past trading decision records |

No API keys are required for basic usage — yfinance and CoinGecko are free. For A-share data, optionally install `akshare`.

---

## Dependencies

| Package | Required | Purpose |
|---|---|---|
| `yfinance` | Yes | US/HK stock data, fundamentals, news |
| `rank-bm25` | Optional | BM25 memory similarity (falls back to term-overlap) |
| `akshare` | Optional | A-share, futures, forex data |

Install all trading dependencies:

```bash
pip install "promethean[trading]"
```
