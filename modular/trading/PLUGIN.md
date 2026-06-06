---
name: trading
version: 1.0.0
description: Trading agent module — multi-agent analysis, backtesting, and strategy generation
author: promethean
tags: [trading, finance, backtest, agent, strategy]
commands:
  - modular.trading.cmd
dependencies:
  - yfinance
  - rank-bm25
homepage: ""
---

# Trading Module

AI-powered multi-agent trading analysis and backtesting system.

## Features

- **Multi-Agent Decision System** — Bull/Bear debate, risk management panel, portfolio manager
- **Backtesting Engine** — SignalEngine contract with equity and crypto engines
- **Data Layer** — Multiple data sources with automatic fallback chains
- **BM25 Memory** — Learn from historical trading decisions
- **Reflection** — Post-trade analysis feeds back into memory

## Commands

- `/trading analyze <SYMBOL>`     — full multi-agent analysis (Bull/Bear debate + risk + PM decision)
- `/trading backtest <strategy>`  — generate and backtest a trading strategy
- `/trading status`               — show active positions and recent signals
- `/trading history`              — view past trading decisions and reflections
- `/trading memory`               — inspect trading memory (learned patterns)

## Dependencies

| Package      | Feature                        |
|--------------|--------------------------------|
| `yfinance`   | Market data (US/HK equities)   |
| `rank-bm25`  | Memory similarity matching     |
