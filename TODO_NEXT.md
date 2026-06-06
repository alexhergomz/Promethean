# TODO — next work

Updated 2026-05-03. Items 1–10 from the previous session are all done
or rolled into new tests; logged below as "shipped" for context.

## Shipped this iteration (rabbit-hole + auto slot paging)
- **Rabbit-hole mode** (`deep-research-rabbit-hole` agent). Long-running
  autonomous research with disk-backed workspace, dedup-by-URL fetch
  (`RabbitFetch`), structured findings, sub-question tree, anti-stuck
  detection, and BM25-based final synthesis reusing the same
  identifier-aware tokenizer that powers `SearchFiles`. Sandboxed via
  tool whitelist + `allowed_root` path jail. 34 tests in
  tests/test_rabbit_hole.py.
- **Auto slot paging in SubAgentManager**: pin subagents to free
  llama-server slots, erase on finish. 6 tests in TestSlotPaging.
- **Rabbit-hole synthesis architecture decisions**: BM25 over findings
  for retrieval, lexical Jaccard for contradiction detection, depth-
  first sub-question tree walk for report layout. No new dependencies.

### Open extensions on the rabbit-hole feature
- ~~Periodic context reset~~ **DONE.** Research-continuity-aware compaction
  in compaction.py:compact_for_research_continuity. Triggers at 50%
  token budget for rabbit-hole agents (vs 70% for others). Keeps system
  prompt + first user message intact; prepends a fresh workspace-state
  block (lossless ground truth) before the LLM-summarized older turns.
  Custom summary prompt prioritizes research continuity. 10 tests in
  TestBuildWorkspaceStateBlock + TestCompactForResearchContinuity.
- ~~Background subagent + park/restore~~ **DONE.** AgentDefinition
  gained a `background: bool` flag. SubAgentManager._background_loop
  blocks on inbox.get() after the initial run, parks the slot to disk
  during idle, restores on message arrival. Exits on cancel,
  workspace-status flip ('finished'), or __SHUTDOWN__ sentinel.
  Cancel + shutdown extended to alive states (running/idle). 5 tests
  in TestBackgroundLoop.
- ~~/research-rabbit-hole slash command~~ **DONE.** `/rabbit-hole`
  with subcommands list/status/msg/stop/report. Aliases /rh and
  /rabbithole. 19 tests in tests/test_rabbit_hole_cmd.py. Side fix:
  SubAgentManager.send_message() also extended to accept idle status
  (was running/pending only) — same kind of bug as cancel/shutdown.
- **Periodic intermediate synthesis**: deferred. `synthesize_workspace(final=False)`
  is callable manually if needed. Auto-call deferred because lossy
  intermediate snapshots could degrade coherence relative to a single
  final synthesis at run end.

## Shipped previously
- **`Think` tool** (Anthropic's no-op scratchpad pattern). Lands a
  thought in the transcript so the model can refer back to it.
  Empirically +54% on tau-bench airline domain in Anthropic's
  original report. ~10 LOC, 3 tests.
- **BM25 + identifier-aware search** in `SearchFiles`. Replaces the
  previous TF-IDF with proper BM25 (k1=1.5, b=0.75), splits identifiers
  on camelCase / snake_case / acronyms / digits, and adds a
  path-component bonus. Encoder-free. New tests for `validateToken`
  matching `validate_token`, partial-word match, and BM25 saturation.
- **Prefix caching** wired into the llama-server side via
  `--cache-reuse 256` on every mode in `run.sh`. First-token latency
  drops to ~0 after the initial turn of a session.
- **TriAttention status corrected** in README/TODO. Calibration stats
  ARE wired (`triattention-ggml/stats/qwen3.5-9b.bin` consumed by
  `tria_maybe_score` every 128 tokens); the open work is binary
  unification, not calibration.
- Symbol-graph navigation: `Neighborhood`, `PathBetween`, `Imports`,
  `SearchFiles` (+ `RepoMap`, `FindSymbol`, `GetCallers`, `Outline`).
  Vendored Aider repomap with dedup fix; bidirectional BFS; transitive
  `Imports(depth=N)`; mtime-fingerprint cache (~160× speedup on repeat).
- CLI viz layer (`agent_tools/visualize.py`) with `/graph-view` toggle,
  Rich-rendered tree / boxed chain / two-panel imports / score bars.
  TTY-auto-detect + `CC_GRAPH_VIEW` env + explicit override.
- Truncation alternation invariant fix: empty assistant after stripping
  malformed tool_calls now gets `[output cut off at max_tokens]` stub
  instead of being popped (which broke user/assistant alternation and
  caused Qwen tool-spam loops).
- Bash deny-list (`tools/security.py:_is_dangerous_bash`) — catastrophic
  patterns (`rm -rf /`, `dd of=/dev/sda`, fork bomb, `curl | sh`, …)
  blocked even in `accept-all`.
- Sensitive-path deny-list (`_is_sensitive_path`) — `~/.ssh`, `~/.aws`,
  `/etc/shadow`, etc. blocked unconditionally for Read/Write/Edit.
- Live UX: Bash output streamed to spinner tail; new-file Write shows
  unified-diff preview; spinner shows last-line of running command.
- Test infra: 716 unit tests passing (was 689). New fixture repo,
  `test_security.py` (63 tests), `test_graph_viz.py`, end-to-end
  truncation drive.

## What's left (real, not low-hanging)

### Capability — bigger lifts
1. **Provider cassettes for streaming/truncation tests.** Anthropic,
   OpenAI, Gemini paths exist but the harness fixes were Qwen-shaped.
   Need a vcr-style recorded-streaming-cassette test per provider:
   tool-call mid-stream, finish_reason=length recovery, rate-limit
   retry, network-drop recovery. Estimate: 1–2 days; needs real keys
   to record once.
2. **Two llama.cpp builds (`tq` vs `tq+`) into one.** Long flagged.
   Either branch-merge or feature-flag at build time. Currently every
   inference fix lands twice or once-and-forgotten. Estimate: 1 day.
3. ~~TriAttention calibration stats.~~ **DONE.** Stats at
   `triattention-ggml/stats/qwen3.5-9b.bin`, loaded via `--triattention`,
   driving `tria_maybe_score` every 128 tokens. The remaining work is
   merging `tq` and `tq+` into one binary so calibration-aware eviction
   becomes a runtime flag rather than a separate build (item 2).

### Robustness
3.5. ~~Auto slot paging in SubAgentManager.~~ **DONE (basic).** Spawn
   pins to idle slot via `id_slot` (forwarded by providers.py
   `extra_body`); finish calls `erase_slot()` to free. Lock-protected
   against concurrent allocation. Opt-in via `enable_slot_paging`
   config flag. Falls back to unpinned + LRU when no idle slot or
   `/slots` unavailable.

   **Open extension:** save+restore (not just erase) for SendMessage
   flows where a parked subagent might receive new input later. Today
   we erase on finish, losing the KV. For multi-turn background
   subagents, switch to `park_slot()` on idle (save+erase) and
   `restore_slot()` on next message. Hooks already exist in
   llama_slots.py; just needs SubAgentManager.send_message() to call
   restore before dispatching to a parked task.

4. **Symbol-graph multi-language verification.** Tree-sitter handles
   JS/TS/Go/Rust if grammars installed; tests + fixture are Python-only.
   Add a Rust mini-fixture and verify `Neighborhood` and `PathBetween`
   work cross-language. Estimate: 2 hours.
5. **`accept-all` defaults to jail at `_worktree_cwd`.** Currently
   `_check_path_allowed` only enforces when `allowed_root` is set.
   Sensitive-path deny-list catches the worst cases, but autonomous
   runs with no explicit root can still read arbitrary user files.
   Default behavior: when `permission_mode=accept-all` and neither
   `allowed_root` nor `_worktree_cwd` is set, fall back to `os.getcwd()`
   with a startup warning. Estimate: 30 min.
6. **Security-test job in CI.** Run `pytest tests/test_security.py`
   as a separate required job so security regressions break the build
   independently of optional-extra tests. Estimate: 10 min.

### Polish (deferred)
- Token + cost display per turn (numbers exist, just surface them).
- `/cost` and `/clear` parity with Claude Code.
- Spinner phrase rotation (currently static per phase).
- `--dry-run` mode that prints the tool plan without executing.

---

## Post-Minimax roadmap (impact-ranked)

Added 2026-05-13 after wiring MiniMax-M2/M1/Text-01 as a first-class
provider (commit pending). Curated for **functionality + effectiveness**,
not UX polish. Each item rated I (impact 1-5) / E (effort, days).

### Tier 1 — quality lifts (do these first)

1. **OptiLLM proxy integration** — I=5, E=0.5d.
   pip-installable proxy (https://github.com/algorithmicsuperintelligence/optillm)
   that wraps any OpenAI-compatible upstream and adds 20+ inference
   techniques (MOA, MCTS, BoN, PlanSearch, MARS, CoT-Reflection,
   self-consistency) selected via model-name prefix. Reported lifts:
   MARS +30 pts on AIME 2025, MOA-GPT-4o-mini matches GPT-4 on
   Arena-Hard, PlanSearch +20% pass@5 on LiveCodeBench.

   **How it fits us:** zero code change in the basic case — run
   `optillm` on `localhost:8000`, set `MINIMAX_BASE_URL=http://localhost:8000/v1`,
   call model `moa-MiniMax-M2`. **What to add:** `/optillm <approach>`
   slash command that wraps the *next* N turns, plus a config knob
   `optillm.default_approach` per agent role (rabbit-hole sub-questions
   benefit from `plansearch`/`mcts`; main coding turns benefit from
   `cot_reflection` or `bon` 2). Cost is N× tokens, so opt-in only.

2. **Per-provider tokenizer calibration** — I=4, E=0.5d.
   Today `compaction.estimate_tokens` uses chars/2.8 for everything.
   Minimax has no public tokenizer and our chars/2.8 undercounts the
   real `usage.prompt_tokens` by ~10-15%. The result: compaction fires
   late on Minimax (silent context-overflow risk) and the cost display
   under-reports. Fix: per-provider correction factor, learned online
   from `(estimated / usage.prompt_tokens)` per turn, smoothed over the
   session, persisted in `~/.promethean/calibration.json`.

3. **Anti-stuck heuristics generalized to the main loop** — I=4, E=0.5d.
   Rabbit-hole already detects "same tool, same args, same result"
   loops. Lift it into the main agent loop (`agent.py:_handle_*`). When
   triggered, inject a meta-prompt with 3 alternative-tool suggestions
   built from the symbol graph (if Edit looping → suggest `RepoMap` /
   `GetCallers` / `PathBetween`). This is the single biggest reliability
   win for weaker models like MiniMax-M2 outside their strongest domain.

### Tier 2 — primitives that unlock everything else

4. **Hooks system in settings.json** — I=5, E=1d.
   Match Claude Code's `hooks.{preToolUse, postToolUse, onStop, onError}`
   driven by `~/.promethean/settings.json` or `.promethean/hooks.json`
   in the repo. Each hook is a shell command with stdin = JSON payload
   (tool name, args, result). Replaces ~5 future "add format-on-save",
   "run mypy on Edit", "auto-commit on Stop" requests with a single
   primitive users can wire themselves.

5. **Cross-provider failover ladder** — I=4, E=1d.
   Extend `circuit_breaker.py`: when a provider trips OPEN, retry the
   *current* request against a fallback ladder defined per "capability
   class". Suggested mappings: reasoning → MiniMax-M2 → DeepSeek-V4-pro
   → claude-sonnet-4-6; fast → MiniMax-M2-highspeed → gpt-4o-mini →
   claude-haiku; long-ctx → MiniMax-M1 → gemini-1.5-pro. Stops a
   transient Minimax 5h quota from killing a session.

6. **JSON-mode shim with auto-repair** — I=3, E=0.5d.
   Minimax has no `response_format`. Add a wrapper that takes a
   pydantic / JSON-Schema model, instructs the model in-prompt, validates
   the response, and does *one* retry pasting the schema violation back.
   Generalizes — also helps Ollama / Kimi / Qwen. Lives in a new
   `structured_output.py`.

### Tier 3 — auto-context (closes the "model forgot to look" gap)

7. **Symbol-graph as automatic context on Edit** — I=4, E=1d.
   When the model invokes Edit/Write on file X, auto-inject (in the
   *next* tool result, capped at e.g. 800 tokens) the most-cited
   callers / callees / co-imports of X derived from our existing
   PageRank symbol graph. Removes the failure mode "model edits X
   without checking who calls X". Implementation: post-tool hook on
   Edit returning a `--- graph context for X ---` block tacked onto
   the tool result. Already have all the machinery in agent_tools/graph/.

8. **FileContextTracker → suggested-Read prefetch** — I=3, E=0.5d.
   Cline-style tracker already records mtime + turn-tag. Add a passive
   heuristic: when a file mentioned in conversation hasn't been Read
   this session, the next assistant turn's first tool_result includes
   a soft suggestion `[Reminder: foo.py was discussed but not read]`.

### Tier 4 — safety / economy

9. **Reversible tool log + `/undo N`** — I=4, E=1d.
   Every Write/Edit/Bash that's side-effectful records an inverse
   (Write→delete or restore prior content; Edit→reverse-patch;
   destructive Bash→snapshot-of-touched-paths). `/undo 3` rewinds the
   last 3 such actions. Reduces accept-all anxiety and replaces "I
   need to git reset" for non-versioned files.

10. **Cost / budget guardrails** — I=3, E=0.5d.
    `/budget 5usd-session 50usd-day` with soft-warn @ 80% and hard
    stop with a confirmation prompt at 100%. Wires to existing
    `quota.py`. Especially relevant under Minimax token plans where
    the 5h rolling bucket is invisible until you hit it.

11. **Worktree isolation for subagents** — I=3, E=1d.
    `EnterWorktree`-style primitive: each subagent gets its own
    `git worktree add` so parallel edits don't collide. Today subagents
    share `cwd`, which fails ugly under multi-agent rabbit-hole +
    main-loop concurrent writes. Auto-merge on Finish (or surface diff
    for user accept).

### Tier 5 — differentiators (heavier, do later)

12. **Conversation branching (git-style)** — I=4, E=2-3d.
    Snapshot the messages array at a chosen turn, fork to try a
    different approach, compare outcomes, keep the better one.
    Slot save/restore is the local-inference primitive we already
    have; the API case is just a messages-array snapshot. Powerful
    for exploratory work. Claude Code is strictly linear so this is
    a genuine differentiator.

13. **Editable compaction** — I=3, E=1d.
    On compaction trigger, render the proposed summary, drop the user
    into `$EDITOR`, commit on save. Lets users surgically rescue
    detail they care about that the compactor would have dropped.

14. **Multi-provider parallel verification in rabbit-hole** — I=3, E=1d.
    Fan a sub-question out to 2-3 providers, BM25-rank claim agreement,
    surface contradictions. Minimax 500 RPM makes this affordable.
    Stronger truth signal than single-model confidence.

15. **Auto-eval at session end** — I=2, E=1d.
    Rubric scorer (goal achievement, tests passing, edit surgical-ness)
    + auto-memory extraction of lessons. Trend-tracked in
    `~/.promethean/sessions.db`.

### Suggested execution order
1 → 2 → 3 → 4 → 5  (Tier 1+2, ~3 days, biggest behavioral lift)
then 7 → 9 → 11    (Tier 3+4 essentials, ~2.5 days)
then revisit 6/8/10/12+ based on observed gaps.

---

Skip / not low-hanging:
- LDA topic modeling — already pushed back, low signal on code corpora.
- Aggressive prompt restructuring — broke local Qwen previously; don't
  retry without a smarter base model.
- 2D-grid graph layout (graphviz-style) — high effort, marginal gain.
- Hardened bash sandbox (proper container/seccomp) — out of scope for
  a CLI tool; the deny-list is the right tradeoff.
