"""Threaded sub-agent system for spawning nested agent loops."""
from __future__ import annotations

import os
import threading
import uuid
import queue
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Set


# ── Agent definition ───────────────────────────────────────────────────────

@dataclass
class AgentDefinition:
    """Definition for a specialized agent type."""
    name: str
    description: str = ""
    system_prompt: str = ""   # extra instructions prepended to the base system prompt
    model: str = ""            # model override; "" = inherit from parent
    tools: list = field(default_factory=list)   # empty list = all tools
    source: str = "user"       # "built-in" | "user" | "project"
    # When True, the spawned subagent enters a background loop after its
    # initial run completes: it blocks on inbox.get(), so SendMessage can
    # deliver new messages over the lifetime of the agent. Slot paging
    # (when enabled) parks the slot during idle periods to free VRAM.
    # The agent exits when cancelled or when its workspace status flips
    # to "finished" (rabbit-hole's Finish() tool does this).
    background: bool = False


# ── Built-in agent definitions ─────────────────────────────────────────────

_BUILTIN_AGENTS: Dict[str, AgentDefinition] = {
    "general-purpose": AgentDefinition(
        name="general-purpose",
        description=(
            "General-purpose agent for researching complex questions, "
            "searching for code, and executing multi-step tasks."
        ),
        system_prompt="",
        source="built-in",
    ),
    "coder": AgentDefinition(
        name="coder",
        description="Specialized coding agent for writing, reading, and modifying code.",
        system_prompt=(
            "You are a specialized coding assistant. Focus on:\n"
            "- Writing clean, idiomatic code\n"
            "- Reading and understanding existing code before modifying\n"
            "- Making minimal targeted changes\n"
            "- Never adding unnecessary features, comments, or error handling\n"
        ),
        source="built-in",
    ),
    "reviewer": AgentDefinition(
        name="reviewer",
        description="Code review agent analyzing quality, security, and correctness.",
        system_prompt=(
            "You are a code reviewer. Analyze code for:\n"
            "- Correctness and logic errors\n"
            "- Security vulnerabilities (injection, XSS, auth bypass, etc.)\n"
            "- Performance issues\n"
            "- Code quality and maintainability\n"
            "Be concise and specific. Categorize findings as: Critical | Warning | Suggestion.\n"
        ),
        tools=["Read", "Glob", "Grep"],
        source="built-in",
    ),
    "researcher": AgentDefinition(
        name="researcher",
        description="Research agent for exploring codebases and answering questions.",
        system_prompt=(
            "You are a research assistant focused on understanding codebases.\n"
            "- Read and analyze code thoroughly before answering\n"
            "- Provide factual, evidence-based answers\n"
            "- Cite specific file paths and line numbers\n"
            "- Be concise and focused\n"
        ),
        tools=["Read", "Glob", "Grep", "WebFetch", "WebSearch"],
        source="built-in",
    ),
    "tester": AgentDefinition(
        name="tester",
        description="Testing agent that writes and runs tests.",
        system_prompt=(
            "You are a testing specialist. Your job:\n"
            "- Write comprehensive tests for the given code\n"
            "- Run existing tests and diagnose failures\n"
            "- Focus on edge cases and error conditions\n"
            "- Keep tests simple, readable, and fast\n"
        ),
        source="built-in",
    ),
    "deep-research-rabbit-hole": AgentDefinition(
        name="deep-research-rabbit-hole",
        description=(
            "Long-running autonomous research agent. Runs UNTIL KILLED by "
            "the user (or until it calls Finish). Decomposes the question "
            "into sub-questions, fetches deduped web sources, saves "
            "structured findings to a workspace on disk, and stays bounded "
            "via aggressive context elision + anti-stuck heuristics. The "
            "final synthesis is built from the on-disk findings via BM25, "
            "not from the agent's context — so the agent's reasoning is "
            "decoupled from the size of the knowledge base. Use for: "
            "open-ended scouting where you want depth ('research X "
            "exhaustively'), comparative surveys ('all known approaches "
            "to Y'), state-of-the-art reports."
        ),
        system_prompt=(
            "You are a long-running autonomous deep-research agent. You run "
            "UNTIL THE USER KILLS YOU or until you call Finish.\n"
            "\n"
            "═══════════════════════════════════════════════════════════════\n"
            "CRITICAL RULES (read these first, follow them exactly):\n"
            "  • Your FIRST response MUST be one or more AddSubQuestion tool "
            "calls. Do NOT write any preamble, explanation, or 'I will now…' "
            "text. Go directly to tools.\n"
            "  • Every response MUST contain at least one tool call. Do not "
            "respond with text-only. Text-only replies are wasted turns.\n"
            "  • You CANNOT stop yourself. There is no Finish tool, no "
            "termination condition you can trigger. ONLY the user can "
            "kill this run (via /rabbit-hole stop). You will be re-prompted "
            "to continue researching as long as the run is alive. Do not "
            "try to wrap up, conclude, or signal completion — just keep "
            "researching, broadening, deepening. Synthesis happens "
            "automatically when the user stops you.\n"
            "  • All your tools are auto-approved — you NEVER need to ask "
            "permission. If a tool ever returns a 'Denied' error, it's a "
            "system bug (not a user rejection); try a different tool, do "
            "not retry the same one in a loop. You are running in a "
            "sandboxed workspace where whitelisted tools all just work.\n"
            "  • You communicate with the USER through Note(text). You do "
            "your private reasoning through Think(thought). Do NOT write "
            "free-form text outside these tools.\n"
            "  • Topic sensitivity: research ANY topic the user asks about. "
            "You are an investigative tool, not a moderator. Refusing to "
            "research a controversial topic is the wrong choice — gather "
            "evidence neutrally, attribute claims to sources, surface "
            "contradictions. The user is responsible for what they do with "
            "the report.\n"
            "═══════════════════════════════════════════════════════════════\n"
            "\n"
            "Your goal is to investigate a question by gathering evidence "
            "and saving structured findings to a workspace.\n"
            "\n"
            "WORKFLOW:\n"
            "1. Read the user's question. Your FIRST tool call sequence: "
            "3-5 AddSubQuestion calls decomposing the question into "
            "investigable parts. No preamble.\n"
            "2. Use ListOpenQuestions to see what's pending. Pick the most "
            "important one.\n"
            "3. Use WebSearch + RabbitFetch to gather sources. RabbitFetch "
            "automatically caches by URL — if you've already fetched a URL, "
            "you'll get the cached content with a [CACHED] marker, so you "
            "don't waste calls. ALWAYS prefer RabbitFetch over WebFetch.\n"
            "4. As you gather evidence, save SPECIFIC findings via "
            "SaveFinding. One finding = one claim + the URLs supporting it. "
            "Granular findings produce a better final report. Don't write "
            "essay-long claims; write atomic ones.\n"
            "5. When a sub-question feels answered, MarkQuestionDone with a "
            "1-2 sentence summary. The summary becomes that section's body "
            "in the final report.\n"
            "6. If new lines of inquiry surface, AddSubQuestion (use "
            "parent_id to nest under the current question).\n"
            "7. Repeat from step 2.\n"
            "\n"
            "DEDUP: RabbitFetch caches by URL. ListSources shows what you've "
            "already pulled. SearchFindings searches your saved findings — "
            "use it to check what you already concluded before researching "
            "again.\n"
            "\n"
            "CONTEXT MANAGEMENT: You accumulate context fast. The harness "
            "elides old tool outputs aggressively, but you should also use "
            "Think SPARINGLY — only for genuinely useful intermediate "
            "consolidations between rounds, not for narrating each step. "
            "The workspace is your real memory; your context is just the "
            "current focus.\n"
            "\n"
            "PROGRESS VISIBILITY: The user is watching you work via "
            "/rabbit-hole status. Use the Note tool every few turns to "
            "log a SHORT (1-2 sentence) outward-facing progress note: "
            "'Switching focus from inference to training', 'Found "
            "surprising claim in [X], will verify next', 'Marking q-abc "
            "done — insufficient evidence'. Note is for the user; Think "
            "is for you. Don't duplicate your Think output as a Note. "
            "One note every 3-5 turns is the right cadence.\n"
            "\n"
            "ANTI-STUCK: If you're not finding new info on a sub-question, "
            "MarkQuestionDone with summary '(insufficient evidence found)' "
            "and move on. Don't loop. The harness watches your progress "
            "counters and may inject a nudge if you're stuck for too long.\n"
            "\n"
            "WHEN TO STOP: You don't. The user controls termination via "
            "/rabbit-hole stop, which triggers automatic synthesis from "
            "your saved findings. Until then, keep gathering. If you feel "
            "you've exhausted the easy angles, broaden: spawn related "
            "sub-questions, look at adjacent fields, check skeptics' "
            "counter-arguments, dig into specific source's caveats, "
            "compare across sources for inconsistencies. There is always "
            "more depth available."
        ),
        tools=[
            "WebSearch",       # primary discovery
            "RabbitFetch",     # deduped fetch
            "Think",           # private scratchpad
            "Note",            # outward-facing progress note for the user
            "AddSubQuestion",
            "ListOpenQuestions",
            "MarkQuestionDone",
            "SaveFinding",
            "SearchFindings",
            "ListSources",
            # Read is allowed but path-jailed to the workspace via
            # eff_config["allowed_root"] — set in spawn().
            "Read",
            # NOTE: Finish is intentionally NOT in this whitelist. The
            # rabbit-hole agent runs UNTIL THE USER KILLS IT — there is
            # no self-stop. Without this restriction, a confused model
            # would call Finish() in a tight loop trying to escape.
            # Pattern stolen from test-time-scaling: force the agent to
            # keep producing useful work even when it wants to stop.
        ],
        source="built-in",
        # Background mode: after the initial decomposition + research,
        # the agent blocks on its inbox so the user can SendMessage to
        # add hints, redirect, or shut down without losing state. The
        # workspace persists across messages.
        background=True,
    ),
    "deep-research": AgentDefinition(
        name="deep-research",
        description=(
            "Deep research agent for open-ended questions that require web "
            "investigation across multiple sources. Decomposes the question, "
            "gathers evidence with WebSearch+WebFetch, uses Think to "
            "consolidate findings between rounds, cross-references claims, "
            "and returns a structured report. Best for: 'find SOTA "
            "technique for X', 'compare implementation approaches across "
            "these repos', 'what happened in [field] in 2025', open-ended "
            "scouting reports."
        ),
        system_prompt=(
            "You are a deep research agent. Your task is to investigate an "
            "open-ended question thoroughly and return a synthesized report. "
            "You have your own context window — the parent agent does not "
            "see your scratch work, only your final report — so be free in "
            "your exploration but disciplined in your output.\n"
            "\n"
            "Approach:\n"
            "1. Decompose the question into 3-5 specific sub-questions. "
            "Write them down with the Think tool first so you can track "
            "progress.\n"
            "2. For each sub-question, use WebSearch + WebFetch to gather "
            "evidence from at least 2-3 distinct sources. Prefer primary "
            "sources (papers, official docs, repos) over secondary "
            "(blog posts, summaries).\n"
            "3. Use the Think tool between rounds to consolidate findings, "
            "note contradictions, and decide what to investigate next. "
            "This is your private scratchpad — be candid.\n"
            "4. Cross-reference claims across sources — flag anything "
            "supported by only one source as 'unverified'. If sources "
            "disagree, surface the disagreement, don't paper over it.\n"
            "5. Synthesize a final report with: a 2-3 sentence TL;DR, "
            "key findings (bullet list with source citations as "
            "[name](url)), open questions / contradictions, and a sources "
            "section at the bottom.\n"
            "\n"
            "Be honest. If you couldn't find evidence for a claim, say so "
            "explicitly. Don't synthesize plausible-sounding answers from "
            "thin air. If the question itself is unanswerable as posed, "
            "say that and explain why."
        ),
        tools=["WebSearch", "WebFetch", "Think", "Read", "Grep"],
        source="built-in",
    ),
}


# ── Loading agent definitions from .md files ──────────────────────────────

def _parse_agent_md(path: Path, source: str = "user") -> AgentDefinition:
    """Parse a .md file with optional YAML frontmatter into an AgentDefinition.

    File format:
        ---
        description: "Short description"
        model: claude-haiku-4-5-20251001
        tools: [Read, Write, Edit, Bash]
        ---

        System prompt body goes here...
    """
    content = path.read_text()
    name = path.stem
    description = ""
    model = ""
    tools: list = []
    system_prompt_body = content

    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm_text = content[3:end].strip()
            system_prompt_body = content[end + 3:].strip()
            try:
                import yaml as _yaml
                fm = _yaml.safe_load(fm_text) or {}
            except ImportError:
                # Manual key: value parse (no yaml dependency required)
                fm: dict = {}
                for line in fm_text.splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        fm[k.strip()] = v.strip()
            description = str(fm.get("description", ""))
            model = str(fm.get("model", ""))
            raw_tools = fm.get("tools", [])
            if isinstance(raw_tools, list):
                tools = [str(t) for t in raw_tools]
            elif isinstance(raw_tools, str):
                # Handle "[Read, Write]" or "Read, Write" format
                s = raw_tools.strip("[]")
                tools = [t.strip() for t in s.split(",") if t.strip()]

    return AgentDefinition(
        name=name,
        description=description,
        system_prompt=system_prompt_body,
        model=model,
        tools=tools,
        source=source,
    )


def load_agent_definitions() -> Dict[str, AgentDefinition]:
    """Load all agent definitions: built-ins → user-level → project-level.

    Search paths:
      ~/.promethean/agents/*.md   (user-level)
      .promethean/agents/*.md     (project-level, overrides user)
    """
    defs: Dict[str, AgentDefinition] = dict(_BUILTIN_AGENTS)

    # User-level
    user_dir = Path.home() / ".promethean" / "agents"
    if user_dir.is_dir():
        for p in sorted(user_dir.glob("*.md")):
            try:
                d = _parse_agent_md(p, source="user")
                defs[d.name] = d
            except Exception:
                pass

    # Project-level (overrides user)
    proj_dir = Path.cwd() / ".promethean" / "agents"
    if proj_dir.is_dir():
        for p in sorted(proj_dir.glob("*.md")):
            try:
                d = _parse_agent_md(p, source="project")
                defs[d.name] = d
            except Exception:
                pass

    return defs


def get_agent_definition(name: str) -> Optional[AgentDefinition]:
    """Look up an agent definition by name. Returns None if not found."""
    return load_agent_definitions().get(name)


# ── SubAgentTask ───────────────────────────────────────────────────────────

@dataclass
class SubAgentTask:
    """Represents a sub-agent task with lifecycle tracking."""
    id: str
    prompt: str
    status: str = "pending"       # pending | running | completed | failed | cancelled
    result: Optional[str] = None
    depth: int = 0
    name: str = ""                # optional human-readable name (addressable by SendMessage)
    worktree_path: str = ""       # set if isolation="worktree"
    worktree_branch: str = ""     # set if isolation="worktree"
    slot_id: Optional[int] = None # llama-server slot (when slot paging enabled)
    slot_parked: bool = False     # True while in background-idle with KV saved to disk
    rabbit_hole_dir: str = ""     # set when running as deep-research-rabbit-hole
    _cancel_flag: bool = False
    _future: Optional[Future] = field(default=None, repr=False)
    _inbox: Any = field(default_factory=queue.Queue, repr=False)  # for send_message


# ── Worktree helpers ───────────────────────────────────────────────────────

def _git_root(cwd: str) -> Optional[str]:
    """Return the git root directory for cwd, or None if not in a git repo."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return None


def _create_worktree(base_dir: str) -> tuple:
    """Create a temporary git worktree.

    Returns:
        (worktree_path, branch_name)
    Raises:
        subprocess.CalledProcessError or OSError on failure.
    """
    branch = f"nano-agent-{uuid.uuid4().hex[:8]}"
    # mkdtemp gives us a path; remove the empty dir so git can create it
    wt_path = tempfile.mkdtemp(prefix="nano-agent-wt-")
    os.rmdir(wt_path)
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, wt_path],
        cwd=base_dir, check=True, capture_output=True, text=True,
    )
    return wt_path, branch


def _remove_worktree(wt_path: str, branch: str, base_dir: str) -> None:
    """Remove a git worktree and delete its branch (best-effort)."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", wt_path],
            cwd=base_dir, capture_output=True,
        )
    except Exception:
        pass
    try:
        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=base_dir, capture_output=True,
        )
    except Exception:
        pass


# ── Internal helpers ───────────────────────────────────────────────────────

def _capture_agent_event_to_workspace(ev, rabbit_hole_dir: str) -> None:
    """For rabbit-hole runs, persist every agent event to disk so the
    user can see what the model is actually doing — including text-only
    responses and Think output that don't trigger workspace events on
    their own.

    Without this, when the model replies with text instead of tool
    calls (which 9B-class models do often), the user sees nothing
    accumulating in /rabbit-hole status and has no way to diagnose.

    Best-effort: capture must never break the run.
    """
    if not rabbit_hole_dir:
        return
    try:
        from rabbit_hole.store import RabbitHoleWorkspace
        ws_cap = RabbitHoleWorkspace(rabbit_hole_dir)
        turn = ws_cap.state.turn
        kind = type(ev).__name__
        if kind in ("TextChunk", "ThinkingChunk"):
            text = getattr(ev, "text", "")
            if not text:
                return
            tag = "thinking" if kind == "ThinkingChunk" else "text"
            transcript_path = ws_cap.notes_dir / f"turn-{turn:05d}-{tag}.md"
            with transcript_path.open("a", encoding="utf-8") as f:
                f.write(text)
            return
        payload = {}
        if kind == "ToolStart":
            payload = {
                "tool": getattr(ev, "name", "?"),
                "inputs_preview": str(getattr(ev, "inputs", {}))[:200],
            }
        elif kind == "ToolEnd":
            payload = {
                "tool": getattr(ev, "name", "?"),
                "result_preview": str(getattr(ev, "result", ""))[:200],
            }
        else:
            return
        ws_cap.append_event(f"agent_{kind.lower()}", **payload)
    except Exception:
        pass


def _agent_run(prompt, state, config, system_prompt, depth=0, cancel_check=None):
    """Lazy-import wrapper to avoid circular dependency with agent module.

    Uses absolute import so this works whether called from inside or outside
    the multi_agent package (sys.path includes the project root).
    """
    import agent as _agent_mod
    return _agent_mod.run(prompt, state, config, system_prompt, depth=depth, cancel_check=cancel_check)


def _extract_final_text(messages):
    """Walk backwards through messages, return first assistant content string."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return None


# ── SubAgentManager ────────────────────────────────────────────────────────

class SubAgentManager:
    """Manages concurrent sub-agent tasks using a thread pool."""

    def __init__(self, max_concurrent: int = 5, max_depth: int = 5):
        self.tasks: Dict[str, SubAgentTask] = {}
        self._by_name: Dict[str, str] = {}   # name → task_id
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent)
        # Slot paging — set of slot ids currently checked out to a subagent.
        # Only populated when config["enable_slot_paging"] is True. Lock
        # protects against two spawn() calls allocating the same slot
        # before either has actually issued its first chat completion.
        self._allocated_slots: Set[int] = set()
        self._slot_lock: threading.Lock = threading.Lock()

    def _allocate_slot(self, server_url: str) -> Optional[int]:
        """Find an idle llama-server slot and reserve it for a subagent.

        Returns the slot id or None if no slot is free / the server doesn't
        expose the /slots admin endpoint. Caller is responsible for
        eventually calling _release_slot(slot_id).
        """
        from llama_slots import LlamaSlotsError, list_slots
        with self._slot_lock:
            try:
                slots = list_slots(server_url)
            except LlamaSlotsError:
                return None
            for s in slots:
                if s.state == "idle" and s.id not in self._allocated_slots:
                    self._allocated_slots.add(s.id)
                    return s.id
            return None

    def _release_slot(self, slot_id: int, server_url: str) -> None:
        """Erase the slot's KV (free it for reuse) and remove from
        the allocated-set. Safe to call on a slot we never allocated —
        the discard is a no-op in that case.
        """
        from llama_slots import LlamaSlotsError, erase_slot
        with self._slot_lock:
            self._allocated_slots.discard(slot_id)
        try:
            erase_slot(slot_id, server_url=server_url)
        except LlamaSlotsError:
            pass  # best-effort — server may have died, slot may be gone

    def _background_loop(
        self,
        task: SubAgentTask,
        state: Any,
        eff_config: dict,
        eff_system: str,
        child_depth: int,
        rabbit_hole_dir: Optional[str],
        allocated_slot: Optional[int],
        slot_server_url: str,
    ) -> None:
        """Background subagent loop — runs after the initial prompt
        completes. The agent stays alive, blocking on its inbox until:
          • the user cancels (task._cancel_flag set)
          • the rabbit-hole workspace is marked finished (Finish() tool)
          • a special "__SHUTDOWN__" message is delivered

        While idle (no message pending) the slot is parked to disk via
        save+erase if slot paging is enabled, freeing VRAM allocation
        for other concurrent subagents. On message arrival the slot is
        restored before dispatching to the agent loop.
        """
        import queue as _q
        from llama_slots import (
            LlamaSlotsError, park_slot, restore_slot,
        )

        slot_paging_active = (
            allocated_slot is not None and bool(slot_server_url)
        )
        parking_filename = f"task-{task.id}.bin"

        def _try_park() -> None:
            if not slot_paging_active:
                return
            try:
                park_slot(allocated_slot, parking_filename, slot_server_url)
                task.slot_parked = True
            except LlamaSlotsError:
                # --slot-save-path missing or server doesn't support
                # save/restore — silently fall back to slot-pinned-but-not-paged
                pass

        def _try_restore() -> None:
            if not slot_paging_active or not task.slot_parked:
                return
            try:
                restore_slot(allocated_slot, parking_filename, slot_server_url)
                task.slot_parked = False
            except LlamaSlotsError:
                # If restore fails the model just refills the slot from
                # scratch on its next request — slower but correct.
                task.slot_parked = False

        def _is_finished_via_workspace() -> bool:
            if not rabbit_hole_dir:
                return False
            try:
                from rabbit_hole.store import RabbitHoleWorkspace
                ws = RabbitHoleWorkspace(rabbit_hole_dir)
                return ws.manifest.get("status") == "finished"
            except Exception:
                return False

        # Force-continue prompt for rabbit-hole agents. When the model
        # exits its turn with no tool calls (would naturally idle), we
        # inject this and re-run. The point of rabbit-hole is to never
        # stop on its own — only the user can kill it. This is the
        # test-time-scaling trick: prevent the model from terminating.
        FORCE_CONTINUE_PROMPT = (
            "Continue researching. Pick the most relevant open sub-question "
            "(use ListOpenQuestions if you need a refresher), or broaden "
            "into a new angle. If you've exhausted easy material, dig into "
            "skeptics' counter-arguments, source caveats, contradictions "
            "between sources, or adjacent fields. SaveFinding for anything "
            "specific you've discovered but not yet committed. There is "
            "always more depth available."
        )
        # How long to wait for an inbox message before force-continuing.
        # Short so the agent stays productive; long enough that user
        # /rabbit-hole msg has a chance to land.
        IDLE_BEFORE_FORCE_CONTINUE_S = 2.0

        while not task._cancel_flag:
            if _is_finished_via_workspace():
                # Only fires if config.json/whitever marked finished
                # externally. The agent itself can't trigger this since
                # Finish is no longer in the rabbit-hole tool whitelist.
                break

            # Park the slot during idle. Quick check first to avoid
            # park-then-immediately-restore when an inbox message is
            # already queued.
            if task._inbox.empty():
                _try_park()

            task.status = "idle"

            # Get next message: SendMessage if any, else force-continue.
            try:
                inbox_msg = task._inbox.get(timeout=IDLE_BEFORE_FORCE_CONTINUE_S)
            except _q.Empty:
                # Force-continue ONLY for rabbit-hole agents. Other
                # background agents stay genuinely idle.
                if rabbit_hole_dir:
                    inbox_msg = FORCE_CONTINUE_PROMPT
                else:
                    continue

            if inbox_msg == "__SHUTDOWN__":
                break

            _try_restore()
            task.status = "running"

            try:
                gen = _agent_run(
                    inbox_msg, state, eff_config, eff_system,
                    depth=child_depth,
                    cancel_check=lambda: task._cancel_flag,
                )
                for _ev in gen:
                    if rabbit_hole_dir:
                        _capture_agent_event_to_workspace(_ev, rabbit_hole_dir)
                    if task._cancel_flag:
                        break
                if not task._cancel_flag:
                    task.result = _extract_final_text(state.messages)
                    task.status = "completed"
            except Exception as e:
                task.status = "failed"
                task.result = f"Error in background turn: {e}"
                # On exception, exit the loop — the user will see the
                # error in task.result. Don't keep retrying.
                break

        # Loop exit reasons:
        #   • task._cancel_flag set    → status should be "cancelled"
        #   • workspace finished       → status stays "completed" (last
        #                                  successful turn) or "idle" if
        #                                  finished was set during idle
        #   • __SHUTDOWN__ inbox msg   → same as above
        # The "idle on exit" case is awkward — the agent stopped doing
        # work but never explicitly completed. Map it to "completed" if
        # there's any result, else leave "idle".
        if task._cancel_flag:
            task.status = "cancelled"
        elif task.status == "idle":
            task.status = "completed" if task.result else "idle"

    def spawn(
        self,
        prompt: str,
        config: dict,
        system_prompt: str,
        depth: int = 0,
        agent_def: Optional[AgentDefinition] = None,
        isolation: str = "",     # "" | "worktree"
        name: str = "",
        resume_workspace_dir: str = "",   # rabbit-hole only — reuse existing
    ) -> SubAgentTask:
        """Spawn a new sub-agent task.

        Args:
            prompt:       user message for the sub-agent
            config:       agent configuration dict (copied before modification)
            system_prompt: base system prompt
            depth:        current nesting depth (prevents infinite recursion)
            agent_def:    optional AgentDefinition with model/system_prompt/tools overrides
            isolation:    "" for normal, "worktree" for isolated git worktree
            name:         optional human-readable name (addressable via SendMessage)

        Returns:
            SubAgentTask tracking the spawned work.
        """
        task_id = uuid.uuid4().hex[:12]
        short_name = name or task_id[:8]
        task = SubAgentTask(id=task_id, prompt=prompt, depth=depth, name=short_name)
        self.tasks[task_id] = task
        if name:
            self._by_name[name] = task_id

        if depth >= self.max_depth:
            task.status = "failed"
            task.result = f"Max depth ({self.max_depth}) exceeded"
            return task

        # Build effective config and system prompt for this sub-agent
        eff_config = dict(config)
        eff_system = system_prompt

        if agent_def:
            if agent_def.model:
                eff_config["model"] = agent_def.model
            if agent_def.system_prompt:
                eff_system = agent_def.system_prompt.rstrip() + "\n\n" + system_prompt
            # Enforce the tool whitelist by exposing only the named tool
            # schemas to the agent loop. agent.py reads this from config and
            # passes the filtered schema list to providers.stream(). Without
            # this, a 9B model dumped with 50+ tool schemas struggles to
            # pick the right ones and often just emits text instead of
            # calling any tool — which looks like "nothing happened" from
            # the user's side.
            if agent_def.tools:
                eff_config["_agent_tools_whitelist"] = list(agent_def.tools)

        # Rabbit-hole mode — set up workspace, enable aggressive elision,
        # and path-jail the agent to its workspace dir.
        #
        # Workspace base path resolves in this order (most-specific first):
        #   1. config["_rabbit_hole_base_dir"]  — explicit per-spawn override
        #   2. env var PROMETHEAN_RABBIT_HOLE_DIR — process-wide override (used
        #      by tests to redirect to tmp_path so they don't pollute the
        #      user's real ~/.promethean/rabbit-hole/)
        #   3. ~/.promethean/rabbit-hole/         — production default
        rabbit_hole_dir: Optional[str] = None
        is_resume: bool = False
        if agent_def and agent_def.name == "deep-research-rabbit-hole":
            from pathlib import Path as _P
            from rabbit_hole.store import RabbitHoleWorkspace
            if resume_workspace_dir:
                # Reuse an existing workspace: the new agent picks up the
                # findings, sub-questions, and source cache from prior runs
                # and continues. The task_id changes (new SubAgentTask)
                # but the workspace dir does not.
                rabbit_hole_dir = str(_P(resume_workspace_dir).resolve())
                if not (_P(rabbit_hole_dir) / "manifest.json").exists():
                    task.status = "failed"
                    task.result = (f"Resume target has no manifest.json: "
                                   f"{rabbit_hole_dir}")
                    return task
                ws_resume = RabbitHoleWorkspace(rabbit_hole_dir)
                # Bump the manifest status back to active in case the
                # prior run had Finish'd or been cancelled.
                ws_resume.manifest["status"] = "active"
                ws_resume.manifest["resumed_at"] = ws_resume.manifest.get(
                    "resumed_at", []) or []
                if not isinstance(ws_resume.manifest["resumed_at"], list):
                    ws_resume.manifest["resumed_at"] = [ws_resume.manifest["resumed_at"]]
                import time as _t
                ws_resume.manifest["resumed_at"].append(_t.time())
                ws_resume.manifest_path.write_text(
                    __import__("json").dumps(ws_resume.manifest, indent=2))
                # Build a continuation-aware prompt that tells the model
                # what's already in the workspace so it doesn't redo work.
                n_q_open = len(ws_resume.list_open_questions())
                n_q_total = len(ws_resume.list_all_questions())
                n_findings = len(ws_resume.list_findings())
                n_sources = len(ws_resume.list_sources())
                prompt = (
                    f"RESUMING research run on workspace {rabbit_hole_dir}.\n\n"
                    f"Original question: {ws_resume.manifest.get('root_question', '?')}\n\n"
                    f"State carried over from prior run(s):\n"
                    f"  • {n_q_total} sub-questions ({n_q_open} still open)\n"
                    f"  • {n_findings} findings already saved\n"
                    f"  • {n_sources} sources already fetched (deduped — "
                    f"RabbitFetch on these URLs returns CACHED instantly)\n"
                    f"  • current turn: {ws_resume.state.turn}\n\n"
                    f"User's continuation request: {prompt}\n\n"
                    f"FIRST STEP: call ListOpenQuestions to see what's open, "
                    f"then SearchFindings on the user's continuation request "
                    f"to check what's already known. THEN proceed with new "
                    f"research. DO NOT re-decompose if the existing "
                    f"sub-questions cover it; just pick one and dig in."
                )
                is_resume = True
            else:
                override = (config.get("_rabbit_hole_base_dir")
                            or os.environ.get("PROMETHEAN_RABBIT_HOLE_DIR"))
                base = _P(override) if override else (_P.home() / ".promethean" / "rabbit-hole")
                base.mkdir(parents=True, exist_ok=True)
                rabbit_hole_dir = str(base / task_id)
                RabbitHoleWorkspace(rabbit_hole_dir, root_question=prompt)
            eff_config["_rabbit_hole_workspace_dir"] = rabbit_hole_dir
            eff_config["_rabbit_hole_task_id"] = task_id
            # Defaults that make rabbit-hole runs survive long horizons.
            eff_config["aggressive_elision"] = True
            eff_config["allowed_root"] = rabbit_hole_dir
            # Force-disable extended thinking. Qwen3.5's reasoning mode
            # burns hundreds-to-thousands of tokens on <think>...</think>
            # blocks INSTEAD of emitting tool calls. Observed empirically:
            # 927 output tokens, zero tool calls, run produced no findings.
            # The rabbit-hole agent already plans via the sub-question
            # tree on disk — it doesn't need additional in-band reasoning.
            eff_config["thinking"] = False
            # Cap the per-turn output so the model can't ramble for
            # thousands of tokens before a tool call. Tool calls are short;
            # 1500 tokens is generous.
            eff_config["max_tokens"] = min(
                eff_config.get("max_tokens", 4000), 1500)
            # Slot-aware context budget. The local llama-server config in
            # run.sh uses -c 229376 -np 4 → each slot only gets 57344
            # tokens, but get_context_limit returns 128000 (provider
            # default for "custom"). Without an explicit override, the
            # 50% compaction threshold becomes ~64K, which is past the
            # slot's actual capacity → the model hits a 400 error from
            # llama-server before compaction fires. Cap at 50K
            # conservatively (87% of slot capacity, leaves headroom for
            # the response).
            # Slot-aware context budget. With np=1 (the default), each
            # slot has the full -c capacity (229376 in our run.sh). Set
            # the harness's compaction threshold to trigger before that
            # wall.
            if not eff_config.get("context_limit"):
                eff_config["context_limit"] = 200000  # ~87% of 229376
            task.rabbit_hole_dir = rabbit_hole_dir

        # Parent-slot preservation across the rabbit-hole run. With
        # np=1 the parent and subagent share slot 0; without coordination
        # the subagent's first chat completion erases the parent's
        # cached KV and forces a full re-prefill on the parent's next
        # turn (very expensive on long sessions). To avoid that we save
        # the parent's KV to disk before the subagent claims the slot,
        # and restore it on subagent exit. Server-side: requires
        # --slot-save-path (run.sh sets it by default).
        #
        # Conditions:
        #   • we're spawning a rabbit-hole (only kind worth this dance)
        #   • config["auto_slot_swap"] != False (default True)
        #   • we can talk to /slots
        # Failure modes are best-effort: if save fails the run still
        # proceeds, parent just pays the re-prefill cost.
        parent_kv_filename: str = ""
        parent_slot_id: int = 0  # np=1 → only slot 0 to consider
        slot_admin_url: str = ""  # always-defined for finally-block closure
        if rabbit_hole_dir and eff_config.get("auto_slot_swap", True):
            base_url = (eff_config.get("custom_base_url") or "").rstrip("/")
            if base_url:
                slot_admin_url = base_url[:-3] if base_url.endswith("/v1") else base_url
                from providers import detect_provider as _dp
                if _dp(eff_config.get("model", "")) == "custom":
                    try:
                        from llama_slots import LlamaSlotsError, save_slot, erase_slot
                        parent_kv_filename = f"parent-{config.get('_session_id', 'default')}.bin"
                        save_slot(parent_slot_id, parent_kv_filename,
                                  server_url=slot_admin_url)
                        erase_slot(parent_slot_id, server_url=slot_admin_url)
                        # Stash for the finally block. Reuse llama_slots
                        # admin URL so we don't recompute.
                        eff_config["_parent_kv_filename"] = parent_kv_filename
                        eff_config["_parent_slot_admin_url"] = slot_admin_url
                        eff_config["_parent_slot_id"] = parent_slot_id
                    except LlamaSlotsError:
                        # Server didn't support save (no --slot-save-path,
                        # endpoints disabled). Fall through silently —
                        # subagent runs as if save/restore weren't there.
                        parent_kv_filename = ""
                    except Exception:
                        parent_kv_filename = ""

        # Slot paging — pin this subagent to a specific llama-server slot
        # so its KV doesn't compete with the parent's for prefix-cache
        # space, and so we can deterministically erase it on finish.
        # Only fires when (a) opted in via enable_slot_paging,
        # (b) the provider is custom (i.e. our local llama-server),
        # (c) the server actually has an idle slot to give us.
        allocated_slot: Optional[int] = None
        slot_server_url: str = ""
        if eff_config.get("enable_slot_paging"):
            base_url = (eff_config.get("custom_base_url") or "").rstrip("/")
            if base_url:
                # /v1 → strip; we want the bare server URL for /slots admin.
                if base_url.endswith("/v1"):
                    slot_server_url = base_url[:-3]
                else:
                    slot_server_url = base_url
                # Only attempt if model is on the custom provider.
                from providers import detect_provider
                if detect_provider(eff_config.get("model", "")) == "custom":
                    allocated_slot = self._allocate_slot(slot_server_url)
                    if allocated_slot is not None:
                        eff_config["_slot_id"] = allocated_slot
                        task.slot_id = allocated_slot

        # Handle worktree isolation
        worktree_path = ""
        worktree_branch = ""
        base_dir = os.getcwd()

        if isolation == "worktree":
            git_root = _git_root(base_dir)
            if not git_root:
                task.status = "failed"
                task.result = "isolation='worktree' requires a git repository"
                return task
            try:
                worktree_path, worktree_branch = _create_worktree(git_root)
                task.worktree_path = worktree_path
                task.worktree_branch = worktree_branch
                notice = (
                    f"\n\n[Note: You are working in an isolated git worktree at "
                    f"{worktree_path} (branch: {worktree_branch}). "
                    f"Your changes are isolated from the main workspace at {git_root}. "
                    f"Commit your changes before finishing so they can be reviewed/merged.]"
                )
                prompt = prompt + notice
                # Pass the worktree path through config so tools (Bash/Glob/Grep)
                # use it as their working directory without touching the process-level
                # cwd (which is shared across all threads).
                eff_config["_worktree_cwd"] = worktree_path
            except Exception as e:
                task.status = "failed"
                task.result = f"Failed to create worktree: {e}"
                return task

        # Background mode: agent stays alive after the initial run and
        # blocks on its inbox so SendMessage can deliver new prompts
        # over its lifetime. Slot is parked during idle periods if slot
        # paging is enabled. Set on the rabbit-hole agent definition.
        is_background = bool(agent_def and agent_def.background)

        # Build a cancel-check that observes BOTH task._cancel_flag AND
        # the rabbit-hole workspace's manifest status. Without observing
        # the workspace status, calling the Finish tool only flips a
        # flag that the BACKGROUND loop reads — but the inner agent
        # turn loop keeps spinning, calling tools indefinitely. We saw
        # the model emit Finish() ~7+ times in a row before this fix.
        #
        # Closure variables captured: task, rabbit_hole_dir.
        def _make_cancel_check():
            def _check() -> bool:
                if task._cancel_flag:
                    return True
                if rabbit_hole_dir:
                    try:
                        from rabbit_hole.store import RabbitHoleWorkspace
                        ws_chk = RabbitHoleWorkspace(rabbit_hole_dir)
                        if ws_chk.manifest.get("status") == "finished":
                            return True
                    except Exception:
                        pass
                return False
            return _check
        _cancel_check = _make_cancel_check()

        def _run():
            import agent as _agent_mod; AgentState = _agent_mod.AgentState
            task.status = "running"
            try:
                state = AgentState()
                gen = _agent_run(
                    prompt, state, eff_config, eff_system,
                    depth=depth + 1,
                    cancel_check=lambda: task._cancel_flag,
                )
                for _event in gen:
                    if rabbit_hole_dir:
                        _capture_agent_event_to_workspace(_event, rabbit_hole_dir)
                    if task._cancel_flag:
                        break

                if task._cancel_flag:
                    task.status = "cancelled"
                    task.result = None
                else:
                    task.result = _extract_final_text(state.messages)
                    task.status = "completed"

                if is_background:
                    self._background_loop(
                        task, state, eff_config, eff_system, depth + 1,
                        rabbit_hole_dir,
                        allocated_slot, slot_server_url,
                    )
                else:
                    # Foreground (legacy) — drain any inbox messages
                    # that landed during the main run and then exit.
                    while not task._inbox.empty() and not task._cancel_flag:
                        inbox_msg = task._inbox.get_nowait()
                        task.status = "running"
                        gen2 = _agent_run(
                            inbox_msg, state, eff_config, eff_system,
                            depth=depth + 1,
                            cancel_check=lambda: task._cancel_flag,
                        )
                        for _ev in gen2:
                            if rabbit_hole_dir:
                                _capture_agent_event_to_workspace(_ev, rabbit_hole_dir)
                            if task._cancel_flag:
                                break
                        if not task._cancel_flag:
                            task.result = _extract_final_text(state.messages)
                            task.status = "completed"

            except Exception as e:
                task.status = "failed"
                task.result = f"Error: {e}"
            finally:
                if worktree_path:
                    _remove_worktree(worktree_path, worktree_branch, base_dir)
                # Rabbit-hole mode: synthesize the final report from the
                # workspace state, regardless of whether the agent finished
                # normally, was cancelled, or crashed. The on-disk findings
                # are the ground truth — the agent's in-context summary is
                # incidental.
                if rabbit_hole_dir:
                    try:
                        from rabbit_hole.store import RabbitHoleWorkspace
                        from rabbit_hole.synthesis import synthesize_workspace
                        ws_final = RabbitHoleWorkspace(rabbit_hole_dir)
                        report = synthesize_workspace(
                            ws_final, final=True,
                            cancelled=task.status == "cancelled",
                        )
                        # Surface the report path to the parent via task.result;
                        # full text would be too long.
                        report_path = ws_final.reports_dir / "final.md"
                        prefix = task.result or ""
                        suffix = (f"\n\n[Rabbit-hole report saved to "
                                  f"{report_path}; {len(report)} chars, "
                                  f"{len(ws_final.list_findings())} findings, "
                                  f"{len(ws_final.list_sources())} sources.]")
                        task.result = prefix + suffix
                    except Exception as e:
                        # Synthesis is best-effort — don't fail the task on
                        # a synthesis error. The workspace files remain.
                        task.result = (task.result or "") + (
                            f"\n\n[Rabbit-hole synthesis failed: {e}; "
                            f"workspace files at {rabbit_hole_dir}]"
                        )
                # Release the pinned slot — erases its KV server-side and
                # removes from the in-memory allocated-set so the next
                # subagent can claim it. Best-effort: server may have
                # become unreachable, in which case erase silently fails.
                if allocated_slot is not None:
                    self._release_slot(allocated_slot, slot_server_url)
                # Parent-slot RESTORE — counterpart to the save+erase that
                # ran in spawn() before the subagent started. Erase the
                # subagent's KV from slot 0, then restore the parent's
                # cached KV so the parent's next REPL turn finds its
                # prefix and skips re-prefill. Best-effort.
                if parent_kv_filename:
                    try:
                        from llama_slots import LlamaSlotsError, restore_slot, erase_slot
                        erase_slot(parent_slot_id, server_url=slot_admin_url)
                        restore_slot(parent_slot_id, parent_kv_filename,
                                     server_url=slot_admin_url)
                    except LlamaSlotsError:
                        # If restore fails, parent loses its cached KV
                        # and pays a re-prefill on next turn. Annoying
                        # but not fatal — the message history still has
                        # the conversation, so the model has all the
                        # context, just needs to recompute attention
                        # over it.
                        pass
                    except Exception:
                        pass

        task._future = self._pool.submit(_run)
        return task

    def wait(self, task_id: str, timeout: float = None) -> Optional[SubAgentTask]:
        """Block until a task completes or timeout expires.

        Returns:
            The task, or None if task_id is unknown.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None
        if task._future is not None:
            try:
                task._future.result(timeout=timeout)
            except Exception:
                pass
        return task

    def get_result(self, task_id: str) -> Optional[str]:
        """Return the result string for a completed task, or None."""
        task = self.tasks.get(task_id)
        return task.result if task else None

    def list_tasks(self) -> List[SubAgentTask]:
        """Return all tracked tasks."""
        return list(self.tasks.values())

    def send_message(self, task_id_or_name: str, message: str) -> bool:
        """Send a message to a running or idle background agent.

        The message is queued; the agent will process it after completing
        any current work, or — for background-idle agents — when its
        next inbox.get() poll fires (within ~1 s).

        Args:
            task_id_or_name: task ID or the human-readable name passed to spawn()
            message:         message text to send

        Returns:
            True if the message was queued, False if task not found or
            already in a terminal state.
        """
        # Resolve name → task_id
        task_id = self._by_name.get(task_id_or_name, task_id_or_name)
        task = self.tasks.get(task_id)
        if task is None:
            return False
        # Alive states: pending (not yet started), running (mid-turn),
        # idle (background subagent waiting on inbox).
        if task.status not in ("running", "pending", "idle"):
            return False
        task._inbox.put(message)
        return True

    def cancel(self, task_id: str) -> bool:
        """Request cancellation of a running or idle task.

        Returns:
            True if the cancel flag was set, False if task not found or
            already in a terminal state (completed/failed/cancelled).
        """
        task = self.tasks.get(task_id)
        if task is None:
            return False
        # "running" and "idle" are both alive states. Idle = background
        # subagent waiting on inbox; the loop polls _cancel_flag at every
        # inbox.get() timeout (1 s), so cancel propagates within ~1 s.
        if task.status in ("running", "idle"):
            task._cancel_flag = True
            return True
        return False

    def shutdown(self) -> None:
        """Cancel all alive tasks (running or background-idle) and shut
        down the thread pool. Background subagents poll _cancel_flag at
        every inbox.get() timeout, so they exit within ~1 s of shutdown
        being called."""
        for task in self.tasks.values():
            if task.status in ("running", "idle"):
                task._cancel_flag = True
        self._pool.shutdown(wait=True)
