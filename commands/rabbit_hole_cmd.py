"""/rabbit-hole — convenience slash command for the deep-research-rabbit-hole subagent.

Subcommands:
  /rabbit-hole <question>       Spawn a new rabbit-hole run (background mode).
  /rabbit-hole list             List active rabbit-hole tasks.
  /rabbit-hole status <name>    Show progress (turn, finding count, sub-questions).
  /rabbit-hole msg <name> <m>   Send a follow-up message to a running task.
  /rabbit-hole stop <name|all>  Kill task(s); synthesis runs on exit.
  /rabbit-hole report <name>    Print the path of the latest synthesis report.

The rabbit-hole agent runs in background mode — after spawn it stays alive
waiting for SendMessage. Synthesis runs automatically on cancel, on Finish(),
or on agent crash. The on-disk workspace at ~/.promethean/rabbit-hole/<id>/
is the durable record; the slash command is just a convenience wrapper over
the SubAgentManager.

Internally a thin layer over multi_agent.tools._agent_tool — same code path
the model uses when it calls Agent(subagent_type="deep-research-rabbit-hole").
"""
from __future__ import annotations

from pathlib import Path

from ui.render import clr, err, info, ok, warn


def _help() -> None:
    info("Usage:")
    info("  /rabbit-hole <question>            spawn a new rabbit-hole run")
    info("  /rabbit-hole list                  list active rabbit-holes")
    info("  /rabbit-hole list --all            list ALL workspaces on disk (incl. paused)")
    info("  /rabbit-hole status <name>         show progress for a task")
    info("  /rabbit-hole msg <name> <message>  send a follow-up to a running task")
    info("  /rabbit-hole stop <name|all>       kill task(s) — TRIGGERS SYNTHESIS")
    info("  /rabbit-hole report <name>         print latest synthesis report path")
    info("  /rabbit-hole resume <id> [hint]    resume a paused/cancelled workspace")
    info("")
    info("Aliases: /rh, /rabbithole")
    info("")
    info("How a run ends:")
    info("  Rabbit-hole agents CANNOT stop themselves — they're force-continued")
    info("  every 2s of idle. To end a run, you /rabbit-hole stop <name>. The")
    info("  finally-block runs synthesize_workspace() automatically and writes")
    info("  reports/final.md. View with /rabbit-hole report <name>.")
    info("  To pick up later: /rabbit-hole resume <id>.")


def _get_rabbit_tasks(manager) -> list:
    """Return only the manager's tasks that are rabbit-hole runs."""
    return [t for t in manager.list_tasks() if getattr(t, "rabbit_hole_dir", "")]


def _resolve_task(manager, target: str):
    """Look up by name first, then by id-prefix. Restricted to rabbit-hole tasks
    so a name collision with a non-rabbit-hole agent doesn't surprise us."""
    rabbit_tasks = _get_rabbit_tasks(manager)
    by_name = {t.name: t for t in rabbit_tasks}
    if target in by_name:
        return by_name[target]
    # Fall back to id-prefix match
    matches = [t for t in rabbit_tasks if t.id.startswith(target)]
    if len(matches) == 1:
        return matches[0]
    return None


def _spawn(question: str, state, config) -> bool:
    from multi_agent.tools import get_agent_manager
    from multi_agent.subagent import get_agent_definition

    agent_def = get_agent_definition("deep-research-rabbit-hole")
    if agent_def is None:
        err("Built-in agent 'deep-research-rabbit-hole' is not registered. "
            "Check that multi_agent.subagent imports cleanly.")
        return True

    mgr = get_agent_manager()
    # Strip private keys before passing to sub-agent (matches what the
    # JSON Agent tool does in multi_agent.tools._agent_tool).
    eff_config = {k: v for k, v in config.items() if not k.startswith("_")}
    system_prompt = config.get("_system_prompt", "You are a helpful assistant.")
    depth = config.get("_depth", 0)

    task = mgr.spawn(
        question, eff_config, system_prompt,
        depth=depth, agent_def=agent_def,
    )

    if task.status == "failed":
        err(f"Failed to spawn rabbit-hole: {task.result}")
        return True

    ok(f"Rabbit-hole spawned: {clr(task.name, 'cyan')}  (id: {task.id})")
    info(f"Workspace: {task.rabbit_hole_dir}")
    info("Background mode — agent will stay alive until you stop it or it calls Finish.")
    info("")
    info(f"  {clr('/rabbit-hole status', 'dim')} {task.name}    show progress")
    info(f"  {clr('/rabbit-hole msg',    'dim')} {task.name} <message>    nudge mid-run")
    info(f"  {clr('/rabbit-hole stop',   'dim')} {task.name}    kill (synthesis runs on exit)")
    return True


def _rabbit_hole_base_dir() -> Path:
    """Resolve the workspace base directory the same way SubAgentManager
    does — PROMETHEAN_RABBIT_HOLE_DIR env var first, then ~/.promethean/."""
    import os as _os
    override = _os.environ.get("PROMETHEAN_RABBIT_HOLE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".promethean" / "rabbit-hole"


def _cmd_list(state, config, show_all: bool = False) -> bool:
    from multi_agent.tools import get_agent_manager
    mgr = get_agent_manager()
    tasks = _get_rabbit_tasks(mgr)

    if show_all:
        # Enumerate disk workspaces too (including finished/abandoned)
        base = _rabbit_hole_base_dir()
        in_memory_dirs = {t.rabbit_hole_dir for t in tasks}
        disk_workspaces = []
        if base.exists():
            for d in sorted(base.iterdir(), key=lambda p: -p.stat().st_mtime if p.is_dir() else 0):
                if not d.is_dir():
                    continue
                if str(d) in in_memory_dirs:
                    continue
                manifest_path = d / "manifest.json"
                if not manifest_path.exists():
                    continue
                disk_workspaces.append(d)

        if not tasks and not disk_workspaces:
            info("No rabbit-hole workspaces (in memory or on disk).")
            return True
        if tasks:
            ok(f"{len(tasks)} active rabbit-hole task(s):")
            for t in tasks:
                color = {"running": "green", "idle": "yellow",
                         "completed": "blue", "cancelled": "red",
                         "failed": "red"}.get(t.status, "white")
                print(f"  {clr('●', color)} {t.name:24s}  "
                      f"{clr(t.status, color):12s}  {clr(t.id, 'dim')}")
        if disk_workspaces:
            print()
            info(f"{len(disk_workspaces)} on-disk workspace(s) (resumable):")
            import json as _json
            for d in disk_workspaces[:30]:
                manifest = {}
                try:
                    manifest = _json.loads((d / "manifest.json").read_text())
                except Exception:
                    pass
                status = manifest.get("status", "?")
                rq = manifest.get("root_question", "(no question)")[:60]
                resumed = manifest.get("resumed_at")
                resume_marker = f" [resumed {len(resumed)}×]" if isinstance(resumed, list) and resumed else ""
                # Quick stats
                n_findings = len(list((d / "findings").glob("*.json"))) if (d / "findings").exists() else 0
                n_sources = len(list((d / "sources").glob("*.json"))) if (d / "sources").exists() else 0
                print(f"  {clr('○', 'dim')} {d.name:14s}  "
                      f"{clr(status, 'dim'):10s}  "
                      f"f={n_findings:>3} s={n_sources:>3}  "
                      f"{rq}{resume_marker}")
            if len(disk_workspaces) > 30:
                print(f"  ... and {len(disk_workspaces) - 30} more")
        return True

    if not tasks:
        info("No rabbit-hole tasks running. Use /rabbit-hole list --all to see "
             "on-disk workspaces.")
        return True
    ok(f"{len(tasks)} rabbit-hole task(s):")
    for t in tasks:
        color = {"running": "green", "idle": "yellow",
                 "completed": "blue", "cancelled": "red",
                 "failed": "red"}.get(t.status, "white")
        print(f"  {clr('●', color)} {t.name:24s}  {clr(t.status, color):12s}  "
              f"{clr(t.id, 'dim')}")
    return True


def _resolve_workspace_dir(target: str) -> Optional[Path]:
    """Resolve target to an on-disk workspace directory. Accepts:
      • full task id (e.g. d8f423e39131)
      • prefix of task id (e.g. d8f4)
      • full path to workspace dir
    Returns None if no unique match.
    """
    p = Path(target)
    if p.is_dir() and (p / "manifest.json").exists():
        return p.resolve()
    base = _rabbit_hole_base_dir()
    if not base.exists():
        return None
    matches = [d for d in base.iterdir()
               if d.is_dir() and d.name.startswith(target)
               and (d / "manifest.json").exists()]
    if len(matches) == 1:
        return matches[0].resolve()
    return None


def _cmd_resume(target: str, hint: str, state, config) -> bool:
    """Resume a previously-paused or finished rabbit-hole workspace.
    Spawns a NEW SubAgentTask but reusing the existing workspace dir,
    so all findings / sources / sub-questions carry over."""
    from multi_agent.tools import get_agent_manager
    from multi_agent.subagent import get_agent_definition
    import json as _json

    ws_dir = _resolve_workspace_dir(target)
    if ws_dir is None:
        err(f"No workspace matching {target!r}. "
            f"Use /rabbit-hole list --all to see resumable workspaces.")
        return True

    manifest_path = ws_dir / "manifest.json"
    try:
        manifest = _json.loads(manifest_path.read_text())
    except Exception as e:
        err(f"Couldn't read manifest at {manifest_path}: {e}")
        return True

    rq = manifest.get("root_question", "(unknown)")
    info(f"Resuming workspace: {clr(ws_dir.name, 'cyan')}")
    info(f"  Original question: {rq[:120]}")
    if manifest.get("status") == "finished":
        warn(f"  Prior run had called Finish — resuming will reopen it.")
    info(f"  Workspace path: {ws_dir}")

    agent_def = get_agent_definition("deep-research-rabbit-hole")
    if agent_def is None:
        err("deep-research-rabbit-hole agent not registered.")
        return True

    mgr = get_agent_manager()
    eff_config = {k: v for k, v in config.items() if not k.startswith("_")}
    system_prompt = config.get("_system_prompt", "You are a helpful assistant.")
    depth = config.get("_depth", 0)
    user_hint = hint.strip() if hint else "Continue the research from where the prior run stopped."

    task = mgr.spawn(
        user_hint, eff_config, system_prompt,
        depth=depth, agent_def=agent_def,
        resume_workspace_dir=str(ws_dir),
    )
    if task.status == "failed":
        err(f"Resume failed: {task.result}")
        return True

    ok(f"Resumed as task: {clr(task.name, 'cyan')}  (id: {task.id})")
    info(f"Continuation hint: {user_hint[:120]}")
    info("")
    info(f"  /rabbit-hole status {task.name}    show progress")
    info(f"  /rabbit-hole stop   {task.name}    pause again (state preserved)")
    return True


def _format_event(ev: dict, base_ts: float) -> str:
    """Render one progress event as a single line for `/rabbit-hole status`.

    Format: `[+MM:SS turn N] kind: short summary`. Relative time is
    measured from the workspace's first event so a long-running rabbit-
    hole reads like a chronological diary.
    """
    delta = max(0.0, ev.get("ts", base_ts) - base_ts)
    mm = int(delta // 60)
    ss = int(delta % 60)
    rel_ts = f"+{mm:>2d}:{ss:02d}"
    kind = ev.get("kind", "?")
    p = ev.get("payload", {}) or {}
    if kind == "note":
        body = p.get("text", "")
        marker = clr("📝 note", "yellow")
    elif kind == "question_added":
        body = p.get("text", "")[:120]
        marker = clr("+ question", "cyan")
        if p.get("parent_id"):
            body = f"(child of {p['parent_id']}) {body}"
    elif kind == "question_closed":
        summary = p.get("summary", "(no summary)")[:120]
        body = f"{p.get('text', '?')[:60]} — {summary}"
        marker = clr("✓ closed", "green")
    elif kind == "source_fetched":
        body = f"{p.get('title', '?')[:80]} ({p.get('n_chars', 0)} chars)"
        marker = clr("⤓ fetched", "blue")
    elif kind == "finding_saved":
        body = p.get("claim", "")[:120]
        marker = clr("◇ finding", "magenta")
    elif kind == "error":
        body = p.get("message", "?")
        marker = clr("⚠ error", "red")
    else:
        body = str(p)[:120]
        marker = clr(kind, "white")
    turn = ev.get("turn", "?")
    return f"  {clr(rel_ts, 'dim')}  turn {turn:>3}  {marker}  {body}"


def _cmd_status(target: str, state, config) -> bool:
    from multi_agent.tools import get_agent_manager
    from rabbit_hole.store import RabbitHoleWorkspace
    mgr = get_agent_manager()
    task = _resolve_task(mgr, target)
    if task is None:
        err(f"No rabbit-hole task matching {target!r}.")
        return True
    info(f"Task: {clr(task.name, 'cyan')}  (id: {task.id})")
    info(f"Status: {task.status}")
    info(f"Workspace: {task.rabbit_hole_dir}")
    if task.slot_id is not None:
        parked = " [parked]" if task.slot_parked else ""
        info(f"Slot: {task.slot_id}{parked}")
    try:
        ws = RabbitHoleWorkspace(task.rabbit_hole_dir)
    except Exception as e:
        err(f"Couldn't open workspace: {e}")
        return True
    open_qs = ws.list_open_questions()
    closed_qs = [q for q in ws.list_all_questions() if q.status == "closed"]
    findings = ws.list_findings()
    sources = ws.list_sources()
    info(f"Turn: {ws.state.turn}    stuck_for: {ws.stuck_for()}")
    info(f"Sub-questions: {len(open_qs)} open, {len(closed_qs)} closed")
    info(f"Findings: {len(findings)} total")
    info(f"Sources fetched: {len(sources)} unique")

    # Recent activity feed — chronological progress events the user can
    # follow live without disrupting the run.
    events = ws.list_events(last_n=15)
    if events:
        info("")
        info(f"Recent activity (last {len(events)} events):")
        base_ts = events[0].get("ts", 0.0)
        for ev in events:
            print(_format_event(ev, base_ts))

    if open_qs:
        info("")
        info("Currently open:")
        for q in open_qs[:5]:
            info(f"  {q.id}: {q.text[:100]}")
        if len(open_qs) > 5:
            info(f"  ... and {len(open_qs) - 5} more")
    return True


def _cmd_msg(target: str, message: str, state, config) -> bool:
    from multi_agent.tools import get_agent_manager
    mgr = get_agent_manager()
    task = _resolve_task(mgr, target)
    if task is None:
        err(f"No rabbit-hole task matching {target!r}.")
        return True
    if task.status not in ("running", "idle"):
        warn(f"Task {task.name} is in state {task.status} — message will not be processed.")
        return True
    if not message.strip():
        err("Empty message.")
        return True
    if mgr.send_message(task.id, message):
        ok(f"Message queued for {task.name}.")
    else:
        err(f"Failed to queue message for {task.name}.")
    return True


def _cmd_stop(target: str, state, config) -> bool:
    from multi_agent.tools import get_agent_manager
    mgr = get_agent_manager()
    if target.lower() == "all":
        tasks = _get_rabbit_tasks(mgr)
        n = 0
        for t in tasks:
            if mgr.cancel(t.id):
                n += 1
        ok(f"Cancelled {n} rabbit-hole task(s). Synthesis will run as each exits.")
        return True
    task = _resolve_task(mgr, target)
    if task is None:
        err(f"No rabbit-hole task matching {target!r}.")
        return True
    if mgr.cancel(task.id):
        ok(f"Cancelled {task.name}. Synthesis will run on exit.")
    else:
        warn(f"Task {task.name} is already in terminal state ({task.status}); nothing to cancel.")
    return True


def _cmd_report(target: str, state, config) -> bool:
    from multi_agent.tools import get_agent_manager
    mgr = get_agent_manager()
    task = _resolve_task(mgr, target)
    if task is None:
        err(f"No rabbit-hole task matching {target!r}.")
        return True
    if not task.rabbit_hole_dir:
        err("Task has no workspace recorded.")
        return True
    final = Path(task.rabbit_hole_dir) / "reports" / "final.md"
    if final.exists():
        ok(f"Final report: {final}")
        return True
    # Show intermediate if any
    intermediates = sorted(Path(task.rabbit_hole_dir).glob("reports/intermediate-*.md"))
    if intermediates:
        info(f"No final report yet (task status: {task.status}). "
             f"Latest intermediate: {intermediates[-1]}")
    else:
        info(f"No reports yet (task status: {task.status}). "
             f"Synthesis runs when the task exits or is cancelled.")
    return True


def cmd_rabbit_hole(args: str, state, config) -> bool:
    """Dispatch /rabbit-hole subcommands. See module docstring for syntax."""
    a = args.strip()
    if not a:
        _help()
        return True

    parts = a.split(None, 1)
    head = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    # Subcommands
    if head in ("list", "ls"):
        show_all = "--all" in rest.split() or rest.strip() in ("all", "-a")
        return _cmd_list(state, config, show_all=show_all)
    if head == "resume":
        if not rest:
            err("Usage: /rabbit-hole resume <id> [continuation-hint]")
            return True
        parts2 = rest.split(None, 1)
        target = parts2[0]
        hint = parts2[1] if len(parts2) > 1 else ""
        return _cmd_resume(target, hint, state, config)
    if head == "status":
        if not rest:
            err("Usage: /rabbit-hole status <name>")
            return True
        return _cmd_status(rest.strip(), state, config)
    if head == "msg":
        msg_parts = rest.split(None, 1) if rest else []
        if len(msg_parts) < 2:
            err("Usage: /rabbit-hole msg <name> <message>")
            return True
        return _cmd_msg(msg_parts[0], msg_parts[1], state, config)
    if head == "stop":
        if not rest:
            err("Usage: /rabbit-hole stop <name|all>")
            return True
        return _cmd_stop(rest.strip(), state, config)
    if head == "report":
        if not rest:
            err("Usage: /rabbit-hole report <name>")
            return True
        return _cmd_report(rest.strip(), state, config)
    if head in ("help", "-h", "--help"):
        _help()
        return True

    # Default: treat the whole arg as a research question and spawn.
    return _spawn(a, state, config)
