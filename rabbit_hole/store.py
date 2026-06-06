"""RabbitHoleWorkspace — disk-backed state for a long-running research agent.

Three concepts live here:

  • Sources    — fetched URL content, deduped by URL hash. The agent calls
                 RabbitFetch which checks `has_source(url)` first.
  • Findings   — structured claim + evidence-URL pairs the agent saves
                 explicitly via SaveFinding. The unit of synthesis later.
  • Questions  — sub-question tree the agent maintains. Each finding ties
                 to one sub-question.

State is also tracked: per-turn counters of "real progress" actions
(new source fetched, new finding saved, sub-question closed). The
agent loop wrapper reads `stuck_for()` to decide when to inject a
nudge or force termination.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


# ── URL normalization for dedup ─────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Stable URL form for dedup. Lowercases the host, drops fragments,
    sorts query parameters. Preserves path case (case-sensitive on most
    servers). Does NOT strip tracking params (utm_*) — that's a policy
    decision; we keep them so different campaigns to the same content
    are still treated as distinct sources unless explicitly identical."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    qsl = parse_qsl(parsed.query, keep_blank_values=True)
    qsl.sort()
    return urlunparse((
        parsed.scheme.lower() or "https",
        host,
        parsed.path.rstrip("/") or "/",
        parsed.params,
        urlencode(qsl),
        "",  # drop fragment
    ))


def _url_key(url: str) -> str:
    """Stable SHA-256 key for an URL. 16 hex chars is plenty for our
    scale (millions of URLs to collide is roughly 1e-6 at 16 hex)."""
    return hashlib.sha256(_normalize_url(url).encode("utf-8")).hexdigest()[:16]


# ── Question / Finding records ──────────────────────────────────────────────

@dataclass
class Question:
    id: str
    text: str
    status: str = "open"           # "open" | "closed"
    parent_id: Optional[str] = None
    finding_ids: List[str] = field(default_factory=list)
    summary: str = ""              # filled at MarkQuestionDone
    created_turn: int = 0
    closed_turn: Optional[int] = None


@dataclass
class Finding:
    id: str
    claim: str
    sub_question_id: str
    evidence_urls: List[str]
    source_shas: List[str]
    created_turn: int = 0


@dataclass
class State:
    turn: int = 0
    last_new_source_turn: int = 0
    last_new_finding_turn: int = 0
    last_question_close_turn: int = 0


# ── Workspace ──────────────────────────────────────────────────────────────

class RabbitHoleWorkspace:
    """Disk-backed workspace for a long-running research subagent.

    All operations are idempotent and crash-safe in the sense that the
    workspace can be re-loaded from disk after a crash without losing
    any saved finding or fetched source. The in-memory caches are
    rebuilt from disk on __init__.
    """

    def __init__(self, root_dir: str, root_question: str = ""):
        self.root = Path(root_dir).resolve()
        self.sources_dir   = self.root / "sources"
        self.findings_dir  = self.root / "findings"
        self.notes_dir     = self.root / "notes"
        self.reports_dir   = self.root / "reports"
        for d in (self.sources_dir, self.findings_dir, self.notes_dir,
                  self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.manifest_path  = self.root / "manifest.json"
        self.questions_path = self.root / "questions.json"
        self.state_path     = self.root / "state.json"
        self.progress_path  = self.root / "progress.jsonl"

        # Manifest: write once on first init, read after.
        if not self.manifest_path.exists():
            self.manifest = {
                "created_at": time.time(),
                "root_question": root_question,
                "status": "active",
            }
            self.manifest_path.write_text(json.dumps(self.manifest, indent=2))
        else:
            self.manifest = json.loads(self.manifest_path.read_text())

        # In-memory caches (rebuilt from disk if files exist).
        self.questions: Dict[str, Question] = {}
        self.state: State = State()
        self._load_questions()
        self._load_state()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_questions(self) -> None:
        if self.questions_path.exists():
            data = json.loads(self.questions_path.read_text())
            for qid, q in data.items():
                self.questions[qid] = Question(**q)

    def _save_questions(self) -> None:
        out = {qid: asdict(q) for qid, q in self.questions.items()}
        self.questions_path.write_text(json.dumps(out, indent=2))

    def _load_state(self) -> None:
        if self.state_path.exists():
            self.state = State(**json.loads(self.state_path.read_text()))

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2))

    # ── Progress event log ─────────────────────────────────────────────────

    def append_event(self, kind: str, **payload) -> None:
        """Append a structured progress event to progress.jsonl.

        Events are auto-emitted by the mutator methods (add_question,
        add_source, add_finding, mark_question_done) and explicitly by
        the Note tool. The /rabbit-hole status slash command shows the
        last N events chronologically — the user's window into what the
        agent has been doing.

        Format: one JSON line per event:
          {"turn": 12, "ts": 1748293847.123,
           "kind": "source_fetched",
           "payload": {"url": "https://...", "title": "..."}}
        """
        event = {
            "turn": self.state.turn,
            "ts": time.time(),
            "kind": kind,
            "payload": payload,
        }
        # Best-effort: progress logging must never fail the parent op.
        try:
            with self.progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            pass

    def list_events(self, last_n: Optional[int] = None) -> List[dict]:
        """Read events from progress.jsonl. If last_n is given, return only
        the most recent N. Order is chronological (oldest first)."""
        if not self.progress_path.exists():
            return []
        events: List[dict] = []
        try:
            with self.progress_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        if last_n is not None:
            events = events[-last_n:]
        return events

    # ── Turn ───────────────────────────────────────────────────────────────

    def advance_turn(self) -> int:
        self.state.turn += 1
        self._save_state()
        return self.state.turn

    def stuck_for(self) -> int:
        """Number of turns since the last 'real progress' event.
        Real progress = new source, new finding, or closed question."""
        last_progress = max(
            self.state.last_new_source_turn,
            self.state.last_new_finding_turn,
            self.state.last_question_close_turn,
        )
        return max(0, self.state.turn - last_progress)

    # ── Sources ────────────────────────────────────────────────────────────

    def has_source(self, url: str) -> bool:
        return (self.sources_dir / f"{_url_key(url)}.json").exists()

    def get_cached_source(self, url: str) -> Optional[dict]:
        """Return the cached source dict (with full content) if present."""
        meta_path = self.sources_dir / f"{_url_key(url)}.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        content_path = self.sources_dir / f"{_url_key(url)}.txt"
        if content_path.exists():
            meta["content"] = content_path.read_text(
                encoding="utf-8", errors="replace")
        return meta

    def add_source(self, url: str, content: str, title: str = "") -> str:
        """Store a fetched source. Returns the SHA key."""
        key = _url_key(url)
        meta = {
            "url": url,
            "normalized_url": _normalize_url(url),
            "title": title or url,
            "fetched_at": time.time(),
            "n_chars": len(content),
            "fetched_at_turn": self.state.turn,
        }
        (self.sources_dir / f"{key}.json").write_text(
            json.dumps(meta, indent=2))
        # Cap content at 1 MB on disk — very long pages (HTML dumps,
        # PDFs converted to text) are noise beyond that for our purposes.
        (self.sources_dir / f"{key}.txt").write_text(
            content[:1_000_000], encoding="utf-8", errors="replace")
        self.state.last_new_source_turn = self.state.turn
        self._save_state()
        self.append_event("source_fetched",
                          url=url, title=title or url, sha=key, n_chars=len(content))
        return key

    def list_sources(self) -> List[dict]:
        out = []
        for meta_path in sorted(self.sources_dir.glob("*.json")):
            try:
                out.append(json.loads(meta_path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    # ── Findings ───────────────────────────────────────────────────────────

    def add_finding(
        self, claim: str, sub_question_id: str,
        evidence_urls: List[str],
    ) -> str:
        fid = f"f-{uuid.uuid4().hex[:8]}"
        source_shas = [_url_key(u) for u in evidence_urls]
        finding = Finding(
            id=fid, claim=claim, sub_question_id=sub_question_id,
            evidence_urls=evidence_urls, source_shas=source_shas,
            created_turn=self.state.turn,
        )
        (self.findings_dir / f"{fid}.json").write_text(
            json.dumps(asdict(finding), indent=2))
        # Backref into the sub-question record.
        if sub_question_id in self.questions:
            self.questions[sub_question_id].finding_ids.append(fid)
            self._save_questions()
        self.state.last_new_finding_turn = self.state.turn
        self._save_state()
        self.append_event("finding_saved",
                          finding_id=fid, sub_question_id=sub_question_id,
                          claim=claim, n_evidence=len(evidence_urls))
        return fid

    def list_findings(
        self, sub_question_id: Optional[str] = None,
    ) -> List[Finding]:
        out: List[Finding] = []
        for fp in sorted(self.findings_dir.glob("f-*.json")):
            try:
                data = json.loads(fp.read_text())
                f = Finding(**data)
            except (json.JSONDecodeError, OSError, TypeError):
                continue
            if sub_question_id is None or f.sub_question_id == sub_question_id:
                out.append(f)
        return out

    # ── Questions ──────────────────────────────────────────────────────────

    def add_question(self, text: str, parent_id: Optional[str] = None) -> str:
        qid = f"q-{uuid.uuid4().hex[:8]}"
        q = Question(
            id=qid, text=text, parent_id=parent_id,
            created_turn=self.state.turn,
        )
        self.questions[qid] = q
        self._save_questions()
        self.append_event("question_added",
                          question_id=qid, text=text, parent_id=parent_id)
        return qid

    def list_open_questions(self) -> List[Question]:
        return [q for q in self.questions.values() if q.status == "open"]

    def list_all_questions(self) -> List[Question]:
        return list(self.questions.values())

    def mark_question_done(self, qid: str, summary: str = "") -> bool:
        q = self.questions.get(qid)
        if q is None or q.status == "closed":
            return False
        q.status = "closed"
        q.summary = summary
        q.closed_turn = self.state.turn
        self._save_questions()
        self.state.last_question_close_turn = self.state.turn
        self._save_state()
        self.append_event("question_closed",
                          question_id=qid, text=q.text, summary=summary,
                          n_findings=len(q.finding_ids))
        return True

    # ── Notes ──────────────────────────────────────────────────────────────

    def append_note(self, text: str) -> None:
        """Append a Think output to today's notes file. One file per turn
        so synthesis can grep them efficiently."""
        path = self.notes_dir / f"turn-{self.state.turn:05d}.md"
        with path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")
