## Programmatic tool calling (preferred for multi-step file/code work)

You're running on a code-trained model. For tasks that touch multiple files,
require loops, or need to aggregate results, you'll get better results writing
ONE Python script and running it via `Bash`, instead of chaining many separate
Read/Grep/Glob/Edit calls.

### The agent_tools helpers — two ways to call them

The repo at `/home/alex/Escritorio/LLM/promethean/` ships an `agent_tools`
package. It uses tree-sitter + PageRank (Aider's repo-map algorithm) —
fully deterministic, no embeddings, no extra GPU/RAM. Eight helpers, each
exposed BOTH as a JSON tool you call directly AND as a Python function
you can import.

| JSON tool      | Python function   | What it does                                     |
|----------------|-------------------|--------------------------------------------------|
| `RepoMap`      | `repo_map`        | Ranked map of files + their key symbols          |
| `FindSymbol`   | `find_symbol`     | Where is `name` defined?                         |
| `GetCallers`   | `get_callers`     | Where is `name` referenced?                      |
| `Outline`      | `outline`         | List every def in one file                       |
| `Neighborhood` | `neighborhood`    | def + callers + callees of `name` in one shot    |
| `PathBetween`  | `path_between`    | BFS call chain from symbol `a` to symbol `b`     |
| `Imports`      | `imports`         | Forward + reverse deps of one file               |
| `SearchFiles`  | `search_files`    | TF-IDF keyword search over file contents         |

**Prefer the JSON tool form for one-shot queries.** It runs in-process,
needs no PYTHONPATH or subprocess, returns a pre-formatted text block.
Examples of one-shot queries: "who calls `validate_token`?", "show me a
path from handler to db", "find files mentioning 'rate limiting'".

**Use the Python import form for multi-step / aggregation** — looping over
many symbols, combining results, custom filtering. The functions return
plain `Hit` namedtuples / dicts you can post-process.

```python
# Python form — for batch work, combining helpers, custom filtering.
from agent_tools import (
    repo_map, find_symbol, get_callers, outline,
    neighborhood, path_between, imports, search_files,
)

# 1) Where is X defined? Where is it used? Side-by-side.
for hit in find_symbol("MyClass", root="/path/to/repo"):
    print(hit.file, hit.line)
for hit in get_callers("MyClass", root="/path/to/repo"):
    print(hit.file, hit.line)

# 2) One-shot symbol view.
nb = neighborhood("compute_total", root="/path/to/repo")
print("def:",     [h.file for h in nb["def"]])
print("callers:", [h.file for h in nb["callers"]])
print("callees:", [h.name for h in nb["callees"]])

# 3) Call chain across modules.
chain = path_between("handle_request", "query_db",
                     root="/path/to/repo", max_hops=4)
print(" -> ".join(h.name for h in chain))
```

### Running the Python form via Bash

The Python form needs the promethean dir on `PYTHONPATH`. Use the same
`python` interpreter that's already running this agent (it has the
required deps: `tree_sitter`, `diskcache`, `networkx`, `pygments`).
The system `python3` typically does NOT — don't reach for that one.

```
PYTHONPATH=/home/alex/Escritorio/LLM/promethean "$(which python)" -c '
from agent_tools import neighborhood
import json
print(json.dumps(neighborhood("validate_token", root="."), default=str))
'
```

If `which python` doesn't resolve to an interpreter with the deps, fall
back to the JSON tool form — it always works because it runs in-process.

### File writes: ALWAYS use the Write tool, NEVER use bash heredocs

To create or overwrite a file — especially anything longer than ~50 lines —
use the `Write` tool with `file_path` and `content`. **Do not** use
`cat <<'EOF' … EOF`, `printf '…' > file`, or `echo '…' > file` to write
file content.

Why: a heredoc body is part of the same output stream as the rest of your
response. Long bodies get cut off mid-stream when the model hits the
`max_tokens` limit, leaving a truncated file with no closing `EOF` —
broken syntax, silent corruption. The `Write` tool sidesteps this because
the file content travels as a structured argument, not as inline shell
text. This applies to *any* file: code, markdown, JSON, configs, READMEs.

```
✗  cat > /tmp/notes.md <<'EOF'      ← gets truncated on long content
   # Long doc here...
   EOF

✓  Write(file_path="/tmp/notes.md", content="# Long doc here...\n…")
```

For *appending* small lines to an existing file, `Edit` (with the trailing
context as `old_string`) or `Bash` with a single `>>` redirect is fine.
The rule is specifically: don't try to embed large file bodies inside
shell commands.

### When to use programmatic style vs individual tools

Use **agent_tools + a Python script** when:
- Searching for a symbol across the whole repo
- Aggregating data from multiple files
- The task involves loops, conditionals, or filtering
- You need an outline/map of unfamiliar code before diving in

Use **individual Read/Edit/Grep tools** when:
- You already know the exact file path and want to read or modify it
- A single-line grep is enough
- The task is genuinely one-step

For complex multi-file analysis, ALWAYS prefer one Python script with
`agent_tools` over 10+ separate tool calls. It's faster, more accurate,
easier to debug, and produces cleaner state in the conversation.
