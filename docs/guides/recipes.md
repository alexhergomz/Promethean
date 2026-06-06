# Recipes — Common Use Cases

Practical examples to get started with Promethean after installation.

---

## 1. Code Review with a Local Ollama Model

Use a free, local model to review code without sending anything to the cloud.

```bash
# Pull a capable model
ollama pull qwen2.5-coder:14b

# Start Promethean with Ollama
promethean --model ollama/qwen2.5-coder:14b
```

```
[project] » Review the code in src/api.py for security issues, performance 
             problems, and potential bugs. Be specific with line numbers.
```

For a full project audit:
```
[project] » Read all Python files in this project and give me a prioritized 
             list of the 10 most important issues to fix before shipping.
```

**Tip:** Ollama models run locally — your code never leaves your machine.

---

## 2. Remote Control via Telegram

Control Promethean from your phone while it runs on your server/workstation.

**Setup (one time):**
1. Message [@BotFather](https://t.me/BotFather) on Telegram, create a bot, get the token
2. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
3. Configure:
```bash
promethean
/config telegram_token=YOUR_BOT_TOKEN
/config telegram_chat_id=YOUR_CHAT_ID
/telegram start
```

**Usage from phone:**
```
You (Telegram): What files changed in the last commit?
Bot: [reads git log, shows diff summary]

You: Fix the bug in auth.py line 42
Bot: [edits file, shows diff, confirms]

You: !git status
Bot: [runs command, returns output]
```

**Tip:** Use `!command` prefix to run shell commands directly from Telegram.

---

## 3. Autonomous Research Agent

Let Promethean research a topic independently while you do other work.

```
[project] » /agent
```

Select **"Research Assistant"** from the wizard, then:
```
Research topic: Compare React Server Components vs Next.js App Router 
                for a production e-commerce site

Output: Write findings to research_output.md
```

The agent will:
- Search the web for current information
- Read documentation and blog posts
- Synthesize findings into the output file
- Continue iterating until the research is complete

**Monitor progress:**
```
/agents              # see running agents
/tasks               # see task progress
```

---

## 4. Quick Bug Fix Workflow

```bash
# Start with the bug context
promethean -p "Fix the TypeError in utils.py:42 where None is passed to len()"
```

Or interactively:
```
[project] » There's a crash when users submit an empty form. The error is 
             TypeError: argument of type 'NoneType' is not iterable in 
             handlers/form.py. Find and fix it.
```

Promethean will:
1. Read the file
2. Identify the root cause
3. Apply the fix
4. Show you the diff for approval

---

## 5. Multi-Model Brainstorm

Get perspectives from different models on a design decision.

```
[project] » /brainstorm

Topic: Should we use PostgreSQL or MongoDB for our user activity tracking 
       system? We expect 10M events/day with complex aggregation queries.
```

The brainstorm spawns multiple sub-agents that discuss and debate, then synthesizes a final recommendation.

---

## 6. Session Persistence Across Days

Work on a long-running project across multiple sessions:

```bash
# Day 1: Start working
promethean
[project] » Let's refactor the authentication module. Start by analyzing 
             the current auth flow...
# ... work happens ...
# Ctrl+D to exit (auto-saves)

# Day 2: Resume where you left off
promethean
[project] » /resume
# Your full conversation context is restored
[project] » Continue with the auth refactor. What's left?
```

**Tip:** Use `/save my-refactor` to name a session for easy retrieval later with `/load my-refactor`.

---

## 7. Monitoring AI Research Papers

Stay updated on topics that matter to you:

```
[project] » /monitor
```

Select **"Add subscription"**, then:
```
Topic: ai_research
Schedule: daily
Notification: --telegram
```

Every day, Promethean will:
- Fetch the latest papers from arXiv
- Summarize the most relevant ones
- Send you a digest via Telegram

Other subscription types: `stock_TSLA`, `crypto_BTC`, `world_news`, `custom:<query>`

---

## 8. Project Bootstrap with /init

Start a new project with AI-readable context:

```bash
mkdir my-new-project && cd my-new-project
git init
promethean
[my-new-project] » /init
```

This creates a `CLAUDE.md` file that Promethean reads on every startup — containing project conventions, tech stack, and guidelines that shape all future interactions.

---

## 9. Search Past Conversations

Find anything you discussed in previous sessions:

```
[project] » /search authentication bug
```

Output:
```
Found 2 session(s) matching "authentication bug":

  [a3f8c2e1] Auth refactor (gpt-4o)
    2026-04-14 15:30:22 · 12 turns
    How do I fix the >>>authentication<<< >>>bug<<< in login.py?

  [c9e2d1b3] Security review (claude-sonnet-4-6)
    2026-04-10 11:00:00 · 6 turns
    ...found an >>>authentication<<< >>>bug<<< in the middleware...
```

Then load and resume:
```
[project] » /load a3f8c2e1
Session loaded from ... (24 messages)
[project] » Continue where we left off with the auth fix.
```

**Tip:** Sessions are automatically indexed. Your first `/search` will import all existing JSON sessions into the search index.

---

## 10. Browse Dynamic Web Pages

Use `WebBrowse` for JavaScript-heavy pages that `WebFetch` can't render:

```
[project] » Go to https://github.com/trending and tell me the top 5 trending repos today.
```

The AI will use `WebBrowse` to render the page with headless Chromium and extract the content.

**Install:** `pip install promethean[browser] && playwright install chromium`

---

## 11. Read and Reply to Emails

```bash
# First, configure email (one time)
promethean
/config email_address=you@gmail.com
/config email_password=your-app-password
/config email_imap_host=imap.gmail.com
/config email_smtp_host=smtp.gmail.com
```

Then use naturally:

```
[project] » Check my latest emails from boss@company.com
[project] » Summarize the quarterly report email
[project] » Draft a reply saying I'll have the analysis ready by Friday
```

The AI reads your inbox, summarizes emails, and drafts replies — always asking for confirmation before sending.

---

## 12. Analyze PDFs and Spreadsheets

```
[project] » Read the contract at ~/Documents/contract.pdf and summarize the key terms
[project] » Open data.xlsx and find the top 10 customers by revenue
[project] » Extract text from this scanned receipt: ~/photos/receipt.jpg
```

**Install:**
```bash
pip install "promethean[files]"    # PDF + Excel
pip install "promethean[ocr]"      # image OCR (also needs: brew install tesseract)
```

---

## Tips

- **`/search <query>`** — full-text search across all past sessions
- **`/status`** — quick overview: model, token usage, cost, session stats
- **`/doctor`** — diagnose connectivity, dependencies, and configuration issues
- **`/compact`** — manually compress conversation when context gets large
- **`/copy`** — copy the last response to clipboard
- **`/export`** — export the full conversation to a Markdown file
- **`Ctrl+C`** — interrupt a long response without losing conversation
- **`!command`** — run a shell command inline (e.g., `!git status`)
