# Security Policy

## Supported versions

Security fixes are provided for the current 3.x release line.

| Version | Supported          |
| ------- | ------------------ |
| 3.x     | :white_check_mark: |
| < 3.0   | :x:                |

## Reporting a vulnerability

Please report security issues **privately** through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository: the **Security** tab → **Report a vulnerability**.

Please do **not** open a public issue for a suspected vulnerability. When you
report, include:

- a description of the issue and its impact,
- steps or a proof of concept to reproduce it,
- the version / commit and your environment (OS, Python, GPU).

You can expect an acknowledgement of your report and follow-up as the issue is
triaged and addressed. Coordinated disclosure is appreciated: please give the
maintainers a reasonable window to ship a fix before any public discussion.

## Security model

Promethean runs an autonomous agent loop locally and ships guardrails that are
relevant context when reporting issues:

- a **bash deny-list** and **sensitive-path jail** in the tool layer
  (`tools/security.py`) that constrain shell and filesystem access;
- **sandboxed sub-agents** that run with a restricted tool whitelist (excluding
  `Bash`/`Write`/`Edit`), with their `Read` access path-jailed to a workspace.

Because the agent can execute tools on the host, run it with the level of
permission appropriate to your trust in the model and the task. In particular,
the `--accept-all` mode disables interactive permission prompts and should only
be used in environments you control.
