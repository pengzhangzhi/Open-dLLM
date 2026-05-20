---
name: create-issue
description: Use when the user asks to create a GitHub issue. Gathers context from the current conversation (findings, commands, file changes, wandb links) and drafts a well-structured issue with reproduce steps, results tables, and links. Use ONLY when the user explicitly asks to create or draft a GitHub issue.
---

# Create GitHub Issue

When the user asks to create a GitHub issue, follow this workflow:

## 1. Determine the target repo

- Run `git remote -v` to list remotes.
- Default to `origin`. If the user specifies a different remote (e.g. "send to upstream"), use that.
- If the user says "not upstream" or similar, confirm which remote to target.

## 2. Gather context from the conversation

Pull from the current session everything relevant to the issue:

- **What was done** — features added, bugs fixed, experiments run
- **Commands to reproduce** — exact shell commands the user ran
- **Results** — loss curves, benchmark numbers, comparison tables
- **Wandb links** — grep logs for `wandb` URLs or ask the user
- **Bugs found and fixed** — with file paths and line numbers
- **Key files changed** — with one-line descriptions
- **Next steps / open questions**

If any of these are missing, ask the user before proceeding.

## 3. Draft the issue body

Structure:

```
## What is [topic]?

One-paragraph summary of the feature/experiment/finding.

## [Key finding or result]

Tables, numbers, comparisons. Use GitHub-flavored markdown tables.

## Reproduce

Step-by-step commands the reader can copy-paste. Include:
- Prerequisites (hardware, data, model weights)
- Exact commands with all flags
- Any environment variables needed

## Wandb / Logs

Links to wandb runs, log file paths, or other artifacts.

## Bugs found and fixed

Numbered list with file:line references.

## Key files

Table of file → purpose.

## Next steps

- [ ] Checkbox items

## Open questions

Numbered list of unresolved items.
```

## 4. Create the issue

```bash
gh issue create --repo <owner>/<repo> --title "<title>" --body "<body>"
```

Use a HEREDOC for the body to preserve formatting. Target the repo determined in step 1.

## 5. Confirm

Return the issue URL to the user.
