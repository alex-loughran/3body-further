---
name: Never use python3 -c with inline scripts
description: Inline python3 -c commands with comments trigger a security prompt that interrupts workflow
type: feedback
---

Never run Python code via `python3 -c "..."` with multi-line content or comments. The `\n#` pattern inside quoted arguments triggers Claude Code's injection detection, requiring manual approval each time.

**Why:** The security check is hardcoded and cannot be disabled via settings. It fires even when `python3` is in the allowlist.

**How to apply:** Always write code to a temporary `.py` file first (using Write tool), then execute with `python3 filename.py`. Delete the temp file after if not needed. This avoids the `\n#` pattern entirely.
