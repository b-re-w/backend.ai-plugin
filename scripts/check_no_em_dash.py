"""Fail if an em dash (U+2014) appears in backend.ai-plugin/ or the workspace AGENTS.md (AGENTS.md rule)."""

import sys
from pathlib import Path

EM_DASH = chr(0x2014)  # spelled as a code point so this file passes its own check
PLUGIN = Path(__file__).resolve().parents[1]
SKIP = {"__pycache__", ".pytest_cache", ".git", ".venv"}

targets = [p for p in PLUGIN.rglob("*") if p.is_file() and not SKIP & set(p.parts)]
targets.append(PLUGIN.parent / "AGENTS.md")

hits = []
for path in targets:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for lineno, line in enumerate(text.splitlines(), 1):
        if EM_DASH in line:
            hits.append(f"{path}:{lineno}: {line.strip()}")

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # Windows consoles default to cp949
print("\n".join(hits) if hits else "no em dashes")
sys.exit(1 if hits else 0)
