"""``introspy guide``: the opt-in pitch, read from ``guide.md`` next to this file.

The prose lives in the Markdown file and nowhere else. The CLI prints it and the
docs site includes it through a snippet, so the two cannot drift. Keep the page
under a screenful and end it with a single ``next: run`` line;
``tests/test_guide.py`` enforces both.
"""

from pathlib import Path

GUIDE_PATH = Path(__file__).with_name("guide.md")


def guide_text() -> str:
    """The page, byte for byte."""
    return GUIDE_PATH.read_text(encoding="utf-8")
