"""
Training Manual Guide — Parses ``docs/training_manual.md`` into page-able
sections for the interactive guide tab.

Provides:
    - ``sections`` — list of dicts with ``title``, ``anchor``, ``body`` (markdown)
    - ``get_page(index)`` — return the section at *index*
    - ``get_toc()`` — list of ``(index, title)`` for the Table of Contents
    - ``total_pages`` — number of sections
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

_DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "training_manual.md"


def _parse_sections() -> List[Dict]:
    """
    Read the training manual and split it into page-able sections.
    Each ``## <number>.`` or ``## Appendix`` heading becomes one section.
    Content before the first such heading becomes the Overview page.
    The ``## Table of Contents`` heading is skipped.
    """
    if not _DOC_PATH.exists():
        return [
            {
                "title": "Guide Not Found",
                "anchor": "guide-not-found",
                "body": (
                    "The training manual was not found at "
                    f"`docs/training_manual.md`.\n\n"
                    "Please ensure the file exists and try again."
                ),
            }
        ]

    text = _DOC_PATH.read_text(encoding="utf-8")

    # Split on ##-level headings that start a numbered section or Appendix.
    # This matches lines like "## 1. ...", "## 10. ..." and "## Appendix: ..."
    # but NOT "## Table of Contents" or "### 4.1 ..." (subsections).
    parts = re.split(
        r"^(?=## (?:(\d+\.)|(Appendix)))", text, flags=re.MULTILINE
    )

    sections: List[Dict] = []
    preamble_parts: List[str] = []

    for part in parts:
        if not part or part in ("\n", ""):
            continue
        part = part.strip()
        if not part:
            continue

        # Check if this part starts with a ## heading that is a numbered
        # section or Appendix (not "Table of Contents").
        heading_match = re.match(
            r"^## (?:(\d+\.\s*.+)|(Appendix:?\s*.+))", part
        )

        if heading_match:
            title = heading_match.group(1) or heading_match.group(2)
            title = title.strip()
            # Strip the heading line(s) from the body
            body = re.sub(r"^## .+\n?", "", part, count=1).strip()
            # Create an anchor from the title (dash-separated lowercase)
            anchor = title.lower()
            anchor = re.sub(r"[^\w\s-]", "", anchor)
            anchor = re.sub(r"[\s_]+", "-", anchor).strip("-")
            sections.append(
                {"title": title, "anchor": anchor, "body": body}
            )
        else:
            # Preamble or skipped heading (Table of Contents, etc.)
            preamble_parts.append(part)

    # Build the Overview page from all preamble content
    preamble_text = "\n\n".join(preamble_parts).strip()
    if preamble_text:
        # Remove the # main title line (use its text as the page title)
        title_match = re.match(r"^#\s+(.+)", preamble_text)
        overview_title = title_match.group(1) if title_match else "Overview"
        body = re.sub(r"^#\s+.+\n?", "", preamble_text, count=1).strip()
        sections.insert(
            0,
            {
                "title": overview_title,
                "anchor": "overview",
                "body": body,
            }
        )

    return sections


# Load sections once at import time
sections: List[Dict] = _parse_sections()
total_pages: int = len(sections)


def get_page(index: int) -> Dict:
    """Return the section dict at *index* (0-based), clamped to valid range."""
    if not sections:
        return {"title": "Empty", "anchor": "empty", "body": "No content available."}
    idx = max(0, min(index, total_pages - 1))
    return sections[idx]


def get_toc() -> List[Tuple[int, str]]:
    """Return a list of ``(index, title)`` pairs for the Table of Contents."""
    return [(i, s["title"]) for i, s in enumerate(sections)]


def has_prev(index: int) -> bool:
    """Return ``True`` if there is a page before *index*."""
    return index > 0


def has_next(index: int) -> bool:
    """Return ``True`` if there is a page after *index*."""
    return index < total_pages - 1
