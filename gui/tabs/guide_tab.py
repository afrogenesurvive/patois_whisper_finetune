"""
Guide Tab — Interactive page-based version of the training manual.

Provides a sidebar Table of Contents, a main content area for the current
page, and Previous / Next navigation buttons.
"""

from typing import Tuple

import gradio as gr

from gui.guide import (
    get_page,
    get_toc,
    has_next,
    has_prev,
    sections,
    total_pages,
)


def _build_toc_choices() -> list:
    """Return a list of ``display / value`` pairs for the TOC radio."""
    return [s["title"] for s in sections]


def _on_toc_select(
    selection: str, current_idx: int
) -> Tuple[int, str, str, gr.Button, gr.Button]:
    """Jump to the page matching the TOC selection."""
    if not selection:
        return current_idx, "", "", gr.Button(), gr.Button()
    # Find the index matching this title
    for i, s in enumerate(sections):
        if s["title"] == selection:
            page = get_page(i)
            nav = _build_nav_label(i)
            prev_btn = gr.Button(visible=has_prev(i))
            next_btn = gr.Button(visible=has_next(i))
            return i, page["body"], nav, prev_btn, next_btn
    return current_idx, "", "", gr.Button(), gr.Button()


def _on_prev(current_idx: int) -> Tuple[int, str, str, gr.Button, gr.Button, str]:
    """Go to the previous page."""
    new_idx = max(0, current_idx - 1)
    page = get_page(new_idx)
    nav = _build_nav_label(new_idx)
    prev_btn = gr.Button(visible=has_prev(new_idx))
    next_btn = gr.Button(visible=has_next(new_idx))
    return new_idx, page["body"], nav, prev_btn, next_btn, page["title"]


def _on_next(current_idx: int) -> Tuple[int, str, str, gr.Button, gr.Button, str]:
    """Go to the next page."""
    new_idx = min(total_pages - 1, current_idx + 1)
    page = get_page(new_idx)
    nav = _build_nav_label(new_idx)
    prev_btn = gr.Button(visible=has_prev(new_idx))
    next_btn = gr.Button(visible=has_next(new_idx))
    return new_idx, page["body"], nav, prev_btn, next_btn, page["title"]


def _build_nav_label(index: int) -> str:
    """Return a navigation label like ``"Page 3 / 14"``."""
    return f"Page {index + 1} / {total_pages}"


def create_guide_tab() -> gr.Tab:
    """Build the interactive guide tab with TOC, content, and navigation."""
    toc_choices = _build_toc_choices()
    first_page = get_page(0)

    with gr.Tab("📖 Guide") as tab:
        gr.Markdown(
            "## Training Manual\n"
            "Browse the complete Whisper Fine-Tune guide below. "
            "Use the sidebar to jump to any section or use Previous / Next "
            "to read sequentially."
        )

        with gr.Row():
            # ── Sidebar: Table of Contents ──────────────────────────
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("### Table of Contents")
                toc = gr.Radio(
                    choices=toc_choices,
                    value=toc_choices[0] if toc_choices else None,
                    label="",
                    interactive=True,
                )

            # ── Main: Content + Navigation ──────────────────────────
            with gr.Column(scale=3):
                page_label = gr.Markdown(
                    value=f"**{_build_nav_label(0)}**",
                )
                content = gr.Markdown(
                    value=first_page["body"],
                    label="Page Content",
                )

                with gr.Row():
                    prev_btn = gr.Button(
                        "◀ Previous",
                        variant="secondary",
                        scale=1,
                        visible=has_prev(0),
                    )
                    next_btn = gr.Button(
                        "Next ▶",
                        variant="primary",
                        scale=1,
                        visible=has_next(0),
                    )

        # ── State: track current page index ─────────────────────────
        page_index = gr.State(value=0)

        # ── Wire events ─────────────────────────────────────────────

        # TOC selection → jump to page
        toc.change(
            fn=_on_toc_select,
            inputs=[toc, page_index],
            outputs=[page_index, content, page_label, prev_btn, next_btn],
        )

        # Previous button
        prev_btn.click(
            fn=_on_prev,
            inputs=[page_index],
            outputs=[page_index, content, page_label, prev_btn, next_btn, toc],
        )

        # Next button
        next_btn.click(
            fn=_on_next,
            inputs=[page_index],
            outputs=[page_index, content, page_label, prev_btn, next_btn, toc],
        )

    return tab
