"""Story 3.2: readable Open WebUI text rendered from pipeline events and the final QueryResult."""

from __future__ import annotations

from collections.abc import Iterator

from .contracts import QueryResult
from .generation.answer import CONFIDENCE_HIGH, MAX_ATTEMPTS
from .generation.events import AttemptChecked, Event, Final, TextPiece

DRAFT_LABEL = "DRAFT — checking evidence"
LOW_CONFIDENCE_LABEL = "DRAFT — low confidence, not the final answer."
UNEXPECTED_LINE = (
    "UNAVAILABLE — an unexpected error stopped this reply; any text above is incomplete "
    "and not validated."
)
UNEXPECTED_LINE_NO_TEXT = (
    "UNAVAILABLE — an unexpected error stopped this reply; no answer was written."
)
_DETAIL_CHARS = 300


def _detail(issues: list[str]) -> str:
    return "; ".join(issues)[:_DETAIL_CHARS]


def _confidence_word(value: float | None) -> str:
    return "high" if value is not None and value >= CONFIDENCE_HIGH else "low"


def _source_line(citation: dict) -> str:
    src = citation.get("source") or {}
    start, end = src.get("page_start"), src.get("page_end")
    pages = "" if start is None else (f"p. {start}" if start == end else f"pp. {start}-{end}")
    parts = [citation["label"], citation["section_id"], citation.get("heading") or "",
             src.get("source_pdf") or "", pages]
    return "- " + " · ".join(p for p in parts if p)


def _model_reason(result: QueryResult) -> str | None:
    for line in (result.generation.trace if result.generation else []):
        if str(line).startswith("model reason: "):
            return str(line)[len("model reason: "):]
    return None


def closing(result: QueryResult) -> str:
    """Text that follows any streamed draft text; always starts with a blank line."""
    text = _closing(result)
    if result.status == "partial_answer" and result.generation is not None:
        missing = "; ".join(e.subquestion for e in result.subquestions if e.status == "no_evidence")
        text += (
            f"\n\nPartial answer — no evidence was found for: {missing}. "
            "This does not compare all parts of the question."
        )
    if result.status == "partial_result" and result.generation is not None:
        text += "\n\nPartial result — evidence for this question was limited; check the sources."
    return text


def _closing(result: QueryResult) -> str:
    gen = result.generation
    if gen is None:
        if result.status in ("unavailable", "hyde_unavailable"):
            msg = result.message.removeprefix("Retrieval unavailable:").strip()
            return f"\n\nUNAVAILABLE — retrieval: {msg}"
        return f"\n\n{result.message}"
    if gen.outcome == "answered":
        sources = "\n".join(_source_line(c) for c in gen.citations)
        return (
            f"\n\n---\nEvidence check passed — confidence: {_confidence_word(gen.confidence)} "
            f"(attempt {gen.attempts} of {MAX_ATTEMPTS}). "
            "The draft text above is now the validated answer.\n\n"
            f"Sources:\n{sources}"
        )
    if gen.outcome == "insufficient_evidence":
        reason = _model_reason(result)
        base = (
            "No supporting passages were found in the indexed documents."
            if gen.context_outcome == "no_results"
            else "The retrieved evidence is not sufficient to answer this question."
        )
        return f"\n\n{base}" + (f" {reason}" if reason else "")
    if gen.outcome == "unavailable":
        if gen.draft_answer:
            return (
                "\n\nUNAVAILABLE — the answer service stopped while writing; the text above is "
                "incomplete and not validated."
            )
        return "\n\nUNAVAILABLE — the answer service could not write an answer."
    # malformed
    if gen.confidence is not None:
        return (
            f"\n\n---\n{LOW_CONFIDENCE_LABEL} {gen.low_confidence_reason or ''}\n"
            f"Failed checks: {_detail(gen.issues)}"
        )
    return f"\n\nNo validated answer: the model reply could not be used. {_detail(gen.issues)}"


def render_final(result: QueryResult) -> str:
    """Non-streaming text. A validated answer carries no DRAFT prefix (nothing was shown yet)."""
    gen = result.generation
    if gen is None:
        return closing(result).lstrip()
    if gen.outcome == "answered":
        return f"{gen.answer}{closing(result)}"
    if gen.draft_answer:
        return f"{DRAFT_LABEL}\n\n{gen.draft_answer}{closing(result)}"
    return closing(result).lstrip()


def stream_chat_text(events: Iterator[Event]) -> Iterator[str]:
    """Text pieces for the streamed reply. Always ends cleanly, even if the run breaks."""
    label_shown = False
    shown_any = False
    try:
        for event in events:
            if isinstance(event, TextPiece):
                shown_any = True
                if not label_shown:
                    label_shown = True
                    yield f"{DRAFT_LABEL}\n\n"
                yield event.text
            elif isinstance(event, AttemptChecked):
                if event.will_retry:
                    if label_shown:
                        yield "\n\n"
                    yield (
                        f"Check failed (attempt {event.attempt} of {event.max_attempts}): "
                        f"{_detail(event.issues)}. Retrying…\n\n"
                    )
                label_shown = False
            elif isinstance(event, Final):
                text = closing(event.result)
                yield text if shown_any else text.lstrip()
    except Exception:  # never leak details; the stream must still close
        yield f"\n\n{UNEXPECTED_LINE}"
