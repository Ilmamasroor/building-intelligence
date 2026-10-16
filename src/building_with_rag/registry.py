"""The one shared mode registry."""

from typing import get_args

from .contracts import Pattern

MODES: dict[str, Pattern] = {f"rag-{p}": p for p in get_args(Pattern)}
