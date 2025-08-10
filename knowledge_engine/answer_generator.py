"""LLM powered answer generation for retrieved context.

The module defines :class:`AnswerGenerator` which wraps the OpenAI chat
completion API.  It accepts a user query and a list of :class:`SearchResult`
objects and produces a final natural language answer.  The implementation is
designed to be optional – if the OpenAI client is not available or no API key
is supplied, the generator falls back to a simple heuristic response so that
the rest of the system can continue to function during tests.
"""

from __future__ import annotations

from typing import List, Optional
from loguru import logger

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - handled gracefully
    OpenAI = None
    logger.warning("OpenAI library not installed. Answer generation disabled.")

from .rrf_retriever import SearchResult


class AnswerGenerator:
    """Generate final answers from search results using an LLM."""

    def __init__(self, api_key: Optional[str], model: str = "gpt-3.5-turbo") -> None:
        self.model = model
        if api_key and OpenAI:
            try:  # pragma: no cover - network interaction
                self.client = OpenAI(api_key=api_key)
            except Exception as exc:  # pragma: no cover - handled gracefully
                logger.error(f"Failed to initialise OpenAI client: {exc}")
                self.client = None
        else:
            self.client = None

    # ------------------------------------------------------------------
    def _build_context(self, results: List[SearchResult], limit: int = 5) -> str:
        """Construct textual context from top ``limit`` results."""

        blocks = []
        for idx, res in enumerate(results[:limit], start=1):
            block = (
                f"[{idx}] {res.qualified_name}\n"
                f"Summary: {res.summary}\n"
                f"Purpose: {res.purpose}\n"
            )
            blocks.append(block)
        return "\n".join(blocks)

    # ------------------------------------------------------------------
    def generate_answer(self, query: str, results: List[SearchResult], limit: int = 5) -> str:
        """Generate an answer for ``query`` based on ``results``.

        When the OpenAI client is unavailable a simple fallback string is
        returned so that command line workflows remain usable in offline tests.
        """

        if not results:
            return "No relevant context available."

        context = self._build_context(results, limit)

        if not self.client:  # pragma: no cover - fallback path
            logger.debug("AnswerGenerator fallback in use")
            return (
                "\n".join([r.summary for r in results[:limit]])
                + f"\n\n(Answer generation disabled for query: '{query}')"
            )

        prompt = (
            "You are an assistant helping with questions about a code base. "
            "Use the following context to answer the user's question.\n\n"\
            + context
            + f"\n\nQuestion: {query}\nAnswer:"
        )

        try:  # pragma: no cover - network interaction
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover - handled gracefully
            logger.error(f"Answer generation failed: {exc}")
            return "Answer generation failed."

