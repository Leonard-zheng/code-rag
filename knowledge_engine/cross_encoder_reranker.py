"""Cross-Encoder based reranking for search results.

This module provides a small wrapper around the `sentence-transformers`
`CrossEncoder` class.  The reranker is optional – if the library or model
cannot be loaded, the class gracefully degrades to a no-op so that the rest of
the system continues to work.  The reranking step scores the query and each
candidate document jointly and sorts the results by this relevance score.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING
from loguru import logger

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - handled gracefully
    CrossEncoder = None
    logger.warning(
        "sentence-transformers not installed. Cross-encoder reranking disabled."
    )


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .rrf_retriever import SearchResult


class CrossEncoderReranker:
    """Rerank results using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Initialise the reranker.

        Parameters
        ----------
        model_name:
            HuggingFace model identifier.  A small default model is used to
            minimise download size.  If the model or library cannot be loaded
            the reranker simply becomes inactive.
        """

        if CrossEncoder is None:
            self.model = None
            return

        try:  # pragma: no cover - network/model loading
            self.model = CrossEncoder(model_name)
        except Exception as exc:  # pragma: no cover - handled gracefully
            logger.warning(f"Failed to load cross encoder model '{model_name}': {exc}")
            self.model = None

    # ------------------------------------------------------------------
    def rerank(self, query: str, results: List["SearchResult"]) -> List["SearchResult"]:
        """Rerank ``results`` for ``query`` using the cross encoder.

        If the model is unavailable the results are returned unchanged.
        """

        if not self.model or not results:  # pragma: no cover - simple guard
            return results

        # Prepare pairs of (query, document text)
        pairs = [(query, f"{r.summary} {r.purpose}") for r in results]

        try:  # pragma: no cover - model inference
            scores = self.model.predict(pairs)
        except Exception as exc:  # pragma: no cover - handled gracefully
            logger.error(f"Cross-encoder prediction failed: {exc}")
            return results

        # Sort results by score and update ranking metadata
        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked: List["SearchResult"] = []
        for rank, (res, score) in enumerate(scored, start=1):
            res.rank = rank
            # Store the cross encoder score so that downstream components can use
            # it if desired.
            res.score = float(score)
            reranked.append(res)

        return reranked

