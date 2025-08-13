"""
Knowledge Engine Module for Code RAG System

This module provides intelligent knowledge extraction and retrieval capabilities
built on top of the AST parsing infrastructure.

Supports both OpenAI and local model versions.
"""

from .dependency_graph import DependencyGraphBuilder
from .rrf_retriever import RRFRetriever
from .answer_generator import AnswerGenerator

# OpenAI version
from .topological_summary import TopologicalSummaryGenerator
from .dual_indexer import DualEngineIndexer

# Local model version (Langchain + BGE-M3 + Ollama) 
try:
    from .local_models import LocalEmbeddingModel, LocalLLMModel, LangchainSummaryChain
    from .local_topological_summary import LocalTopologicalSummaryGenerator
    from .local_dual_indexer import LocalDualEngineIndexer
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False

__all__ = [
    # Core components (both versions)
    "DependencyGraphBuilder",
    "RRFRetriever",
    "AnswerGenerator",
    
    # OpenAI version
    "TopologicalSummaryGenerator", 
    "DualEngineIndexer",
]

# Add local model exports if available
if LOCAL_MODELS_AVAILABLE:
    __all__.extend([
        "LocalEmbeddingModel",
        "LocalLLMModel",
        "LangchainSummaryChain",
        "LocalTopologicalSummaryGenerator",
        "LocalDualEngineIndexer",
    ])
