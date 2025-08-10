"""
RRF Retriever Module

Implements Reciprocal Rank Fusion (RRF) for combining results from
vector search and BM25 keyword search engines.
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from loguru import logger

from .dual_indexer import DualEngineIndexer
from .cross_encoder_reranker import CrossEncoderReranker


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    qualified_name: str
    score: float
    rank: int
    source: str  # 'vector' or 'bm25' or 'fusion'
    function_name: str
    summary: str
    purpose: str
    file_path: str
    complexity: str
    matched_terms: List[str] = None


class QueryAnalyzer:
    """Analyzes queries to determine search strategy weights."""
    
    def __init__(self):
        # Patterns that suggest different search strategies
        self.code_patterns = [
            r'def\s+\w+', r'class\s+\w+', r'import\s+', r'from\s+\w+',
            r'\w+\(.*\)', r'\.py$', r'\.js$', r'\.java$'
        ]
        
        self.semantic_keywords = [
            'how', 'what', 'why', 'when', 'where', 'explain', 'describe',
            'purpose', 'function', 'behavior', 'usage', 'example'
        ]
        
        self.technical_keywords = [
            'implement', 'algorithm', 'method', 'function', 'class',
            'variable', 'parameter', 'return', 'error', 'exception'
        ]
    
    def analyze_query(self, query: str) -> Dict[str, float]:
        """
        Analyze query and return strategy weights.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary with 'vector_weight' and 'bm25_weight' keys
        """
        query_lower = query.lower()
        
        # Default weights
        vector_weight = 0.5
        bm25_weight = 0.5
        
        # Adjust based on code patterns
        code_score = sum(1 for pattern in self.code_patterns 
                        if re.search(pattern, query, re.IGNORECASE))
        
        # Adjust based on semantic vs technical keywords
        semantic_score = sum(1 for keyword in self.semantic_keywords 
                           if keyword in query_lower)
        technical_score = sum(1 for keyword in self.technical_keywords 
                            if keyword in query_lower)
        
        # Heuristics for weight adjustment
        if code_score > 0:
            # Code snippets favor BM25 (exact matching)
            bm25_weight += 0.2
            vector_weight -= 0.2
            
        if semantic_score > technical_score:
            # Semantic questions favor vector search
            vector_weight += 0.2
            bm25_weight -= 0.2
            
        if len(query.split()) > 10:
            # Longer queries favor vector search
            vector_weight += 0.1
            bm25_weight -= 0.1
        elif len(query.split()) <= 3:
            # Short queries favor BM25
            bm25_weight += 0.1
            vector_weight -= 0.1
        
        # Normalize weights
        total_weight = vector_weight + bm25_weight
        vector_weight /= total_weight
        bm25_weight /= total_weight
        
        return {
            'vector_weight': max(0.1, min(0.9, vector_weight)),
            'bm25_weight': max(0.1, min(0.9, bm25_weight)),
            'code_score': code_score,
            'semantic_score': semantic_score,
            'technical_score': technical_score
        }


class RRFRetriever:
    """Retrieves and fuses results from multiple search engines using RRF."""
    
    def __init__(
        self, 
        indexer: DualEngineIndexer,
        rrf_k: int = 60,
        default_limit: int = 10
    ):
        """
        Initialize RRF Retriever.
        
        Args:
            indexer: DualEngineIndexer instance
            rrf_k: RRF parameter (typically 60)
            default_limit: Default number of results to return
        """
        self.indexer = indexer
        self.rrf_k = rrf_k
        self.default_limit = default_limit
        self.query_analyzer = QueryAnalyzer()

        # Optional cross-encoder reranker.  If the underlying model isn't
        # available the reranker becomes a no-op.
        try:
            self.reranker = CrossEncoderReranker()
        except Exception as exc:  # pragma: no cover - handled gracefully
            logger.warning(f"Cross-encoder reranker disabled: {exc}")
            self.reranker = None
    
    def reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF Score = vector_weight * (1/(rank_v + k)) + bm25_weight * (1/(rank_b + k))
        
        Args:
            vector_results: List of (qualified_name, score) from vector search
            bm25_results: List of (qualified_name, score) from BM25 search
            vector_weight: Weight for vector results
            bm25_weight: Weight for BM25 results
            
        Returns:
            List of (qualified_name, rrf_score) tuples sorted by RRF score
        """
        # Create rank mappings
        vector_ranks = {qn: rank + 1 for rank, (qn, _) in enumerate(vector_results)}
        bm25_ranks = {qn: rank + 1 for rank, (qn, _) in enumerate(bm25_results)}
        
        # Get all unique documents
        all_documents = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        
        rrf_scores = {}
        for qn in all_documents:
            score = 0.0
            
            # Add vector contribution
            if qn in vector_ranks:
                score += vector_weight * (1.0 / (vector_ranks[qn] + self.rrf_k))
            
            # Add BM25 contribution
            if qn in bm25_ranks:
                score += bm25_weight * (1.0 / (bm25_ranks[qn] + self.rrf_k))
            
            rrf_scores[qn] = score
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.debug(f"RRF fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 "
                    f"-> {len(sorted_results)} fused results")
        
        return sorted_results
    
    def extract_matched_terms(self, query: str, text: str) -> List[str]:
        """Extract terms from query that appear in the text."""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        text_lower = text.lower()
        
        matched = [term for term in query_terms if term in text_lower]
        return matched
    
    def search(
        self, 
        query: str, 
        limit: Optional[int] = None,
        vector_limit: int = 20,
        bm25_limit: int = 20,
        use_query_analysis: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search using RRF fusion.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            vector_limit: Number of results to get from vector search
            bm25_limit: Number of results to get from BM25 search
            use_query_analysis: Whether to use query analysis for weighting
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        if limit is None:
            limit = self.default_limit
        
        logger.info(f"Searching for: '{query}' (limit={limit})")
        
        # Analyze query to determine weights
        if use_query_analysis:
            analysis = self.query_analyzer.analyze_query(query)
            vector_weight = analysis['vector_weight']
            bm25_weight = analysis['bm25_weight']
            
            logger.debug(f"Query analysis: vector={vector_weight:.2f}, bm25={bm25_weight:.2f}")
        else:
            vector_weight = 0.5
            bm25_weight = 0.5
        
        # Perform searches in parallel (conceptually)
        vector_results = self.indexer.vector_search(query, limit=vector_limit)
        bm25_results = self.indexer.bm25_search(query, limit=bm25_limit)
        
        if not vector_results and not bm25_results:
            logger.warning("No results from either search engine")
            return []
        
        # Fuse results using RRF
        fused_results = self.reciprocal_rank_fusion(
            vector_results, bm25_results, vector_weight, bm25_weight
        )
        
        # Convert to SearchResult objects with metadata
        search_results = []
        for rank, (qn, rrf_score) in enumerate(fused_results[:limit]):
            doc = self.indexer.get_document(qn)
            if not doc:
                logger.warning(f"Document not found for {qn}")
                continue
            
            # Determine source
            source = "fusion"
            if qn in [r[0] for r in vector_results] and qn in [r[0] for r in bm25_results]:
                source = "both"
            elif qn in [r[0] for r in vector_results]:
                source = "vector"
            elif qn in [r[0] for r in bm25_results]:
                source = "bm25"
            
            # Extract matched terms
            matched_terms = self.extract_matched_terms(
                query, 
                f"{doc.get('summary', '')} {doc.get('purpose', '')} {doc.get('function_name', '')}"
            )
            
            result = SearchResult(
                qualified_name=qn,
                score=rrf_score,
                rank=rank + 1,
                source=source,
                function_name=doc.get('function_name', ''),
                summary=doc.get('summary', ''),
                purpose=doc.get('purpose', ''),
                file_path=doc.get('file_path', ''),
                complexity=doc.get('complexity', 'UNKNOWN'),
                matched_terms=matched_terms
            )
            
            search_results.append(result)
        
        # Optional cross-encoder re-ranking for higher precision
        if hasattr(self, "reranker") and self.reranker:
            search_results = self.reranker.rerank(query, search_results)

        logger.info(f"Returned {len(search_results)} fused results")

        return search_results
    
    def search_similar_functions(
        self, 
        function_qualified_name: str, 
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Find functions similar to the given function.
        
        Args:
            function_qualified_name: Qualified name of the reference function
            limit: Number of similar functions to return
            
        Returns:
            List of SearchResult objects
        """
        # Get the reference function's document
        ref_doc = self.indexer.get_document(function_qualified_name)
        if not ref_doc:
            logger.error(f"Reference function not found: {function_qualified_name}")
            return []
        
        # Use the function's summary and purpose as query
        query = f"{ref_doc.get('summary', '')} {ref_doc.get('purpose', '')}"
        
        results = self.search(query, limit=limit + 1)  # +1 to account for self-match
        
        # Remove the reference function itself from results
        similar_results = [r for r in results if r.qualified_name != function_qualified_name]
        
        return similar_results[:limit]
    
    def search_by_complexity(
        self, 
        complexity: str, 
        limit: int = 20
    ) -> List[SearchResult]:
        """
        Find functions by complexity level.
        
        Args:
            complexity: Complexity level ('LOW', 'MEDIUM', 'HIGH')
            limit: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # This is a simple filter-based search since we don't have complex filtering
        # In a real implementation, you might want to add this to Weaviate queries
        
        all_functions = []
        for qn, doc in self.indexer.document_store.items():
            if doc.get('complexity', '').upper() == complexity.upper():
                result = SearchResult(
                    qualified_name=qn,
                    score=1.0,  # All have equal score for this type of search
                    rank=len(all_functions) + 1,
                    source="filter",
                    function_name=doc.get('function_name', ''),
                    summary=doc.get('summary', ''),
                    purpose=doc.get('purpose', ''),
                    file_path=doc.get('file_path', ''),
                    complexity=doc.get('complexity', 'UNKNOWN')
                )
                all_functions.append(result)
                
                if len(all_functions) >= limit:
                    break
        
        logger.info(f"Found {len(all_functions)} functions with {complexity} complexity")
        return all_functions
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search index."""
        stats = {
            'total_documents': len(self.indexer.document_store),
            'vector_index_available': self.indexer.weaviate_client is not None,
            'bm25_index_available': self.indexer.bm25_index is not None,
            'complexity_distribution': defaultdict(int),
            'file_distribution': defaultdict(int)
        }
        
        for doc in self.indexer.document_store.values():
            complexity = doc.get('complexity', 'UNKNOWN')
            stats['complexity_distribution'][complexity] += 1
            
            file_path = doc.get('file_path', '')
            if file_path:
                file_name = file_path.split('/')[-1]
                stats['file_distribution'][file_name] += 1
        
        return dict(stats)