"""
Enhanced Graph Updater

Extends the original GraphUpdater with intelligent summary generation and indexing capabilities.
Implements the complete pipeline: AST parsing -> dependency analysis -> summary generation -> indexing.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from code_parser.graph_updater import GraphUpdater
from code_parser.services.graph_service import MemgraphIngestor
from knowledge_engine.dependency_graph import DependencyGraphBuilder
from knowledge_engine.topological_summary import TopologicalSummaryGenerator
from knowledge_engine.dual_indexer import DualEngineIndexer
from knowledge_engine.rrf_retriever import RRFRetriever


class EnhancedGraphUpdater(GraphUpdater):
    """Enhanced version of GraphUpdater with intelligent knowledge extraction capabilities."""
    
    def __init__(
        self,
        ingestor: MemgraphIngestor,
        repo_path: Path,
        parsers: dict[str, Any],
        queries: dict[str, Any],
        # New parameters for knowledge engine
        openai_api_key: str,
        weaviate_url: str = "http://localhost:8080",
        enable_summary_generation: bool = True,
        enable_indexing: bool = True,
        summary_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize Enhanced Graph Updater.
        
        Args:
            ingestor: Memgraph database ingestor
            repo_path: Path to repository
            parsers: Tree-sitter parsers
            queries: AST queries
            openai_api_key: OpenAI API key for LLM and embeddings
            weaviate_url: Weaviate database URL
            enable_summary_generation: Whether to generate summaries
            enable_indexing: Whether to build search indices
            summary_model: Model for summary generation
            embedding_model: Model for embeddings
        """
        # Initialize parent class
        super().__init__(ingestor, repo_path, parsers, queries)
        
        # Knowledge engine configuration
        self.openai_api_key = openai_api_key
        self.weaviate_url = weaviate_url
        self.enable_summary_generation = enable_summary_generation
        self.enable_indexing = enable_indexing
        self.summary_model = summary_model
        self.embedding_model = embedding_model
        
        # Knowledge engine components (initialized lazily)
        self.dependency_builder: Optional[DependencyGraphBuilder] = None
        self.summary_generator: Optional[TopologicalSummaryGenerator] = None
        self.dual_indexer: Optional[DualEngineIndexer] = None
        self.retriever: Optional[RRFRetriever] = None
        
        # Storage for results
        self.summaries: Dict[str, Any] = {}
        self.knowledge_index_built = False
    
    def _initialize_knowledge_components(self):
        """Initialize knowledge engine components lazily."""
        logger.info("Initializing knowledge engine components...")
        
        try:
            # Dependency graph builder
            self.dependency_builder = DependencyGraphBuilder(
                ingestor=self.ingestor,
                project_name=self.project_name
            )
            logger.info("✓ Dependency graph builder initialized")
            
            # Summary generator (if enabled)
            if self.enable_summary_generation and self.openai_api_key:
                self.summary_generator = TopologicalSummaryGenerator(
                    openai_api_key=self.openai_api_key,
                    model=self.summary_model
                )
                logger.info("✓ Summary generator initialized")
            else:
                logger.info("⚠ Summary generation disabled or no OpenAI API key")
            
            # Dual indexer (if enabled)
            if self.enable_indexing and self.openai_api_key:
                self.dual_indexer = DualEngineIndexer(
                    weaviate_url=self.weaviate_url,
                    openai_api_key=self.openai_api_key,
                    embedding_model=self.embedding_model
                )
                logger.info("✓ Dual indexer initialized")
            else:
                logger.info("⚠ Indexing disabled or no OpenAI API key")
                
        except Exception as e:
            logger.error(f"Failed to initialize knowledge components: {e}")
            raise
    
    def run(self) -> None:
        """
        Enhanced run method with knowledge extraction pipeline.
        
        Pipeline:
        1. Original AST parsing (3 phases)
        2. Build dependency graph
        3. Generate summaries topologically
        4. Build search indices
        5. Initialize retriever
        """
        logger.info("=== Starting Enhanced Graph Update Process ===")
        
        # Phase 1-3: Original AST parsing
        logger.info("\n--- PHASE 1-3: AST Parsing and Graph Construction ---")
        super().run()  # Run original pipeline
        
        # Initialize knowledge engine components
        self._initialize_knowledge_components()
        
        # Phase 4: Build dependency graph
        logger.info("\n--- PHASE 4: Dependency Graph Analysis ---")
        self._build_dependency_graph()
        
        # Phase 5: Generate summaries (if enabled)
        if self.enable_summary_generation and self.summary_generator:
            logger.info("\n--- PHASE 5: Topological Summary Generation ---")
            self._generate_summaries()
        
        # Phase 6: Build search indices (if enabled)
        if self.enable_indexing and self.dual_indexer and self.summaries:
            logger.info("\n--- PHASE 6: Search Index Construction ---")
            self._build_search_indices()
        
        # Phase 7: Initialize retriever
        if self.dual_indexer:
            logger.info("\n--- PHASE 7: Initialize Retriever ---")
            self._initialize_retriever()
        
        logger.info("\n=== Enhanced Graph Update Process Complete ===")
        self._print_summary_statistics()
    
    def _build_dependency_graph(self):
        """Build function dependency graph for topological processing."""
        if not self.dependency_builder:
            logger.error("Dependency builder not initialized")
            return
            
        try:
            # Build call graph
            call_graph = self.dependency_builder.build_call_graph()
            logger.info(f"Built dependency graph with {call_graph.number_of_nodes()} functions")
            
            # Check for circular dependencies
            sccs = self.dependency_builder.detect_strongly_connected_components()
            if sccs:
                logger.warning(f"Found {len(sccs)} circular dependency groups")
            
        except Exception as e:
            logger.error(f"Failed to build dependency graph: {e}")
            raise
    
    def _generate_summaries(self):
        """Generate function summaries in topological order."""
        if not self.dependency_builder or not self.summary_generator:
            logger.error("Required components not initialized for summary generation")
            return
            
        try:
            # Get topological batches
            batches = self.dependency_builder.get_topological_batches()
            if not batches:
                logger.warning("No function batches generated, skipping summary generation")
                return
                
            logger.info(f"Processing {len(batches)} dependency-ordered batches")
            
            # Convert function metadata for processing
            function_metadata = {}
            for qn, data in self.dependency_builder.function_data.items():
                function_metadata[qn] = data
                
            # Process batches and generate summaries
            def get_context_library(batch_qns: List[str]) -> Dict[str, str]:
                """Get context library for a batch."""
                context = {}
                for qn in batch_qns:
                    deps = self.dependency_builder.get_function_dependencies(qn)
                    for dep_qn in deps:
                        if dep_qn in self.summaries:
                            context[dep_qn] = self.summaries[dep_qn].summary
                return context
            
            # Convert batches to use function metadata
            batch_metadata = []
            for batch_qns in batches:
                batch_functions = []
                for qn in batch_qns:
                    if qn in function_metadata:
                        batch_functions.append(function_metadata[qn])
                batch_metadata.append(batch_functions)
            
            # Generate summaries
            self.summaries = self.summary_generator.process_batches(
                batch_metadata, 
                get_context_library
            )
            
            logger.info(f"Generated {len(self.summaries)} function summaries")
            
            # Export summaries for backup
            summary_file = self.repo_path / "function_summaries.json"
            self.summary_generator.export_summaries(str(summary_file))
            
        except Exception as e:
            logger.error(f"Failed to generate summaries: {e}")
            # Don't raise - continue with available data
            logger.warning("Continuing without summaries...")
    
    def _build_search_indices(self):
        """Build vector and BM25 search indices."""
        if not self.dual_indexer:
            logger.error("Dual indexer not initialized")
            return
            
        if not self.summaries:
            logger.warning("No summaries available for indexing")
            return
            
        try:
            # Get function metadata
            function_metadata = {}
            if self.dependency_builder:
                function_metadata = self.dependency_builder.function_data
            
            # Build indices
            success = self.dual_indexer.index_summaries(
                self.summaries,
                function_metadata
            )
            
            if success:
                self.knowledge_index_built = True
                logger.info("Successfully built search indices")
                
                # Export index for backup
                index_file = self.repo_path / "search_index.json"
                self.dual_indexer.export_index(str(index_file))
            else:
                logger.error("Failed to build search indices")
                
        except Exception as e:
            logger.error(f"Failed to build search indices: {e}")
            logger.warning("Continuing without search indices...")
    
    def _initialize_retriever(self):
        """Initialize the RRF retriever for searches."""
        if not self.dual_indexer:
            logger.warning("Dual indexer not available, retriever not initialized")
            return
            
        try:
            self.retriever = RRFRetriever(self.dual_indexer)
            logger.info("Search retriever initialized and ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
    
    def _print_summary_statistics(self):
        """Print summary statistics of the enhanced processing."""
        logger.info("\n=== PROCESSING STATISTICS ===")
        
        # Original stats
        logger.info(f"Functions processed: {len(self.function_registry)}")
        logger.info(f"AST cache entries: {len(self.ast_cache)}")
        
        # Knowledge engine stats
        if self.summaries:
            successful_summaries = sum(1 for s in self.summaries.values() if s.success)
            logger.info(f"Summaries generated: {successful_summaries}/{len(self.summaries)}")
        
        if self.knowledge_index_built:
            logger.info("Search indices: ✓ Built")
        else:
            logger.info("Search indices: ✗ Not built")
            
        if self.retriever:
            stats = self.retriever.get_search_statistics()
            logger.info(f"Indexed documents: {stats['total_documents']}")
        
        logger.info("=============================\n")
    
    def search(self, query: str, limit: int = 10):
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results or None if retriever not available
        """
        if not self.retriever:
            logger.error("Retriever not initialized. Run the enhanced updater first.")
            return None
            
        return self.retriever.search(query, limit=limit)
    
    def get_function_summary(self, qualified_name: str):
        """Get summary for a specific function."""
        if qualified_name in self.summaries:
            return self.summaries[qualified_name]
        return None
    
    def export_knowledge_base(self, output_dir: Path = None):
        """Export the entire knowledge base to files."""
        if output_dir is None:
            output_dir = self.repo_path / "knowledge_base_export"
        
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Exporting knowledge base to {output_dir}")
        
        try:
            # Export summaries
            if self.summary_generator and self.summaries:
                summary_file = output_dir / "summaries.json"
                self.summary_generator.export_summaries(str(summary_file))
            
            # Export search index
            if self.dual_indexer and self.knowledge_index_built:
                index_file = output_dir / "search_index.json"
                self.dual_indexer.export_index(str(index_file))
            
            # Export dependency graph (as JSON)
            if self.dependency_builder:
                import networkx as nx
                graph_file = output_dir / "dependency_graph.json"
                graph_data = nx.node_link_data(self.dependency_builder.call_graph)
                
                import json
                with open(graph_file, 'w') as f:
                    json.dump(graph_data, f, indent=2)
            
            logger.info("Knowledge base export complete")
            
        except Exception as e:
            logger.error(f"Failed to export knowledge base: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.dual_indexer:
            self.dual_indexer.close()
        logger.info("Enhanced updater cleanup complete")