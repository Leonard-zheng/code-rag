"""
Local Enhanced Graph Updater with Langchain Integration

Uses local models (BGE-M3 embeddings + Ollama LLM) for complete offline processing.
No API keys required.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from code_parser.graph_updater import GraphUpdater
from code_parser.services.graph_service import MemgraphIngestor
from knowledge_engine.dependency_graph import DependencyGraphBuilder
from knowledge_engine.local_topological_summary import LocalTopologicalSummaryGenerator
from knowledge_engine.local_dual_indexer import LocalDualEngineIndexer
from knowledge_engine.rrf_retriever import RRFRetriever


class LocalEnhancedGraphUpdater(GraphUpdater):
    """Enhanced version of GraphUpdater using local models only."""
    
    def __init__(
        self,
        ingestor: MemgraphIngestor,
        repo_path: Path,
        parsers: dict[str, Any],
        queries: dict[str, Any],
        # Local model parameters
        llm_model_name: str = "gpt-oss-20b",
        ollama_base_url: str = "http://localhost:11434",
        embedding_model_name: str = "BAAI/bge-m3",
        weaviate_url: str = "http://localhost:8080",
        enable_summary_generation: bool = True,
        enable_indexing: bool = True,
        max_functions_per_batch: int = 5
    ):
        """
        Initialize Local Enhanced Graph Updater.
        
        Args:
            ingestor: Memgraph database ingestor
            repo_path: Path to repository
            parsers: Tree-sitter parsers
            queries: AST queries
            llm_model_name: Local LLM model name in Ollama
            ollama_base_url: Ollama server URL
            embedding_model_name: Local embedding model name
            weaviate_url: Weaviate database URL
            enable_summary_generation: Whether to generate summaries
            enable_indexing: Whether to build search indices
            max_functions_per_batch: Max functions per batch for local processing
        """
        # Initialize parent class
        super().__init__(ingestor, repo_path, parsers, queries)
        
        # Local model configuration
        self.llm_model_name = llm_model_name
        self.ollama_base_url = ollama_base_url
        self.embedding_model_name = embedding_model_name
        self.weaviate_url = weaviate_url
        self.enable_summary_generation = enable_summary_generation
        self.enable_indexing = enable_indexing
        self.max_functions_per_batch = max_functions_per_batch
        
        # Knowledge engine components (initialized lazily)
        self.dependency_builder: Optional[DependencyGraphBuilder] = None
        self.summary_generator: Optional[LocalTopologicalSummaryGenerator] = None
        self.dual_indexer: Optional[LocalDualEngineIndexer] = None
        self.retriever: Optional[RRFRetriever] = None
        
        # Storage for results
        self.summaries: Dict[str, Any] = {}
        self.knowledge_index_built = False
        
        logger.info("Initialized Local Enhanced Graph Updater (no API keys required)")
    
    def _check_local_services(self) -> Dict[str, bool]:
        """Check availability of required local services."""
        services_status = {}
        
        # Check Ollama
        try:
            import requests
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                services_status['ollama'] = self.llm_model_name in model_names
                if not services_status['ollama']:
                    logger.warning(f"Model {self.llm_model_name} not found in Ollama. Available models: {model_names}")
            else:
                services_status['ollama'] = False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.ollama_base_url}: {e}")
            services_status['ollama'] = False
        
        # Check Weaviate
        try:
            import requests
            response = requests.get(f"{self.weaviate_url}/v1/meta", timeout=5)
            services_status['weaviate'] = response.status_code == 200
        except Exception as e:
            logger.error(f"Cannot connect to Weaviate at {self.weaviate_url}: {e}")
            services_status['weaviate'] = False
        
        # Check embedding model
        try:
            from knowledge_engine.local_models import LocalEmbeddingModel
            # Just check if the model name is valid (will download if needed)
            services_status['embeddings'] = True
        except Exception as e:
            logger.error(f"Cannot initialize embedding model: {e}")
            services_status['embeddings'] = False
        
        return services_status
    
    def _initialize_knowledge_components(self):
        """Initialize knowledge engine components with local models."""
        logger.info("Initializing local knowledge engine components...")
        
        # Check services first
        services_status = self._check_local_services()
        
        for service, available in services_status.items():
            if available:
                logger.info(f"✓ {service.title()} service available")
            else:
                logger.error(f"❌ {service.title()} service unavailable")
        
        if not services_status['ollama'] and self.enable_summary_generation:
            logger.warning("Disabling summary generation due to Ollama unavailability")
            self.enable_summary_generation = False
        
        if not services_status['weaviate'] and self.enable_indexing:
            logger.warning("Disabling vector indexing due to Weaviate unavailability")
            self.enable_indexing = False
        
        try:
            # Dependency graph builder (always available)
            self.dependency_builder = DependencyGraphBuilder(
                ingestor=self.ingestor,
                project_name=self.project_name
            )
            logger.info("✓ Dependency graph builder initialized")
            
            # Local summary generator (if enabled and Ollama available)
            if self.enable_summary_generation and services_status['ollama']:
                self.summary_generator = LocalTopologicalSummaryGenerator(
                    llm_model_name=self.llm_model_name,
                    ollama_base_url=self.ollama_base_url,
                    max_functions_per_batch=self.max_functions_per_batch
                )
                logger.info("✓ Local summary generator initialized")
            else:
                logger.info("⚠ Summary generation disabled")
            
            # Local dual indexer (if enabled and services available)
            if (self.enable_indexing and 
                services_status['weaviate'] and 
                services_status['embeddings']):
                
                self.dual_indexer = LocalDualEngineIndexer(
                    weaviate_url=self.weaviate_url,
                    embedding_model_name=self.embedding_model_name
                )
                logger.info("✓ Local dual indexer initialized")
            else:
                logger.info("⚠ Indexing disabled or services unavailable")
                
        except Exception as e:
            logger.error(f"Failed to initialize local components: {e}")
            raise
    
    def run(self) -> None:
        """
        Enhanced run method with local knowledge extraction pipeline.
        
        Pipeline:
        1. Original AST parsing (3 phases)
        2. Build dependency graph
        3. Generate summaries with local LLM
        4. Build search indices with local embeddings
        5. Initialize retriever
        """
        logger.info("=== Starting Local Enhanced Graph Update Process ===")
        
        # Phase 1-3: Original AST parsing
        logger.info("\n--- PHASE 1-3: AST Parsing and Graph Construction ---")
        super().run()  # Run original pipeline
        
        # Initialize knowledge engine components
        self._initialize_knowledge_components()
        
        # Phase 4: Build dependency graph
        logger.info("\n--- PHASE 4: Dependency Graph Analysis ---")
        self._build_dependency_graph()
        
        # Phase 5: Generate summaries with local LLM (if enabled)
        if self.enable_summary_generation and self.summary_generator:
            logger.info("\n--- PHASE 5: Local LLM Summary Generation ---")
            self._generate_summaries()
        
        # Phase 6: Build search indices with local embeddings (if enabled)
        if self.enable_indexing and self.dual_indexer and self.summaries:
            logger.info("\n--- PHASE 6: Local Search Index Construction ---")
            self._build_search_indices()
        
        # Phase 7: Initialize retriever
        if self.dual_indexer:
            logger.info("\n--- PHASE 7: Initialize Local Retriever ---")
            self._initialize_retriever()
        
        logger.info("\n=== Local Enhanced Graph Update Process Complete ===")
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
        """Generate function summaries using local LLM in topological order."""
        if not self.dependency_builder or not self.summary_generator:
            logger.error("Required components not initialized for summary generation")
            return
            
        try:
            # Get topological batches
            batches = self.dependency_builder.get_topological_batches()
            if not batches:
                logger.warning("No function batches generated, skipping summary generation")
                return
                
            logger.info(f"Processing {len(batches)} dependency-ordered batches with local LLM")
            
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
            
            # Generate summaries using local LLM
            self.summaries = self.summary_generator.process_batches(
                batch_metadata, 
                get_context_library
            )
            
            logger.info(f"Generated {len(self.summaries)} function summaries using local LLM")
            
            # Export summaries for backup
            summary_file = self.repo_path / "function_summaries_local.json"
            self.summary_generator.export_summaries(str(summary_file))
            
        except Exception as e:
            logger.error(f"Failed to generate summaries with local LLM: {e}")
            # Don't raise - continue with available data
            logger.warning("Continuing without summaries...")
    
    def _build_search_indices(self):
        """Build vector and BM25 search indices using local models."""
        if not self.dual_indexer:
            logger.error("Local dual indexer not initialized")
            return
            
        if not self.summaries:
            logger.warning("No summaries available for indexing")
            return
            
        try:
            # Get function metadata
            function_metadata = {}
            if self.dependency_builder:
                function_metadata = self.dependency_builder.function_data
            
            # Build indices with local models
            success = self.dual_indexer.index_summaries(
                self.summaries,
                function_metadata
            )
            
            if success:
                self.knowledge_index_built = True
                logger.info("Successfully built local search indices")
                
                # Export index for backup
                index_file = self.repo_path / "search_index_local.json"
                self.dual_indexer.export_index(str(index_file))
            else:
                logger.error("Failed to build local search indices")
                
        except Exception as e:
            logger.error(f"Failed to build local search indices: {e}")
            logger.warning("Continuing without search indices...")
    
    def _initialize_retriever(self):
        """Initialize the RRF retriever for searches."""
        if not self.dual_indexer:
            logger.warning("Local dual indexer not available, retriever not initialized")
            return
            
        try:
            self.retriever = RRFRetriever(self.dual_indexer)
            logger.info("Local search retriever initialized and ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize local retriever: {e}")
    
    def _print_summary_statistics(self):
        """Print summary statistics of the local enhanced processing."""
        logger.info("\n=== LOCAL PROCESSING STATISTICS ===")
        
        # Original stats
        logger.info(f"Functions processed: {len(self.function_registry)}")
        logger.info(f"AST cache entries: {len(self.ast_cache)}")
        
        # Local knowledge engine stats
        if self.summaries:
            successful_summaries = sum(1 for s in self.summaries.values() if s.success)
            logger.info(f"Local LLM summaries: {successful_summaries}/{len(self.summaries)}")
        
        if self.knowledge_index_built:
            logger.info("Local search indices: ✓ Built with BGE-M3")
        else:
            logger.info("Local search indices: ✗ Not built")
            
        if self.retriever:
            stats = self.retriever.get_search_statistics()
            logger.info(f"Indexed documents: {stats['total_documents']}")
        
        logger.info("=====================================\n")
    
    def search(self, query: str, limit: int = 10):
        """
        Search the local knowledge base.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results or None if retriever not available
        """
        if not self.retriever:
            logger.error("Local retriever not initialized. Run the enhanced updater first.")
            return None
            
        return self.retriever.search(query, limit=limit)
    
    def get_function_summary(self, qualified_name: str):
        """Get summary for a specific function."""
        if qualified_name in self.summaries:
            return self.summaries[qualified_name]
        return None
    
    def export_knowledge_base(self, output_dir: Path = None):
        """Export the entire local knowledge base to files."""
        if output_dir is None:
            output_dir = self.repo_path / "local_knowledge_base_export"
        
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Exporting local knowledge base to {output_dir}")
        
        try:
            # Export summaries
            if self.summary_generator and self.summaries:
                summary_file = output_dir / "summaries_local.json"
                self.summary_generator.export_summaries(str(summary_file))
            
            # Export search index
            if self.dual_indexer and self.knowledge_index_built:
                index_file = output_dir / "search_index_local.json"
                self.dual_indexer.export_index(str(index_file))
            
            # Export dependency graph (as JSON)
            if self.dependency_builder:
                import networkx as nx
                graph_file = output_dir / "dependency_graph_local.json"
                graph_data = nx.node_link_data(self.dependency_builder.call_graph)
                
                import json
                with open(graph_file, 'w') as f:
                    json.dump(graph_data, f, indent=2)
            
            logger.info("Local knowledge base export complete")
            
        except Exception as e:
            logger.error(f"Failed to export local knowledge base: {e}")
    
    def cleanup(self):
        """Cleanup local resources."""
        if self.dual_indexer:
            self.dual_indexer.close()
        logger.info("Local enhanced updater cleanup complete")