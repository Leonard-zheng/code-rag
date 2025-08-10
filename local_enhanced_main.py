"""
Local Enhanced Main Entry Point

Runs the complete Code RAG pipeline using local models only:
1. AST parsing and graph construction
2. Dependency analysis
3. Topological summary generation with local LLM (Ollama)
4. Search index construction with local embeddings (BGE-M3)
5. Local query interface

No API keys required!
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Add current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from local_enhanced_graph_updater import LocalEnhancedGraphUpdater
from code_parser.parser_loader import load_parsers
from code_parser.services.graph_service import MemgraphIngestor
from query_interface import QueryInterface


def check_local_requirements():
    """Check if all required local services are available."""
    issues = []
    
    # Check Memgraph (attempt connection)
    try:
        with MemgraphIngestor("localhost", 7687) as ingestor:
            # Test connection
            ingestor.fetch_all("MATCH (n) RETURN count(n) as count LIMIT 1")
            logger.info("✓ Memgraph connection successful")
    except Exception as e:
        issues.append(f"❌ Memgraph connection failed: {e}")
        issues.append("   Make sure Memgraph is running:")
        issues.append("   docker run -p 7687:7687 -p 7444:7444 -p 3000:3000 memgraph/memgraph-platform")
    
    # Check Weaviate
    try:
        import requests
        response = requests.get("http://localhost:8080/v1/meta", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Weaviate connection successful")
        else:
            issues.append(f"❌ Weaviate returned status {response.status_code}")
    except requests.RequestException as e:
        issues.append(f"❌ Weaviate connection failed: {e}")
        issues.append("   Make sure Weaviate is running:")
        issues.append("   docker run -p 8080:8080 semitechnologies/weaviate:latest")
    except ImportError:
        issues.append("❌ requests library not installed: pip install requests")
    
    # Check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            logger.info(f"✓ Ollama connection successful. Available models: {model_names}")
            
            # Check for GPT OSS 20B or similar models
            suggested_models = ['gpt-oss-20b', 'llama2', 'codellama', 'mistral', 'qwen']
            available_suggested = [m for m in suggested_models if m in model_names]
            
            if not available_suggested:
                issues.append(f"⚠️ No recommended models found. Available: {model_names}")
                issues.append("   Consider: ollama pull gpt-oss-20b (or llama2, codellama, mistral)")
        else:
            issues.append(f"❌ Ollama returned status {response.status_code}")
    except requests.RequestException as e:
        issues.append(f"❌ Ollama connection failed: {e}")
        issues.append("   Make sure Ollama is running:")
        issues.append("   1. Install Ollama: https://ollama.ai/")
        issues.append("   2. Start service: ollama serve")
        issues.append("   3. Pull model: ollama pull gpt-oss-20b")
    
    # Check local embedding dependencies
    try:
        import sentence_transformers
        logger.info("✓ Sentence transformers available")
    except ImportError:
        issues.append("❌ sentence-transformers not installed: pip install sentence-transformers")
    
    try:
        import torch
        logger.info(f"✓ PyTorch available: {torch.__version__}")
    except ImportError:
        issues.append("❌ PyTorch not installed: pip install torch")
    
    try:
        import langchain
        logger.info(f"✓ Langchain available: {langchain.__version__}")
    except ImportError:
        issues.append("❌ Langchain not installed: pip install langchain langchain-community")
    
    return issues


def run_local_enhanced_pipeline(
    repo_path: str,
    clean_db: bool = True,
    enable_summaries: bool = True,
    enable_indexing: bool = True,
    llm_model_name: str = "gpt-oss-20b",
    embedding_model_name: str = "BAAI/bge-m3"
):
    """
    Run the complete enhanced Code RAG pipeline using local models.
    
    Args:
        repo_path: Path to the repository to analyze
        clean_db: Whether to clean the database before processing
        enable_summaries: Whether to generate LLM summaries
        enable_indexing: Whether to build search indices
        llm_model_name: Local LLM model name in Ollama
        embedding_model_name: Local embedding model name
    """
    logger.info("=== Starting Local Enhanced Code RAG Pipeline ===")
    logger.info("Using 100% local models - no API keys required!")
    
    # Check local system requirements
    logger.info("Checking local system requirements...")
    issues = check_local_requirements()
    
    if issues:
        logger.error("Local system requirements not met:")
        for issue in issues:
            if issue.startswith("   "):
                logger.info(issue)  # Instructions in normal color
            else:
                logger.error(issue)  # Errors in error color
        
        if not enable_summaries and not enable_indexing:
            logger.info("Continuing with AST parsing only...")
        else:
            logger.error("Cannot continue with knowledge engine features. Fix issues above.")
            return False
    
    # Load parsers
    logger.info("Loading Tree-sitter parsers...")
    parsers, queries = load_parsers()
    logger.info(f"Loaded {len(parsers)} language parsers")
    
    # Setup database connection
    memgraph_host = "localhost"
    memgraph_port = 7687
    weaviate_url = "http://localhost:8080"
    ollama_base_url = "http://localhost:11434"
    
    # Run local enhanced pipeline
    with MemgraphIngestor(host=memgraph_host, port=memgraph_port) as ingestor:
        if clean_db:
            logger.info("Cleaning database...")
            ingestor.clean_database()
        
        # Ensure database constraints
        ingestor.ensure_constraints()
        
        # Initialize local enhanced updater
        logger.info(f"Initializing Local Enhanced Graph Updater with {llm_model_name}...")
        updater = LocalEnhancedGraphUpdater(
            ingestor=ingestor,
            repo_path=Path(repo_path),
            parsers=parsers,
            queries=queries,
            llm_model_name=llm_model_name,
            ollama_base_url=ollama_base_url,
            embedding_model_name=embedding_model_name,
            weaviate_url=weaviate_url,
            enable_summary_generation=enable_summaries,
            enable_indexing=enable_indexing
        )
        
        try:
            # Run the complete pipeline
            updater.run()
            
            # Export knowledge base
            logger.info("Exporting local knowledge base...")
            updater.export_knowledge_base()
            
            logger.info("=== Local Pipeline Complete ===")
            
            # Test the search system
            if updater.retriever:
                logger.info("\n=== Testing Local Search System ===")
                test_query = "用户认证"  # Chinese test query
                results = updater.search(test_query, limit=3)
                if results:
                    logger.info(f"✓ Local search test successful: found {len(results)} results for '{test_query}'")
                else:
                    logger.warning(f"⚠ Local search test returned no results for '{test_query}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Local pipeline failed: {e}")
            return False
        finally:
            # Cleanup
            updater.cleanup()


def interactive_mode():
    """Start interactive query mode using local models."""
    logger.info("Starting local interactive query interface...")
    
    # Load local retriever
    from knowledge_engine.local_dual_indexer import LocalDualEngineIndexer
    from knowledge_engine.rrf_retriever import RRFRetriever
    
    try:
        # Initialize local dual indexer
        indexer = LocalDualEngineIndexer(
            weaviate_url="http://localhost:8080",
            embedding_model_name="BAAI/bge-m3"
        )
        
        # Check if indices exist and are populated
        if not indexer.document_store:
            logger.warning("No documents found in local index. Please run the pipeline first:")
            logger.warning("python local_enhanced_main.py --repo-path /path/to/your/code")
            return
        
        # Create retriever
        retriever = RRFRetriever(indexer)
        logger.info(f"Loaded local retriever with {len(indexer.document_store)} documents")
        
        # Start interface
        interface = QueryInterface(retriever)
        interface.interactive_mode()
        
        # Cleanup
        indexer.close()
        
    except Exception as e:
        logger.error(f"Failed to start local interactive mode: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Local Enhanced Code RAG System")
    parser.add_argument("--repo-path", "-r", type=str, 
                       default="/Users/zhengzilong/code-graph/code-graph-rag",
                       help="Path to repository to analyze")
    parser.add_argument("--clean-db", action="store_true", default=True,
                       help="Clean database before processing")
    parser.add_argument("--no-clean-db", action="store_false", dest="clean_db",
                       help="Don't clean database")
    parser.add_argument("--no-summaries", action="store_false", dest="enable_summaries",
                       default=True, help="Skip LLM summary generation")
    parser.add_argument("--no-indexing", action="store_false", dest="enable_indexing", 
                       default=True, help="Skip search index construction")
    parser.add_argument("--llm-model", type=str, default="gpt-oss-20b",
                       help="Ollama LLM model name (default: gpt-oss-20b)")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3",
                       help="Local embedding model name (default: BAAI/bge-m3)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Start interactive query mode (skip pipeline)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", 
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    if args.interactive:
        interactive_mode()
    else:
        success = run_local_enhanced_pipeline(
            repo_path=args.repo_path,
            clean_db=args.clean_db,
            enable_summaries=args.enable_summaries,
            enable_indexing=args.enable_indexing,
            llm_model_name=args.llm_model,
            embedding_model_name=args.embedding_model
        )
        
        if success:
            logger.info("\n🎉 Local pipeline completed successfully!")
            logger.info("You can now use the local query interface:")
            logger.info("  python local_enhanced_main.py --interactive")
            logger.info("\nOr use the query interface directly:")
            logger.info("  python query_interface.py --interactive")
        else:
            logger.error("Local pipeline failed. Check the logs above for details.")
            sys.exit(1)