"""
Enhanced Main Entry Point

Runs the complete Code RAG pipeline:
1. AST parsing and graph construction
2. Dependency analysis
3. Topological summary generation
4. Search index construction
5. Query interface initialization
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Add current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from enhanced_graph_updater import EnhancedGraphUpdater
from code_parser.parser_loader import load_parsers
from code_parser.services.graph_service import MemgraphIngestor
from query_interface import QueryInterface, load_retriever


def check_requirements():
    """Check if all required services and API keys are available."""
    issues = []
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        issues.append("❌ OPENAI_API_KEY environment variable not set")
    else:
        logger.info("✓ OpenAI API key found")
    
    # Check Memgraph (attempt connection)
    try:
        with MemgraphIngestor("localhost", 7687) as ingestor:
            # Test connection
            ingestor.fetch_all("MATCH (n) RETURN count(n) as count LIMIT 1")
            logger.info("✓ Memgraph connection successful")
    except Exception as e:
        issues.append(f"❌ Memgraph connection failed: {e}")
        issues.append("   Make sure Memgraph is running: docker-compose up -d")
    
    # Check Weaviate (basic connection test)
    try:
        import requests
        response = requests.get("http://localhost:8080/v1/meta", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Weaviate connection successful")
        else:
            issues.append(f"❌ Weaviate returned status {response.status_code}")
    except requests.RequestException as e:
        issues.append(f"❌ Weaviate connection failed: {e}")
        issues.append("   Make sure Weaviate is running: docker run -p 8080:8080 semitechnologies/weaviate:latest")
    except ImportError:
        issues.append("❌ requests library not installed: pip install requests")
    
    return issues


def run_enhanced_pipeline(
    repo_path: str,
    clean_db: bool = True,
    enable_summaries: bool = True,
    enable_indexing: bool = True
):
    """
    Run the complete enhanced Code RAG pipeline.
    
    Args:
        repo_path: Path to the repository to analyze
        clean_db: Whether to clean the database before processing
        enable_summaries: Whether to generate LLM summaries
        enable_indexing: Whether to build search indices
    """
    logger.info("=== Starting Enhanced Code RAG Pipeline ===")
    
    # Check system requirements
    logger.info("Checking system requirements...")
    issues = check_requirements()
    
    if issues:
        logger.error("System requirements not met:")
        for issue in issues:
            logger.error(issue)
        
        if not enable_summaries and not enable_indexing:
            logger.info("Continuing with AST parsing only...")
        else:
            logger.error("Cannot continue with knowledge engine features. Aborting.")
            return False
    
    # Load parsers
    logger.info("Loading Tree-sitter parsers...")
    parsers, queries = load_parsers()
    logger.info(f"Loaded {len(parsers)} language parsers")
    
    # Setup database connection
    memgraph_host = "localhost"
    memgraph_port = 7687
    weaviate_url = "http://localhost:8080"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Run enhanced pipeline
    with MemgraphIngestor(host=memgraph_host, port=memgraph_port) as ingestor:
        if clean_db:
            logger.info("Cleaning database...")
            ingestor.clean_database()
        
        # Ensure database constraints
        ingestor.ensure_constraints()
        
        # Initialize enhanced updater
        logger.info("Initializing Enhanced Graph Updater...")
        updater = EnhancedGraphUpdater(
            ingestor=ingestor,
            repo_path=Path(repo_path),
            parsers=parsers,
            queries=queries,
            openai_api_key=openai_api_key,
            weaviate_url=weaviate_url,
            enable_summary_generation=enable_summaries and openai_api_key is not None,
            enable_indexing=enable_indexing and openai_api_key is not None
        )
        
        try:
            # Run the complete pipeline
            updater.run()
            
            # Export knowledge base
            logger.info("Exporting knowledge base...")
            updater.export_knowledge_base()
            
            logger.info("=== Pipeline Complete ===")
            
            # Test the search system
            if updater.retriever:
                logger.info("\n=== Testing Search System ===")
                test_query = "authentication"
                results = updater.search(test_query, limit=3)
                if results:
                    logger.info(f"✓ Search test successful: found {len(results)} results for '{test_query}'")
                else:
                    logger.warning(f"⚠ Search test returned no results for '{test_query}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
        finally:
            # Cleanup
            updater.cleanup()


def interactive_mode():
    """Start interactive query mode."""
    logger.info("Starting interactive query interface...")
    
    # Load retriever
    openai_api_key = os.getenv("OPENAI_API_KEY")
    weaviate_url = "http://localhost:8080"
    
    retriever = load_retriever(weaviate_url, openai_api_key) if openai_api_key else None
    
    # Start interface
    interface = QueryInterface(retriever)
    interface.interactive_mode()
    
    # Cleanup
    if retriever and retriever.indexer:
        retriever.indexer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Code RAG System")
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
        success = run_enhanced_pipeline(
            repo_path=args.repo_path,
            clean_db=args.clean_db,
            enable_summaries=args.enable_summaries,
            enable_indexing=args.enable_indexing
        )
        
        if success:
            logger.info("\n🎉 Pipeline completed successfully!")
            logger.info("You can now use the query interface:")
            logger.info("  python query_interface.py --interactive")
        else:
            logger.error("Pipeline failed. Check the logs above for details.")
            sys.exit(1)