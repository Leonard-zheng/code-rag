"""
Command Line Query Interface for Code RAG System

Provides interactive and CLI-based access to the intelligent code search system.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    logger.warning("Rich library not installed. Install with: pip install rich")
    Console = None

from knowledge_engine.rrf_retriever import RRFRetriever, SearchResult
from knowledge_engine.dual_indexer import DualEngineIndexer
from knowledge_engine.answer_generator import AnswerGenerator


class QueryInterface:
    """Command-line interface for querying the code knowledge base."""
    
    def __init__(
        self,
        retriever: Optional[RRFRetriever] = None,
        answer_generator: Optional[AnswerGenerator] = None,
    ):
        """Initialize query interface."""
        self.retriever = retriever
        self.answer_generator = answer_generator
        
        # Initialize console for rich output
        if Console:
            self.console = Console()
        else:
            self.console = None
        
        # Check if retriever is available
        self.system_ready = retriever is not None
    
    def print_welcome(self):
        """Print welcome message."""
        if self.console:
            welcome_text = """
🔍 Code RAG - Intelligent Code Search System
===========================================

Features:
• 🧠 Semantic search with vector embeddings
• 🔤 Keyword search with BM25
• 🔗 Dependency-aware function analysis
• ⚡ Hybrid search with RRF fusion
"""
            self.console.print(Panel(welcome_text, title="Welcome", border_style="cyan"))
        else:
            print("=== Code RAG - Intelligent Code Search System ===")
            print("Features: Semantic search, Keyword search, Dependency analysis")
    
    def print_system_status(self):
        """Print system status information."""
        if not self.console:
            print(f"System Ready: {self.system_ready}")
            return
            
        if self.system_ready and self.retriever:
            stats = self.retriever.get_search_statistics()
            
            table = Table(title="System Status", border_style="green")
            table.add_column("Component", style="cyan", no_wrap=True)
            table.add_column("Status", style="green")
            table.add_column("Details", style="yellow")
            
            table.add_row("System", "✓ Ready", "All components loaded")
            table.add_row("Documents", "✓ Indexed", f"{stats['total_documents']} functions")
            table.add_row("Vector Search", "✓ Available" if stats['vector_index_available'] else "✗ Unavailable", "Weaviate")
            table.add_row("Keyword Search", "✓ Available" if stats['bm25_index_available'] else "✗ Unavailable", "BM25")
            
            # Complexity distribution
            complexity_dist = stats.get('complexity_distribution', {})
            complexity_str = ", ".join([f"{k}: {v}" for k, v in complexity_dist.items()])
            table.add_row("Complexity", "ℹ Distribution", complexity_str)
            
            self.console.print(table)
        else:
            self.console.print(Panel("❌ System Not Ready - Please run the enhanced updater first", 
                                   title="Status", border_style="red"))
    
    def format_search_results(self, results: List[SearchResult], query: str = "") -> None:
        """Format and display search results."""
        if not results:
            if self.console:
                self.console.print("🔍 No results found", style="yellow")
            else:
                print("No results found")
            return
        
        if self.console:
            # Rich formatting
            self.console.print(f"\n🔍 Found {len(results)} results for: '{query}'", style="bold cyan")
            
            for i, result in enumerate(results, 1):
                # Create result panel
                title = f"{i}. {result.function_name} ({result.complexity})"
                
                content = f"""
📍 {result.qualified_name}
📁 {result.file_path}
🎯 {result.summary}

💡 Purpose: {result.purpose}

🔗 Source: {result.source} | Score: {result.score:.4f}"""
                
                if result.matched_terms:
                    content += f"\n🏷️  Matched: {', '.join(result.matched_terms)}"
                
                # Color based on complexity
                color_map = {
                    "LOW": "green",
                    "MEDIUM": "yellow", 
                    "HIGH": "red",
                    "UNKNOWN": "white"
                }
                border_color = color_map.get(result.complexity, "white")
                
                self.console.print(Panel(content, title=title, border_style=border_color))
        else:
            # Plain text formatting
            print(f"\nFound {len(results)} results for: '{query}'")
            print("=" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.function_name} ({result.complexity})")
                print(f"   Location: {result.qualified_name}")
                print(f"   File: {result.file_path}")
                print(f"   Summary: {result.summary}")
                print(f"   Purpose: {result.purpose}")
                print(f"   Score: {result.score:.4f} | Source: {result.source}")
                if result.matched_terms:
                    print(f"   Matched: {', '.join(result.matched_terms)}")
                print("-" * 60)
    
    def search_command(self, query: str, limit: int = 10) -> bool:
        """Execute search command."""
        if not self.system_ready:
            if self.console:
                self.console.print("❌ System not ready. Please run the enhanced updater first.", style="red")
            else:
                print("System not ready. Please run the enhanced updater first.")
            return False
        
        try:
            if self.console:
                with self.console.status(f"[bold green]Searching for '{query}'..."):
                    results = self.retriever.search(query, limit=limit)
            else:
                print(f"Searching for: '{query}'...")
                results = self.retriever.search(query, limit=limit)
            
            self.format_search_results(results, query)

            # If an answer generator is available, produce a final answer using
            # the retrieved context and display it below the raw search results.
            if self.answer_generator:
                answer = self.answer_generator.generate_answer(query, results)
                if self.console:
                    self.console.print(Panel(answer, title="AI Answer", border_style="magenta"))
                else:
                    print("\nAI Answer:\n" + answer)

            return True
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            if self.console:
                self.console.print(f"❌ Search failed: {e}", style="red")
            else:
                print(f"Search failed: {e}")
            return False
    
    def similar_command(self, function_qn: str, limit: int = 5) -> bool:
        """Find functions similar to the given function."""
        if not self.system_ready:
            if self.console:
                self.console.print("❌ System not ready.", style="red")
            else:
                print("System not ready.")
            return False
        
        try:
            if self.console:
                with self.console.status(f"[bold green]Finding functions similar to '{function_qn}'..."):
                    results = self.retriever.search_similar_functions(function_qn, limit=limit)
            else:
                print(f"Finding similar functions to: '{function_qn}'...")
                results = self.retriever.search_similar_functions(function_qn, limit=limit)
            
            self.format_search_results(results, f"similar to {function_qn}")
            return True
            
        except Exception as e:
            logger.error(f"Similar search failed: {e}")
            if self.console:
                self.console.print(f"❌ Similar search failed: {e}", style="red")
            else:
                print(f"Similar search failed: {e}")
            return False
    
    def complexity_command(self, complexity: str, limit: int = 20) -> bool:
        """Find functions by complexity level."""
        if not self.system_ready:
            return False
        
        valid_complexity = ["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
        if complexity.upper() not in valid_complexity:
            if self.console:
                self.console.print(f"❌ Invalid complexity. Valid options: {valid_complexity}", style="red")
            else:
                print(f"Invalid complexity. Valid options: {valid_complexity}")
            return False
        
        try:
            results = self.retriever.search_by_complexity(complexity.upper(), limit=limit)
            self.format_search_results(results, f"complexity = {complexity.upper()}")
            return True
            
        except Exception as e:
            logger.error(f"Complexity search failed: {e}")
            return False
    
    def interactive_mode(self):
        """Start interactive query mode."""
        self.print_welcome()
        self.print_system_status()
        
        if not self.system_ready:
            return
        
        if self.console:
            self.console.print("\n💬 Interactive Mode - Type your queries below (or 'quit' to exit)", 
                             style="bold green")
            self.console.print("Commands: search <query>, similar <function_qn>, complexity <level>, status, help")
        else:
            print("\nInteractive Mode - Type your queries below (or 'quit' to exit)")
            print("Commands: search <query>, similar <function_qn>, complexity <level>, status, help")
        
        while True:
            try:
                if self.console:
                    user_input = Prompt.ask("\n🔍 Query", default="").strip()
                else:
                    user_input = input("\nQuery> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self.print_help()
                elif user_input.lower() == 'status':
                    self.print_system_status()
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        self.search_command(query)
                elif user_input.startswith('similar '):
                    func_qn = user_input[8:].strip()
                    if func_qn:
                        self.similar_command(func_qn)
                elif user_input.startswith('complexity '):
                    complexity = user_input[11:].strip()
                    if complexity:
                        self.complexity_command(complexity)
                else:
                    # Default to search
                    self.search_command(user_input)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Interactive mode error: {e}")
                if self.console:
                    self.console.print(f"❌ Error: {e}", style="red")
                else:
                    print(f"Error: {e}")
        
        if self.console:
            self.console.print("👋 Goodbye!", style="cyan")
        else:
            print("Goodbye!")
    
    def print_help(self):
        """Print help information."""
        help_text = """
Available Commands:
------------------
• search <query>          - Search for functions using hybrid search
• similar <function_qn>    - Find functions similar to the given function
• complexity <level>       - Find functions by complexity (LOW/MEDIUM/HIGH)
• status                   - Show system status
• help                     - Show this help message
• quit/exit/q             - Exit interactive mode

Examples:
---------
• search user authentication
• similar myproject.auth.login
• complexity HIGH
        """
        
        if self.console:
            self.console.print(Panel(help_text, title="Help", border_style="blue"))
        else:
            print(help_text)


def setup_logging():
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="INFO", format="<level>{level}</level> | <cyan>{message}</cyan>")


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Code RAG Query Interface")
    
    # Mode selection
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Start interactive query mode")
    
    # Direct query commands
    parser.add_argument("--search", "-s", type=str, 
                       help="Perform a search query")
    parser.add_argument("--similar", type=str, 
                       help="Find functions similar to the given qualified name")
    parser.add_argument("--complexity", "-c", type=str, 
                       help="Find functions by complexity level (LOW/MEDIUM/HIGH)")
    
    # Options
    parser.add_argument("--limit", "-l", type=int, default=10, 
                       help="Maximum number of results to return")
    parser.add_argument("--status", action="store_true", 
                       help="Show system status")
    
    # Configuration
    parser.add_argument("--weaviate-url", default="http://localhost:8080",
                       help="Weaviate database URL")
    parser.add_argument("--openai-api-key", 
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    return parser


def load_retriever(weaviate_url: str, openai_api_key: str) -> Optional[RRFRetriever]:
    """Load the retriever from existing indices."""
    try:
        # Initialize dual indexer
        indexer = DualEngineIndexer(
            weaviate_url=weaviate_url,
            openai_api_key=openai_api_key
        )
        
        # Check if indices exist and are populated
        if not indexer.document_store:
            logger.warning("No documents found in index. Please run the enhanced updater first.")
            return None
        
        # Create retriever
        retriever = RRFRetriever(indexer)
        logger.info(f"Loaded retriever with {len(indexer.document_store)} documents")
        
        return retriever
        
    except Exception as e:
        logger.error(f"Failed to load retriever: {e}")
        return None


def main():
    """Main entry point for the query interface."""
    setup_logging()
    
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Get OpenAI API key
    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("No OpenAI API key provided. Some features may not work.")
    
    # Load retriever
    logger.info("Loading search system...")
    retriever = load_retriever(args.weaviate_url, openai_api_key) if openai_api_key else None

    # Create answer generator if API key is available
    answer_generator = AnswerGenerator(openai_api_key) if openai_api_key else None

    # Create interface
    interface = QueryInterface(retriever, answer_generator)
    
    # Handle different modes
    if args.status:
        interface.print_system_status()
        
    elif args.search:
        interface.search_command(args.search, limit=args.limit)
        
    elif args.similar:
        interface.similar_command(args.similar, limit=args.limit)
        
    elif args.complexity:
        interface.complexity_command(args.complexity, limit=args.limit)
        
    elif args.interactive or not any([args.search, args.similar, args.complexity, args.status]):
        # Default to interactive mode
        interface.interactive_mode()
    
    # Cleanup
    if retriever and retriever.indexer:
        retriever.indexer.close()


if __name__ == "__main__":
    main()
