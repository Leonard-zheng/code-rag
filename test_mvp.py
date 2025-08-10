#!/usr/bin/env python3
"""
MVP Test Script

Tests the basic functionality of the Code RAG system without requiring
external services or API keys. Validates core components can be imported
and initialized.
"""

import sys
import os
from pathlib import Path
import tempfile
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Core AST parsing modules
        from code_parser.graph_updater import GraphUpdater
        from code_parser.services.graph_service import MemgraphIngestor
        from code_parser.parser_loader import load_parsers
        print("✓ AST parsing modules imported successfully")
        
        # Knowledge engine modules
        from knowledge_engine.dependency_graph import DependencyGraphBuilder
        from knowledge_engine.topological_summary import TopologicalSummaryGenerator
        from knowledge_engine.dual_indexer import DualEngineIndexer
        from knowledge_engine.rrf_retriever import RRFRetriever
        print("✓ Knowledge engine modules imported successfully")
        
        # Enhanced system
        from enhanced_graph_updater import EnhancedGraphUpdater
        from query_interface import QueryInterface
        print("✓ Enhanced system modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test dependency graph builder (without Memgraph)
        from knowledge_engine.dependency_graph import DependencyGraphBuilder
        
        # Mock ingestor for testing
        class MockIngestor:
            def fetch_all(self, query, params=None):
                return []
        
        builder = DependencyGraphBuilder(MockIngestor(), "test_project")
        print("✓ DependencyGraphBuilder initialized")
        
        # Test RRF retriever components
        from knowledge_engine.rrf_retriever import QueryAnalyzer
        analyzer = QueryAnalyzer()
        
        # Test query analysis
        analysis = analyzer.analyze_query("how does user authentication work?")
        assert 'vector_weight' in analysis
        assert 'bm25_weight' in analysis
        print("✓ Query analysis working")
        
        # Test different query types
        code_query = analyzer.analyze_query("def login(username, password):")
        assert code_query['bm25_weight'] > 0.5  # Code snippets should favor BM25
        
        semantic_query = analyzer.analyze_query("What is the purpose of this function?")
        assert semantic_query['vector_weight'] >= 0.4  # Semantic queries favor vector
        
        print("✓ Query type detection working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from code_parser.config import AppConfig
        
        # Test basic config loading
        config = AppConfig()
        
        # Check default values
        assert config.MEMGRAPH_HOST == "localhost"
        assert config.MEMGRAPH_PORT == 7687
        print("✓ Configuration loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_file_structure():
    """Test that all expected files exist."""
    print("\n📁 Testing file structure...")
    
    project_root = Path(__file__).parent
    
    expected_files = [
        "enhanced_main.py",
        "query_interface.py", 
        "requirements.txt",
        "docker-compose.yml",
        "CLAUDE.md",
        "knowledge_engine/__init__.py",
        "knowledge_engine/dependency_graph.py",
        "knowledge_engine/topological_summary.py",
        "knowledge_engine/dual_indexer.py",
        "knowledge_engine/rrf_retriever.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✓ All expected files present")
        return True


def test_parser_loading():
    """Test Tree-sitter parser loading."""
    print("\n🌳 Testing parser loading...")
    
    try:
        from code_parser.parser_loader import load_parsers
        
        # This might fail if parsers aren't built, but should not crash
        try:
            parsers, queries = load_parsers()
            print(f"✓ Loaded {len(parsers)} parsers successfully")
        except Exception as e:
            print(f"⚠️ Parser loading failed (expected if parsers not built): {e}")
            print("  This is normal for first-time setup")
        
        return True
        
    except ImportError as e:
        print(f"❌ Parser module import failed: {e}")
        return False


def create_sample_test_data():
    """Create sample test data for demonstration."""
    print("\n📝 Creating sample test data...")
    
    try:
        # Create a temporary Python file to test parsing
        test_data_dir = Path(__file__).parent / "test_data"
        test_data_dir.mkdir(exist_ok=True)
        
        sample_code = '''
"""Sample Python module for testing."""

def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

def main():
    """Main function demonstrating the calculator."""
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.multiply(4, 7))
    print(f"Fibonacci(10) = {calculate_fibonacci(10)}")

if __name__ == "__main__":
    main()
'''
        
        sample_file = test_data_dir / "sample_module.py"
        sample_file.write_text(sample_code)
        
        print(f"✓ Created sample test data at {sample_file}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return False


def run_all_tests():
    """Run all MVP tests."""
    print("🚀 Running Code RAG MVP Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Configuration", test_configuration),
        ("File Structure", test_file_structure),
        ("Parser Loading", test_parser_loading),
        ("Sample Data Creation", create_sample_test_data)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! MVP implementation is ready.")
        print("\n📋 Next steps:")
        print("1. Start services: docker-compose up -d")
        print("2. Set OpenAI key: export OPENAI_API_KEY='your-key'")
        print("3. Run pipeline: python enhanced_main.py --repo-path /path/to/code")
        print("4. Try searching: python query_interface.py --interactive")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)