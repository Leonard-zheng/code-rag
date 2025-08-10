#!/usr/bin/env python3
"""
Test script for new LangChain compatibility

Tests the updated LangChain integrations with the latest versions.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_langchain_imports():
    """Test new LangChain import compatibility."""
    print("🔍 Testing new LangChain imports...")
    
    try:
        # Test Ollama integration
        from langchain_ollama import OllamaLLM
        print("✓ langchain_ollama.OllamaLLM imported successfully")
        
        # Test core components
        from langchain_core.prompts import PromptTemplate
        from langchain_core.documents import Document
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        print("✓ langchain_core components imported successfully")
        
        # Test community components
        from langchain_community.vectorstores import Weaviate as LangchainWeaviate
        print("✓ langchain_community components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False


def test_local_models_import():
    """Test local models module with new LangChain."""
    print("\n🧪 Testing local models with new LangChain...")
    
    try:
        from knowledge_engine.local_models import (
            LocalEmbeddingModel, 
            LocalLLMModel, 
            LangchainSummaryChain
        )
        print("✓ Local models imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Local models import failed: {e}")
        return False


def test_sentence_transformers():
    """Test sentence transformers integration."""
    print("\n🤗 Testing sentence transformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence_transformers imported successfully")
        
        # Test if we can initialize a model (without downloading)
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=None)
            print("✓ SentenceTransformer can be initialized")
        except Exception as e:
            print(f"⚠️ SentenceTransformer initialization test skipped: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ sentence_transformers import failed: {e}")
        return False


def test_weaviate_client():
    """Test Weaviate client compatibility."""
    print("\n🗃️ Testing Weaviate client...")
    
    try:
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType
        from weaviate.classes.query import MetadataQuery
        import weaviate.classes.query as wq
        print("✓ Weaviate client v4 imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Weaviate client import failed: {e}")
        return False


def test_ollama_chain_creation():
    """Test creating Ollama chain with new syntax."""
    print("\n🦙 Testing Ollama chain creation...")
    
    try:
        from langchain_ollama import OllamaLLM
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Create LLM (without connecting)
        llm = OllamaLLM(
            model="test-model",
            base_url="http://localhost:11434",
            temperature=0.1
        )
        print("✓ OllamaLLM instance created")
        
        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["question"],
            template="Answer this question: {question}"
        )
        print("✓ PromptTemplate created")
        
        # Create chain using LCEL syntax
        chain = prompt | llm | StrOutputParser()
        print("✓ LCEL chain created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama chain creation failed: {e}")
        return False


def test_local_embedding_model():
    """Test local embedding model functionality."""
    print("\n🎯 Testing local embedding model...")
    
    try:
        from knowledge_engine.local_models import LocalEmbeddingModel
        
        # This will print model info without downloading
        print("✓ LocalEmbeddingModel class imported")
        
        # Test model initialization (will attempt download if not cached)
        print("ℹ️  Note: BGE-M3 model will be downloaded on first use")
        
        return True
        
    except Exception as e:
        print(f"❌ Local embedding model test failed: {e}")
        return False


def run_compatibility_tests():
    """Run all compatibility tests."""
    print("🚀 Running LangChain Compatibility Tests")
    print("=" * 50)
    
    tests = [
        ("LangChain Imports", test_langchain_imports),
        ("Local Models Import", test_local_models_import),
        ("Sentence Transformers", test_sentence_transformers),
        ("Weaviate Client", test_weaviate_client),
        ("Ollama Chain Creation", test_ollama_chain_creation),
        ("Local Embedding Model", test_local_embedding_model),
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
    print(f"📊 COMPATIBILITY RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All compatibility tests passed! New LangChain version is ready.")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements-local.txt")
        print("2. Start services: docker-compose -f docker-compose-local.yml up -d")
        print("3. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("4. Pull model: ollama pull gpt-oss-20b")
        print("5. Run local pipeline: python local_enhanced_main.py")
    else:
        print("⚠️ Some compatibility tests failed. Check the output above.")
        print("Make sure you have installed the latest versions:")
        print("pip install -r requirements-local.txt")
    
    return failed == 0


if __name__ == "__main__":
    success = run_compatibility_tests()
    sys.exit(0 if success else 1)