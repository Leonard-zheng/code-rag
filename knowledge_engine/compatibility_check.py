"""
Compatibility Check Module

Checks for required dependencies and their versions for the local model setup.
Provides helpful error messages and suggestions for missing packages.
"""

import sys
from typing import Dict, List, Tuple
from loguru import logger


def check_package_version(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and optionally verify its version.
    
    Returns:
        Tuple of (is_available, version_or_error_message)
    """
    try:
        import importlib.metadata
        version = importlib.metadata.version(package_name)
        
        if min_version:
            from packaging import version as pkg_version
            if pkg_version.parse(version) >= pkg_version.parse(min_version):
                return True, version
            else:
                return False, f"Version {version} < required {min_version}"
        
        return True, version
        
    except importlib.metadata.PackageNotFoundError:
        return False, "Package not installed"
    except Exception as e:
        return False, f"Error checking package: {e}"


def check_langchain_compatibility() -> Dict[str, bool]:
    """Check LangChain components compatibility."""
    results = {}
    
    # Required LangChain packages
    langchain_packages = [
        ("langchain", "0.3.0"),
        ("langchain-community", "0.3.0"),
        ("langchain-ollama", "0.3.0"),
        ("langchain-core", "0.3.0"),
    ]
    
    for package, min_version in langchain_packages:
        is_available, info = check_package_version(package, min_version)
        results[package] = is_available
        
        if is_available:
            logger.info(f"✓ {package}: {info}")
        else:
            logger.error(f"❌ {package}: {info}")
    
    return results


def check_ml_dependencies() -> Dict[str, bool]:
    """Check ML/AI dependencies."""
    results = {}
    
    # Required ML packages
    ml_packages = [
        ("torch", "2.0.0"),
        ("transformers", "4.20.0"),
        ("sentence-transformers", "2.0.0"),
    ]
    
    for package, min_version in ml_packages:
        is_available, info = check_package_version(package, min_version)
        results[package] = is_available
        
        if is_available:
            logger.info(f"✓ {package}: {info}")
        else:
            logger.error(f"❌ {package}: {info}")
    
    return results


def check_vector_db_dependencies() -> Dict[str, bool]:
    """Check vector database dependencies."""
    results = {}
    
    # Required vector DB packages
    vector_packages = [
        ("weaviate-client", "4.0.0"),
        ("rank-bm25", "0.2.0"),
    ]
    
    for package, min_version in vector_packages:
        is_available, info = check_package_version(package, min_version)
        results[package] = is_available
        
        if is_available:
            logger.info(f"✓ {package}: {info}")
        else:
            logger.error(f"❌ {package}: {info}")
    
    return results


def check_core_dependencies() -> Dict[str, bool]:
    """Check core project dependencies."""
    results = {}
    
    # Core packages
    core_packages = [
        ("loguru", "0.7.0"),
        ("networkx", "3.0.0"),
        ("rich", "13.0.0"),
        ("requests", "2.28.0"),
        ("pydantic", "2.0.0"),
    ]
    
    for package, min_version in core_packages:
        is_available, info = check_package_version(package, min_version)
        results[package] = is_available
        
        if is_available:
            logger.info(f"✓ {package}: {info}")
        else:
            logger.error(f"❌ {package}: {info}")
    
    return results


def get_installation_suggestions(missing_packages: List[str]) -> str:
    """Generate installation suggestions for missing packages."""
    suggestions = []
    
    # Group packages by installation method
    langchain_packages = [
        "langchain", "langchain-community", "langchain-ollama", "langchain-core"
    ]
    
    ml_packages = ["torch", "transformers", "sentence-transformers"]
    vector_packages = ["weaviate-client", "rank-bm25"]
    
    langchain_missing = [p for p in missing_packages if p in langchain_packages]
    ml_missing = [p for p in missing_packages if p in ml_packages]
    vector_missing = [p for p in missing_packages if p in vector_packages]
    core_missing = [p for p in missing_packages if p not in langchain_packages + ml_packages + vector_packages]
    
    if langchain_missing:
        suggestions.append("# Install LangChain components:")
        suggestions.append("pip install " + " ".join(langchain_missing))
    
    if ml_missing:
        suggestions.append("# Install ML/AI dependencies:")
        if "torch" in ml_missing:
            suggestions.append("# For PyTorch, visit https://pytorch.org/get-started/locally/")
            suggestions.append("pip install torch --index-url https://download.pytorch.org/whl/cpu")
        suggestions.append("pip install " + " ".join([p for p in ml_missing if p != "torch"]))
    
    if vector_missing:
        suggestions.append("# Install vector database dependencies:")
        suggestions.append("pip install " + " ".join(vector_missing))
    
    if core_missing:
        suggestions.append("# Install core dependencies:")
        suggestions.append("pip install " + " ".join(core_missing))
    
    if suggestions:
        suggestions.insert(0, "\n📦 Installation suggestions:")
        suggestions.append("\n# Or install everything at once:")
        suggestions.append("pip install -r requirements-local.txt")
    
    return "\n".join(suggestions)


def run_full_compatibility_check() -> bool:
    """Run full compatibility check and provide suggestions."""
    logger.info("🔍 Running compatibility check for local model setup...")
    
    all_results = {}
    
    # Check all dependency categories
    logger.info("\n--- Core Dependencies ---")
    all_results.update(check_core_dependencies())
    
    logger.info("\n--- LangChain Components ---")  
    all_results.update(check_langchain_compatibility())
    
    logger.info("\n--- ML/AI Dependencies ---")
    all_results.update(check_ml_dependencies())
    
    logger.info("\n--- Vector Database Dependencies ---")
    all_results.update(check_vector_db_dependencies())
    
    # Summarize results
    total_packages = len(all_results)
    available_packages = sum(all_results.values())
    missing_packages = [pkg for pkg, available in all_results.items() if not available]
    
    logger.info(f"\n📊 Summary: {available_packages}/{total_packages} packages available")
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        suggestions = get_installation_suggestions(missing_packages)
        print(suggestions)
        return False
    else:
        logger.info("🎉 All required packages are available!")
        return True


if __name__ == "__main__":
    success = run_full_compatibility_check()
    sys.exit(0 if success else 1)