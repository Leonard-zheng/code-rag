# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an intelligent Code RAG (Retrieval-Augmented Generation) system that combines AST parsing with AI-powered knowledge extraction and hybrid search capabilities. The system:

1. **Parses codebases** using Tree-sitter to build knowledge graphs in Memgraph
2. **Generates intelligent summaries** of functions using dependency-aware topological processing  
3. **Builds hybrid search indices** combining vector embeddings (Weaviate) and keyword search (BM25)
4. **Provides intelligent code search** with semantic understanding and contextual retrieval

The system transforms raw code into searchable, semantically-rich knowledge that can answer complex queries about code functionality, relationships, and purpose.

## Architecture

The project follows a modular architecture with distinct processing phases:

### Core Components

- **GraphUpdater** (`code_parser/graph_updater.py`): Main orchestrator that coordinates the parsing process through three phases:
  1. Structure identification (packages, folders)
  2. File processing and AST caching with definition collection
  3. Function call processing using cached ASTs

- **ProcessorFactory** (`code_parser/parsers/factory.py`): Dependency injection container that creates processor instances with proper dependencies

- **MemgraphIngestor** (`code_parser/services/graph_service.py`): Handles all database communication with batching and buffering for performance

### Processor Pipeline

1. **StructureProcessor**: Identifies packages, folders, and project structure
2. **DefinitionProcessor**: Extracts functions, classes, methods from source files
3. **CallProcessor**: Analyzes function calls and method invocations
4. **ImportProcessor**: Tracks import relationships between modules
5. **TypeInferenceEngine**: Infers types for better call resolution

### Key Data Structures

- **FunctionRegistryTrie**: Optimized trie for function qualified name lookups with prefix/suffix search capabilities
- **AST Cache**: Stores parsed Tree-sitter ASTs for reuse during call processing phase
- **Simple Name Lookup**: Maps simple function names to qualified names for resolution

## Enhanced Knowledge Engine

The system includes an intelligent knowledge extraction and retrieval layer built on top of the AST parsing infrastructure:

### Knowledge Engine Components

- **DependencyGraphBuilder** (`knowledge_engine/dependency_graph.py`): Constructs function call dependency graphs and performs topological sorting for dependency-aware processing

- **TopologicalSummaryGenerator** (`knowledge_engine/topological_summary.py`): Generates intelligent function summaries using LLM in dependency order, ensuring all called functions are summarized before their callers

- **DualEngineIndexer** (`knowledge_engine/dual_indexer.py`): Builds hybrid search indices combining vector embeddings (Weaviate) and keyword search (BM25) for comprehensive retrieval

- **RRFRetriever** (`knowledge_engine/rrf_retriever.py`): Implements Reciprocal Rank Fusion for combining semantic and keyword search results with intelligent query analysis

- **EnhancedGraphUpdater** (`enhanced_graph_updater.py`): Orchestrates the complete pipeline from AST parsing through knowledge extraction to search index construction

### Processing Pipeline

The enhanced system follows a 7-phase pipeline:

1. **Structure Identification**: Original AST parsing (packages, folders)
2. **File Processing**: Original AST parsing (functions, classes, caching)  
3. **Call Analysis**: Original AST parsing (function calls, relationships)
4. **Dependency Graph**: Build function call dependency graph with cycle detection
5. **Summary Generation**: LLM-powered function summarization in topological order
6. **Index Construction**: Build vector and BM25 search indices
7. **Retriever Initialization**: Setup hybrid search with RRF fusion

### Search Capabilities

- **Semantic Search**: Vector similarity using OpenAI embeddings
- **Keyword Search**: BM25-based exact term matching
- **Hybrid Fusion**: RRF combination with query-adaptive weighting
- **Similarity Search**: Find functions similar to a reference function
- **Complexity Filtering**: Search by function complexity levels
- **Context-Aware Results**: Results include summaries, purposes, and dependencies

## Common Development Tasks

### Enhanced System Usage

#### 🔥 本地版本 (推荐 - 无需 API key)

```bash
# 安装本地版本依赖
pip install -r requirements-local.txt

# 启动本地服务
docker-compose -f docker-compose-local.yml up -d

# 安装和配置 Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull gpt-oss-20b  # 或其他本地模型

# 运行完整本地流水线
python local_enhanced_main.py --repo-path /path/to/your/codebase

# 交互式本地搜索
python local_enhanced_main.py --interactive

# 本地搜索命令
python query_interface.py --search "用户认证相关函数"
python query_interface.py --similar "myproject.auth.login"
```

#### OpenAI 版本

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"

# Start required services (Memgraph + Weaviate)
docker run -p 7687:7687 -p 7444:7444 -p 3000:3000 memgraph/memgraph-platform
docker run -p 8080:8080 semitechnologies/weaviate:latest

# Run complete enhanced pipeline (AST + Knowledge Engine)
python enhanced_main.py --repo-path /path/to/your/codebase

# Interactive search interface
python query_interface.py --interactive
```

#### 原始版本 (仅 AST 解析)

```bash
# Run AST parsing only (original functionality)
python main.py
```

### System Requirements

#### 本地版本 (推荐)
- **Python 3.12+** with packages (see `requirements-local.txt`)
- **Memgraph Database**: Graph database for AST relationships (port 7687)
- **Weaviate Database**: Vector database for semantic search (port 8080)
- **Ollama**: 本地 LLM 服务 (port 11434)
  - 安装: https://ollama.ai/
  - 模型: `ollama pull llama3.1:8b` (或gemma2:9b)
- **BGE-M3**: 自动下载的中文优化嵌入模型

#### OpenAI 版本
- **Python 3.12+** with packages (see `requirements.txt`)
- **Memgraph Database**: Graph database for AST relationships (port 7687)
- **Weaviate Database**: Vector database for semantic search (port 8080)
- **OpenAI API Key**: For LLM summaries and embeddings (set `OPENAI_API_KEY`)

### Entry Points

**Enhanced Versions:**
- `local_enhanced_main.py`: 🔥 **本地版本** - 使用 Langchain + BGE-M3 + Ollama，无需 API key
- `enhanced_main.py`: OpenAI 版本 - 使用 OpenAI API

**Original Version:**
- `main.py`: 原始 AST 解析功能

**Interactive Interface:**
- `query_interface.py`: 交互式搜索界面（支持两个版本）

### Configuration

Configuration is handled through `code_parser/config.py` using Pydantic settings and environment variables:

- **AST Parsing**: Memgraph connection settings (localhost:7687), ignore patterns
- **Knowledge Engine**: OpenAI API settings, Weaviate connection (localhost:8080)
- **Models**: OpenAI GPT-3.5/4 for summaries, text-embedding-3-small for vectors
- **Search**: RRF fusion parameters, batch processing limits

Environment variables:
```bash
OPENAI_API_KEY=your-key-here        # Required for knowledge engine features
MEMGRAPH_HOST=localhost             # Memgraph host (default: localhost) 
WEAVIATE_URL=http://localhost:8080  # Weaviate URL (default: localhost:8080)
```

### Data Storage and Outputs

The system generates several artifacts:

- **Memgraph Database**: Function definitions, call relationships, project structure
- **Weaviate Database**: Vector embeddings for semantic search
- **JSON Exports**: Function summaries, dependency graphs, search indices
- **Knowledge Base**: Exported to `knowledge_base_export/` directory

Access Memgraph Lab at http://localhost:3000 to visualize the generated knowledge graph.

### Language Support

The system uses Tree-sitter parsers located in `code_parser/grammars/` and supports multiple programming languages through `language_config.py`. Parser loading is handled by `parser_loader.py`.

## File Structure Patterns

```
├── code_parser/              # Original AST parsing infrastructure
│   ├── parsers/             # Core parsing logic with modular processors  
│   ├── services/            # Database and external service integrations
│   └── grammars/            # Tree-sitter grammar repositories
├── knowledge_engine/        # Enhanced knowledge extraction and search
│   ├── dependency_graph.py  # Function dependency analysis
│   ├── topological_summary.py # OpenAI LLM summary generation
│   ├── dual_indexer.py      # OpenAI embedding + hybrid search
│   ├── local_models.py      # 🔥 Langchain + local model integration
│   ├── local_topological_summary.py  # 🔥 Ollama LLM summary generation  
│   ├── local_dual_indexer.py # 🔥 BGE-M3 embedding + hybrid search
│   └── rrf_retriever.py     # Search and result fusion
├── enhanced_graph_updater.py # OpenAI version pipeline orchestrator
├── local_enhanced_graph_updater.py # 🔥 Local version pipeline orchestrator
├── query_interface.py       # Interactive search interface (supports both)
├── enhanced_main.py         # OpenAI version entry point
├── local_enhanced_main.py   # 🔥 Local version entry point (no API key)
├── main.py                  # Original AST parsing entry point
├── requirements.txt         # OpenAI version dependencies
├── requirements-local.txt   # 🔥 Local version dependencies
├── docker-compose.yml       # OpenAI version services
├── docker-compose-local.yml # 🔥 Local version services
└── README-LOCAL.md          # 🔥 Local version detailed guide
```

## Development Notes

### AST Parsing Layer (Original)
- Three-pass processing: structure → definitions → calls 
- AST caching prevents redundant parsing during call analysis
- Batched database writes improve ingestion performance
- Function registry uses trie data structure for efficient qualified name lookups
- Ignore patterns exclude common directories (node_modules, .git, __pycache__, etc.)

### Knowledge Engine Layer (Enhanced)  
- **Dependency-Aware Processing**: Functions summarized in topological order ensuring dependencies are processed first
- **Circular Dependency Handling**: Strong Connected Components (SCCs) detected and processed as units
- **Batch Processing**: LLM calls batched with token estimation to optimize API usage and costs
- **Hybrid Search**: Combines vector similarity (semantic) with BM25 (keyword) using RRF fusion
- **Query Analysis**: Automatic query type detection adjusts search strategy weights
- **Error Recovery**: Fallback mechanisms handle LLM failures without stopping pipeline
- **Version Compatibility**: Supports both new LangChain 0.3+ (LCEL) and legacy versions automatically

### Performance Considerations
- **Token Management**: Batch size dynamically adjusted based on token counts
- **Rate Limiting**: Built-in delays between LLM API calls
- **Caching Strategy**: AST cache reused across processing phases
- **Memory Management**: Large codebases processed in dependency-ordered batches
- **Index Optimization**: Both vector and keyword indices optimized for retrieval speed

### Architecture Principles
- **Separation of Concerns**: AST parsing and knowledge extraction as separate layers
- **Dependency Injection**: ProcessorFactory provides clean component dependencies  
- **Error Boundaries**: Each processing phase isolated with fallback strategies
- **Extensibility**: Modular design allows adding new search engines or LLM providers

## Testing and Quality

MVP implementation focuses on core functionality. Recommended additions:
- Unit tests for each knowledge engine component
- Integration tests for end-to-end pipeline
- Search relevance evaluation metrics
- Performance benchmarking for large codebases

## 库用法校验规则（API 使用前必须执行）
- 在使用任何第三方库/框架前，先执行“版本与文档确认”：
  1) 确认目标库的**主版本号**；若用户未指定，则选用**最新稳定版本**并在输出中标记版本号。
  2) 通过 WebSearch/WebFetch 查找该版本的**官方文档/迁移指南/发布日志**，识别是否有弃用/破坏式变更。
  3) 仅采用**官方推荐**的写法；若我最初的方案与推荐不一致，必须改为推荐写法并解释原因。