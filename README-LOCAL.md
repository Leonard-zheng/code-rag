# Code RAG - Local Version 🚀

## 为什么选择本地版本？

你说得对！使用本地模型有很多优势：

- **🔒 隐私保护**: 代码不离开本地环境
- **💰 零成本**: 无需 OpenAI API key，完全免费
- **🌐 离线工作**: 无需网络连接
- **🎯 中文优化**: BGE-M3 对中文支持更好
- **🛠 完全控制**: 自由选择和调优模型

## 技术栈

### 🤖 AI 模型层
- **LLM**: Ollama + GPT OSS 20B (或其他本地模型)
- **Embedding**: BGE-M3 中文优化嵌入模型
- **Framework**: Langchain 统一接口

### 📊 存储层  
- **图数据库**: Memgraph (AST 关系存储)
- **向量数据库**: Weaviate (语义搜索)
- **关键词搜索**: BM25 (精确匹配)

### 🔍 检索层
- **混合搜索**: 向量 + BM25 双引擎
- **结果融合**: RRF (Reciprocal Rank Fusion)
- **智能路由**: 查询类型自适应权重

## 快速开始

### 1. 环境准备

```bash
# 安装最新版本依赖（已更新到 LangChain 0.3+）
pip install -r requirements-local.txt

# 测试兼容性（可选）
python test_new_langchain.py

# 启动数据库服务
docker-compose -f docker-compose-local.yml up -d

# 安装和配置 Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# 拉取 LLM 模型（根据你的需求选择）
ollama pull gpt-oss-20b    # 推荐：高质量 20B 模型
ollama pull llama3.2       # 备用：最新 Llama 模型  
ollama pull codellama      # 代码专用：代码生成优化
ollama pull qwen2.5        # 中文优化：阿里通义千问
```

#### 📦 主要版本更新

- **LangChain**: 0.3.27 (使用新的 LCEL 语法)
- **langchain-ollama**: 0.3.6 (专用 Ollama 集成)
- **sentence-transformers**: 5.1.0+ (BGE-M3 支持)
- **weaviate-client**: 4.16.6+ (v4 API)
- **transformers**: 4.55.0+ (最新 Hugging Face)

#### ⚠️ 重要变化

1. **Ollama 集成**：从 `langchain.llms.Ollama` 改为 `langchain_ollama.OllamaLLM`
2. **LCEL 语法**：使用 `|` 管道操作符构建链，`invoke()` 替代 `run()`
3. **Weaviate API**：使用 v4 客户端 API
4. **模块分离**：核心组件移至 `langchain_core`，社区组件移至 `langchain_community`

### 2. 运行完整流水线

```bash
# 运行完整的本地 RAG 流水线
python local_enhanced_main.py --repo-path /path/to/your/code

# 自定义模型
python local_enhanced_main.py \
  --repo-path /path/to/your/code \
  --llm-model codellama \
  --embedding-model BAAI/bge-m3
```

### 3. 交互式搜索

```bash
# 启动交互界面
python local_enhanced_main.py --interactive

# 或者直接使用查询接口
python query_interface.py --interactive
```

## 核心特性

### 🧠 依赖感知摘要生成
- 拓扑排序确保函数按依赖顺序处理
- 上下文注入提升摘要质量
- 循环依赖检测和处理

### 🔍 智能混合搜索
- **语义搜索**: "用户认证相关的函数"
- **关键词搜索**: "def login(username, password)"  
- **智能融合**: 自动调节语义/关键词权重

### 📊 处理流水线
1. **AST 解析**: Tree-sitter 多语言支持
2. **依赖分析**: NetworkX 图分析
3. **摘要生成**: Ollama LLM 批处理
4. **向量化**: BGE-M3 本地嵌入
5. **索引构建**: Weaviate + BM25 双引擎
6. **智能检索**: RRF 融合排序

## 使用示例

### 命令行搜索
```bash
# 语义搜索
python query_interface.py --search "处理用户登录的函数"

# 寻找相似函数
python query_interface.py --similar "myproject.auth.login"

# 按复杂度查找
python query_interface.py --complexity HIGH
```

### 交互模式
```
🔍 Query> 用户认证相关的函数

🔍 Found 5 results for: '用户认证相关的函数'

1. authenticate_user (MEDIUM)
📍 myproject.auth.authenticate_user
📁 /src/auth/handlers.py
🎯 验证用户凭据并返回认证状态

💡 Purpose: 接收用户名和密码，通过数据库验证用户身份...

🔗 Source: vector | Score: 0.8542
```

## 配置选项

### LLM 模型选择
```bash
# 不同规模的模型选择
python local_enhanced_main.py --llm-model llama2        # 7B 轻量级
python local_enhanced_main.py --llm-model codellama     # 7B 代码专用
python local_enhanced_main.py --llm-model gpt-oss-20b   # 20B 高质量
```

### 批处理调优
```python
# 在 local_enhanced_main.py 中调整
max_functions_per_batch = 3  # 减少批次大小，适合小内存
max_functions_per_batch = 8  # 增加批次大小，提升效率
```

## 性能优化

### 内存优化
- 小批量处理避免内存溢出
- AST 缓存复用减少重复解析
- 依赖顺序处理确保上下文可用

### 速度优化  
- 本地模型推理速度快
- 批量嵌入减少模型调用
- 向量索引支持快速检索

## 故障排除

### 常见问题

**Ollama 连接失败**
```bash
# 检查 Ollama 服务
curl http://localhost:11434/api/tags

# 重启 Ollama
ollama serve
```

**模型未找到**
```bash
# 查看已安装模型
ollama list

# 拉取模型
ollama pull gpt-oss-20b
```

**BGE-M3 下载慢**
```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
python local_enhanced_main.py
```

**Weaviate 连接失败**
```bash
# 检查服务状态
docker-compose -f docker-compose-local.yml ps

# 重启服务
docker-compose -f docker-compose-local.yml restart weaviate
```

## 项目结构

```
├── knowledge_engine/           # 本地知识引擎
│   ├── local_models.py        # Langchain + 本地模型集成
│   ├── local_dual_indexer.py  # BGE-M3 + Weaviate 索引
│   ├── local_topological_summary.py  # Ollama 摘要生成
│   └── rrf_retriever.py       # RRF 检索融合
├── local_enhanced_graph_updater.py   # 本地流水线协调器
├── local_enhanced_main.py     # 本地版本入口
├── requirements-local.txt      # 本地版本依赖
└── docker-compose-local.yml   # 本地服务编排
```

## 与 OpenAI 版本对比

| 特性 | 本地版本 | OpenAI 版本 |
|------|----------|------------|
| 成本 | 🆓 免费 | 💰 API 费用 |
| 隐私 | 🔒 完全本地 | ☁️ 云端处理 |
| 中文支持 | 🎯 BGE-M3 优化 | 🌐 通用模型 |
| 离线使用 | ✅ 支持 | ❌ 需要网络 |
| 处理速度 | ⚡ 本地推理 | 🌐 网络延迟 |
| 模型选择 | 🎛️ 完全自由 | 🔒 OpenAI 限制 |

## 贡献指南

欢迎贡献！特别是：
- 支持更多本地 LLM 模型
- 优化中文处理能力  
- 添加更多嵌入模型选择
- 性能优化和内存管理

## 许可证

MIT License - 开源免费使用