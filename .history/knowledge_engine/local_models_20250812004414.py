"""
Local Models Integration with Langchain

Integrates BGE-M3 embeddings and Ollama LLM for local processing without API keys.
"""

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.messages import HumanMessage
except ImportError:
    raise

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.warning("sentence-transformers not installed. Please run: pip install sentence-transformers")
    SentenceTransformer = None


class LocalEmbeddingModel:
    """BGE-M3 local embedding model using sentence-transformers."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize BGE-M3 embedding model."""
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✓ BGE-M3 embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using encode_document for IR tasks."""
        if not self.model:
            raise RuntimeError("Model not initialized")
            
        logger.debug(f"Embedding {len(texts)} documents...")
        # Use encode_document for information retrieval tasks as recommended in v5
        try:
            embeddings = self.model.encode_document(texts, normalize_embeddings=True)
        except AttributeError:
            # Fallback to encode if encode_document is not available
            logger.debug("encode_document not available, falling back to encode")
            embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using encode_query for IR tasks."""
        if not self.model:
            raise RuntimeError("Model not initialized")
            
        # Use encode_query for information retrieval tasks as recommended in v5
        try:
            embedding = self.model.encode_query(text, normalize_embeddings=True)
            return embedding.tolist()
        except AttributeError:
            # Fallback to encode if encode_query is not available
            logger.debug("encode_query not available, falling back to encode")
            embedding = self.model.encode([text], normalize_embeddings=True)
            return embedding[0].tolist()


class LocalLLMModel:
    """Local LLM using Ollama."""
    
    def __init__(
        self, 
        model_name: str = "llama3.1:8b",  # 或者用户实际使用的模型名
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 4000
    ):
        """Initialize Ollama LLM."""
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Ollama LLM."""
        try:
            logger.info(f"Connecting to Ollama at {self.ollama_base_url}")
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.ollama_base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens
            )
            
            # Test connection
            test_msg = self.llm.invoke("Hello")
            test_text = getattr(test_msg, "content", str(test_msg))
            logger.info(f"✓ Ollama chat model ({self.model_name}) connected successfully")
            logger.debug(f"Test response: {test_text[:100]}...")
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            logger.error(f"And model is available: ollama pull {self.model_name}")
            raise
    
    def generate(self, prompt: str) -> str:
        """Generate text using the local Chat LLM."""
        if not self.llm:
            raise RuntimeError("LLM not initialized")    
        try:
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate for prompt: {e}")
                results.append(f"Generation failed: {e}")
        
        return results


class LangchainSummaryChain:
    """Langchain-based summary generation chain."""
    
    def __init__(self, llm_model: LocalLLMModel):
        """Initialize summary chain."""
        self.llm = llm_model
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup the Langchain summary chains using new LCEL syntax."""
        # Function summary prompt template
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的代码分析专家。你的任务是分析函数代码并生成结构化的JSON摘要。

你需要严格按照以下JSON格式输出，不要添加任何额外的解释、标记或文本：
{{
  "summary": "1-2句话简述函数功能",
  "purpose": "详细说明函数的目的和作用", 
  "parameters": [
    {{"name": "参数名", "type": "类型", "description": "参数作用"}}
  ],
  "returns": "返回值描述",
  "dependencies": ["依赖的内部函数列表"],
  "complexity": "LOW|MEDIUM|HIGH"
}}

分析要求：
1. 使用中文描述
2. 重点关注函数的业务逻辑，而非实现细节  
3. 只列出项目内部函数依赖，不包括库函数
4. 复杂度基于逻辑复杂性和代码行数判断"""),
            ("human", """请分析以下函数并提供结构化的JSON摘要：

函数信息：
- 名称: {function_name}
- 完整路径: {qualified_name}
- 文档字符串: {docstring}

源代码：
```python
{source_code}
```

依赖上下文：
{dependencies_context}

请直接返回JSON，不要添加任何其他内容：""")
        ])
        
        # Batch summary prompt for multiple functions
        self.batch_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的代码分析专家。你需要分析多个函数并为每个函数生成结构化的JSON摘要。

你需要严格按照以下JSON数组格式输出，不要添加任何额外的解释、标记或文本：
[
  {{
    "qualified_name": "函数完整路径",
    "summary": "1-2句话简述功能",
    "purpose": "详细目的说明",
    "parameters": [{{"name": "参数名", "type": "类型", "description": "说明"}}],
    "returns": "返回值描述", 
    "dependencies": ["内部函数依赖"],
    "complexity": "LOW|MEDIUM|HIGH"
  }}
]

分析要求：
1. 使用中文描述
2. 基于上下文库理解函数依赖关系
3. 只列出项目内部函数依赖，不包括库函数
4. 按提供的函数顺序返回结果"""),
            ("human", """请分析以下多个函数并为每个函数提供结构化的JSON摘要：

上下文库（依赖函数的摘要）：
{context_library}

待分析的函数：
{functions_data}

请直接返回JSON数组，不要添加任何其他内容：""")
        ])
        
        # Create chains using LCEL syntax with JSON parsing
        self.json_parser = JsonOutputParser()
        self.summary_chain = self.summary_prompt | self.llm.llm | self.json_parser
        self.batch_summary_chain = self.batch_summary_prompt | self.llm.llm | self.json_parser
    
    def generate_single_summary(
        self, 
        function_name: str,
        qualified_name: str,
        source_code: str,
        docstring: str = "",
        dependencies_context: str = ""
    ) -> Dict[str, Any]:
        """Generate summary for a single function."""
        try:
            result = self.summary_chain.invoke({
                "function_name": function_name,
                "qualified_name": qualified_name,
                "source_code": source_code,
                "docstring": docstring or "无文档字符串",
                "dependencies_context": dependencies_context or "无依赖"
            })
            return result
            
        except Exception as e:
            logger.error(f"Single summary generation failed: {e}")
            raise
    
    def generate_batch_summaries(
        self,
        functions_data: str,
        context_library: str
    ) -> List[Dict[str, Any]]:
        """Generate summaries for multiple functions."""
        try:
            result = self.batch_summary_chain.invoke({
                "functions_data": functions_data,
                "context_library": context_library or "暂无依赖上下文"
            })
            return result
            
        except Exception as e:
            logger.error(f"Batch summary generation failed: {e}")
            raise


def test_local_models():
    """Test local models functionality."""
    print("🧪 Testing Local Models...")
    
    # Test embedding model
    try:
        print("Testing BGE-M3 embedding...")
        embedding_model = LocalEmbeddingModel()
        
        test_texts = ["这是一个测试函数", "用于用户认证的方法"]
        embeddings = embedding_model.embed_documents(test_texts)
        print(f"✓ Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False
    
    # Test LLM model
    try:
        print("Testing Ollama LLM...")
        llm_model = LocalLLMModel()
        
        test_prompt = "请用一句话介绍Python编程语言。"
        response = llm_model.generate(test_prompt)
        print(f"✓ LLM response: {response[:100]}...")
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False
    
    # Test summary chain
    try:
        print("Testing summary chain...")
        summary_chain = LangchainSummaryChain(llm_model)
        
        test_summary = summary_chain.generate_single_summary(
            function_name="test_function",
            qualified_name="module.test_function",
            source_code="def test_function():\n    return 'hello'",
            docstring="测试函数"
        )
        print(f"✓ Summary generated: {test_summary[:100]}...")
        
    except Exception as e:
        print(f"❌ Summary chain test failed: {e}")
        return False
    
    print("🎉 All local model tests passed!")
    return True


if __name__ == "__main__":
    test_local_models()