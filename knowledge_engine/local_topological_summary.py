"""
Local Topological Summary Generator

Uses local Ollama LLM (GPT OSS 20B) and Langchain for dependency-aware function summarization.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .local_models import LocalLLMModel, LangchainSummaryChain
from .topological_summary import SummaryResult  # Reuse the result class


class LocalTopologicalSummaryGenerator:
    """Generates function summaries using local LLM in topological order."""
    
    def __init__(
        self, 
        llm_model_name: str = "gpt-oss-20b",
        ollama_base_url: str = "http://localhost:11434",
        max_functions_per_batch: int = 5,  # Reduced for local model
        temperature: float = 0.1
    ):
        """
        Initialize local topological summary generator.
        
        Args:
            llm_model_name: Ollama model name (e.g., 'gpt-oss-20b')
            ollama_base_url: Ollama server URL
            max_functions_per_batch: Max functions per batch (smaller for local)
            temperature: LLM temperature for generation
        """
        self.llm_model_name = llm_model_name
        self.ollama_base_url = ollama_base_url
        self.max_functions_per_batch = max_functions_per_batch
        self.temperature = temperature
        
        # Initialize local LLM and summary chain
        logger.info("Initializing local LLM and summary chain...")
        self.llm_model = LocalLLMModel(
            model_name=llm_model_name,
            ollama_base_url=ollama_base_url,
            temperature=temperature,
            max_tokens=4000
        )
        
        self.summary_chain = LangchainSummaryChain(self.llm_model)
        
        # Storage for generated summaries
        self.summaries: Dict[str, SummaryResult] = {}
        
        logger.info(f"✓ Local summary generator ready with {llm_model_name}")
    
    def _format_dependencies_context(self, dependencies: Dict[str, str]) -> str:
        """Format dependencies as context string."""
        if not dependencies:
            return "无依赖函数"
        
        context_parts = []
        for dep_qn, summary in dependencies.items():
            context_parts.append(f"- {dep_qn}: {summary}")
        
        return "\n".join(context_parts)
    
    def _format_functions_for_batch(self, functions: List[Dict[str, Any]]) -> str:
        """Format multiple functions for batch processing."""
        formatted_functions = []
        
        for i, func in enumerate(functions, 1):
            func_text = f"""
函数 {i}:
- 名称: {func.get('name', '未知')}
- 完整路径: {func.get('qn', '未知')}
- 文件: {func.get('file_path', '未知')}
- 文档字符串: {func.get('docstring', '无')}

源代码:
```python
{func.get('source_code', '源代码不可用')}
```
"""
            formatted_functions.append(func_text)
        
        return "\n" + "="*50 + "\n".join(formatted_functions)
    
    def generate_batch_summaries(
        self, 
        functions: List[Dict[str, Any]], 
        context_library: Dict[str, str]
    ) -> List[SummaryResult]:
        """
        Generate summaries for a batch of functions using local LLM.
        
        Args:
            functions: List of function metadata
            context_library: Dictionary of dependency summaries
            
        Returns:
            List of SummaryResult objects
        """
        if not functions:
            return []
        
        # Limit batch size for local processing
        if len(functions) > self.max_functions_per_batch:
            logger.warning(f"Batch size {len(functions)} > {self.max_functions_per_batch}, splitting...")
            return self._split_and_process_batch(functions, context_library)
        
        logger.info(f"Generating summaries for batch of {len(functions)} functions using local LLM")
        
        try:
            # Format inputs for the local LLM
            functions_data = self._format_functions_for_batch(functions)
            dependencies_context = self._format_dependencies_context(context_library)
            
            # Generate summaries using Langchain
            response_text = self.summary_chain.generate_batch_summaries(
                functions_data=functions_data,
                context_library=dependencies_context
            )
            
            logger.debug(f"Local LLM response: {response_text[:200]}...")
            
            # Parse JSON response
            try:
                # Clean the response text to extract JSON
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                else:
                    # Fallback: try to find JSON-like content
                    json_text = response_text
                
                summaries_data = json.loads(json_text)
                
                if not isinstance(summaries_data, list):
                    raise ValueError("Response must be a JSON array")
                    
                results = []
                for i, summary_data in enumerate(summaries_data):
                    if i >= len(functions):
                        logger.warning(f"More summaries returned than functions provided")
                        break
                        
                    func = functions[i]
                    result = SummaryResult(
                        qualified_name=summary_data.get('qualified_name', func.get('qn', 'unknown')),
                        summary=summary_data.get('summary', ''),
                        purpose=summary_data.get('purpose', ''),
                        parameters=summary_data.get('parameters', []),
                        returns=summary_data.get('returns', ''),
                        dependencies=summary_data.get('dependencies', []),
                        complexity=summary_data.get('complexity', 'UNKNOWN'),
                        success=True
                    )
                    results.append(result)
                    
                logger.info(f"Successfully generated {len(results)} summaries")
                return results
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Response text: {response_text}")
                return self._create_fallback_summaries(functions, f"JSON parsing error: {e}")
                
        except Exception as e:
            logger.error(f"Error with local LLM: {e}")
            return self._create_fallback_summaries(functions, f"LLM error: {e}")
    
    def _split_and_process_batch(
        self, 
        functions: List[Dict[str, Any]], 
        context_library: Dict[str, str]
    ) -> List[SummaryResult]:
        """Split oversized batch and process in smaller chunks."""
        mid = len(functions) // 2
        logger.info(f"Splitting batch: {len(functions)} -> {mid} + {len(functions) - mid}")
        
        results1 = self.generate_batch_summaries(functions[:mid], context_library)
        time.sleep(2)  # Rate limiting for local model
        results2 = self.generate_batch_summaries(functions[mid:], context_library)
        
        return results1 + results2
    
    def _create_fallback_summaries(
        self, 
        functions: List[Dict[str, Any]], 
        error_message: str
    ) -> List[SummaryResult]:
        """Create fallback summaries when LLM call fails."""
        logger.warning(f"Creating fallback summaries due to: {error_message}")
        
        results = []
        for func in functions:
            # Extract basic info from docstring and function name
            docstring = func.get('docstring', '')
            name = func.get('name', func.get('qn', 'unknown').split('.')[-1])
            
            fallback_summary = f"函数 {name}"
            if docstring:
                # Use first sentence of docstring
                first_sentence = docstring.split('。')[0] + '。' if '。' in docstring else docstring
                if not first_sentence and '.' in docstring:
                    first_sentence = docstring.split('.')[0] + '.'
                fallback_summary = first_sentence.strip() if first_sentence else fallback_summary
            
            result = SummaryResult(
                qualified_name=func.get('qn', 'unknown'),
                summary=fallback_summary,
                purpose=fallback_summary,
                parameters=[],
                returns="未知",
                dependencies=[],
                complexity="UNKNOWN",
                success=False,
                error_message=error_message
            )
            results.append(result)
        
        return results
    
    def process_batches(
        self, 
        batches: List[List[Dict[str, Any]]],
        get_context_library_func: callable
    ) -> Dict[str, SummaryResult]:
        """
        Process multiple batches in topological order using local LLM.
        
        Args:
            batches: List of function batches from topological sorting
            get_context_library_func: Function to get context library for a batch
            
        Returns:
            Dictionary mapping qualified names to SummaryResult objects
        """
        all_summaries = {}
        
        for batch_idx, batch_functions in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} "
                       f"with {len(batch_functions)} functions")
            
            # Get context library for this batch
            context_library = get_context_library_func(
                [func.get('qn') for func in batch_functions]
            )
            
            # Update context library with previously generated summaries
            for qn, result in all_summaries.items():
                if result.success and qn not in context_library:
                    context_library[qn] = result.summary
            
            # Generate summaries for current batch
            batch_results = self.generate_batch_summaries(batch_functions, context_library)
            
            # Store results
            for result in batch_results:
                all_summaries[result.qualified_name] = result
                self.summaries[result.qualified_name] = result
            
            # Rate limiting between batches (more conservative for local)
            if batch_idx < len(batches) - 1:
                logger.info("Waiting 3 seconds before next batch...")
                time.sleep(3)
        
        logger.info(f"Completed processing {len(batches)} batches, "
                   f"generated {len(all_summaries)} summaries")
        
        # Report success rate
        successful = sum(1 for r in all_summaries.values() if r.success)
        success_rate = successful / len(all_summaries) * 100 if all_summaries else 0
        logger.info(f"Success rate: {successful}/{len(all_summaries)} ({success_rate:.1f}%)")
        
        return all_summaries
    
    def get_summary(self, qualified_name: str) -> Optional[SummaryResult]:
        """Get summary for a specific function."""
        return self.summaries.get(qualified_name)
    
    def export_summaries(self, file_path: str) -> None:
        """Export all summaries to JSON file."""
        export_data = {}
        for qn, result in self.summaries.items():
            export_data[qn] = {
                'summary': result.summary,
                'purpose': result.purpose,
                'parameters': result.parameters,
                'returns': result.returns,
                'dependencies': result.dependencies,
                'complexity': result.complexity,
                'success': result.success,
                'error_message': result.error_message
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(export_data)} summaries to {file_path}")
    
    def load_summaries(self, file_path: str) -> None:
        """Load summaries from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                export_data = json.load(f)
            
            for qn, data in export_data.items():
                result = SummaryResult(
                    qualified_name=qn,
                    summary=data.get('summary', ''),
                    purpose=data.get('purpose', ''),
                    parameters=data.get('parameters', []),
                    returns=data.get('returns', ''),
                    dependencies=data.get('dependencies', []),
                    complexity=data.get('complexity', 'UNKNOWN'),
                    success=data.get('success', False),
                    error_message=data.get('error_message')
                )
                self.summaries[qn] = result
            
            logger.info(f"Loaded {len(export_data)} summaries from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load summaries from {file_path}: {e}")
            raise