"""
Topological Summary Generator

Generates intelligent summaries for functions using LLM,
processing them in dependency-aware batches to ensure context availability.
"""

import json
import time
import tiktoken
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    logger.warning("OpenAI library not installed. Please run: pip install openai")
    OpenAI = None


@dataclass
class SummaryResult:
    """Result of a function summary generation."""
    qualified_name: str
    summary: str
    purpose: str
    parameters: List[Dict[str, str]]
    returns: str
    dependencies: List[str]
    complexity: str
    success: bool
    error_message: Optional[str] = None


class TopologicalSummaryGenerator:
    """Generates function summaries in topological order with dependency context."""
    
    def __init__(
        self, 
        openai_api_key: str,
        model: str = "gpt-3.5-turbo",
        max_tokens_per_batch: int = 12000,
        max_functions_per_batch: int = 8
    ):
        if not OpenAI:
            raise ImportError("OpenAI library required. Install with: pip install openai")
            
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_functions_per_batch = max_functions_per_batch
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
        # Storage for generated summaries
        self.summaries: Dict[str, SummaryResult] = {}
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self.tokenizer.encode(text))
    
    def build_batch_prompt(
        self, 
        functions: List[Dict[str, Any]], 
        context_library: Dict[str, str]
    ) -> str:
        """
        Build a comprehensive prompt for batch processing multiple functions.
        
        Args:
            functions: List of function metadata dictionaries
            context_library: Dictionary of dependency summaries
            
        Returns:
            Formatted prompt string
        """
        
        prompt = f"""You are a code analysis expert. I will provide you with {len(functions)} functions from a codebase that need to be analyzed and summarized.

CONTEXT LIBRARY (Dependencies):
The following are summaries of functions that the target functions may depend on:

"""
        
        # Add context library
        for dep_qn, summary in context_library.items():
            prompt += f"- {dep_qn}: {summary}\n"
        
        prompt += f"""

TARGET FUNCTIONS TO ANALYZE:
Please analyze each of the following {len(functions)} functions and provide a structured summary.

"""
        
        # Add function details
        for i, func in enumerate(functions, 1):
            prompt += f"""
--- FUNCTION {i} ---
Qualified Name: {func.get('qn', 'unknown')}
File: {func.get('file_path', 'unknown')}
Lines: {func.get('start_line', '?')}-{func.get('end_line', '?')}

Source Code:
```python
{func.get('source_code', 'Source code not available')}
```

Docstring: {func.get('docstring') or 'No docstring provided'}

"""
        
        prompt += """
REQUIRED OUTPUT FORMAT:
For each function, provide a JSON object with the following structure:

{
  "qualified_name": "exact.qualified.name",
  "summary": "1-2 sentence concise description of what this function does",
  "purpose": "detailed explanation of the function's purpose and role in the codebase",
  "parameters": [
    {"name": "param_name", "type": "param_type", "description": "what this parameter does"}
  ],
  "returns": "description of return value and type",
  "dependencies": ["list", "of", "internal.function.qualified.names", "this.function.calls"],
  "complexity": "LOW|MEDIUM|HIGH - based on logic complexity and number of operations"
}

Return a JSON array containing one object for each function, in the same order they were provided.

IMPORTANT RULES:
1. Focus on WHAT the function does, not HOW it does it
2. Use the context library to understand dependencies
3. Be precise about parameter types and return values
4. Only include internal function calls in dependencies (not library calls)
5. Keep summaries concise but informative
6. If source code is unclear, indicate uncertainty

JSON Response:
"""
        
        return prompt
    
    def generate_batch_summaries(
        self, 
        functions: List[Dict[str, Any]], 
        context_library: Dict[str, str]
    ) -> List[SummaryResult]:
        """
        Generate summaries for a batch of functions using LLM.
        
        Args:
            functions: List of function metadata
            context_library: Dictionary of dependency summaries
            
        Returns:
            List of SummaryResult objects
        """
        if not functions:
            return []
        
        logger.info(f"Generating summaries for batch of {len(functions)} functions")
        
        # Build prompt
        prompt = self.build_batch_prompt(functions, context_library)
        
        # Check token count
        token_count = self.count_tokens(prompt)
        logger.info(f"Batch prompt token count: {token_count}")
        
        if token_count > self.max_tokens_per_batch:
            logger.warning(f"Prompt too long ({token_count} tokens), splitting batch")
            return self._split_and_process_batch(functions, context_library)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            logger.debug(f"LLM Response: {response_text}")
            
            # Parse JSON response
            try:
                summaries_data = json.loads(response_text)
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
            logger.error(f"Error calling OpenAI API: {e}")
            return self._create_fallback_summaries(functions, f"API error: {e}")
    
    def _split_and_process_batch(
        self, 
        functions: List[Dict[str, Any]], 
        context_library: Dict[str, str]
    ) -> List[SummaryResult]:
        """Split oversized batch and process in smaller chunks."""
        mid = len(functions) // 2
        logger.info(f"Splitting batch: {len(functions)} -> {mid} + {len(functions) - mid}")
        
        results1 = self.generate_batch_summaries(functions[:mid], context_library)
        time.sleep(1)  # Rate limiting
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
            
            fallback_summary = f"Function {name}"
            if docstring:
                # Use first sentence of docstring
                first_sentence = docstring.split('.')[0] + '.' if '.' in docstring else docstring
                fallback_summary = first_sentence.strip()
            
            result = SummaryResult(
                qualified_name=func.get('qn', 'unknown'),
                summary=fallback_summary,
                purpose=fallback_summary,
                parameters=[],
                returns="Unknown",
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
        Process multiple batches in topological order.
        
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
            
            # Rate limiting between batches
            if batch_idx < len(batches) - 1:
                logger.info("Waiting 2 seconds before next batch...")
                time.sleep(2)
        
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