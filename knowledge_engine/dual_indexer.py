"""
Dual Engine Indexer

Manages both vector (Weaviate) and keyword (BM25) indices for hybrid retrieval.
Provides unified interface for indexing function summaries and metadata.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from loguru import logger

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
except ImportError:
    logger.warning("Weaviate client not installed. Please run: pip install weaviate-client")
    weaviate = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logger.warning("rank_bm25 not installed. Please run: pip install rank-bm25")
    BM25Okapi = None

try:
    from openai import OpenAI
except ImportError:
    logger.warning("OpenAI library not installed. Please run: pip install openai")
    OpenAI = None

from .topological_summary import SummaryResult


class DualEngineIndexer:
    """Manages vector and BM25 indices for hybrid search."""
    
    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        openai_api_key: str = None,
        collection_name: str = "CodeFunctions",
        embedding_model: str = "text-embedding-3-small"
    ):
        if not all([weaviate, BM25Okapi, OpenAI]):
            raise ImportError("Missing required libraries. Install with: "
                            "pip install weaviate-client rank-bm25 openai")
        
        self.weaviate_url = weaviate_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize OpenAI client for embeddings
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            logger.warning("No OpenAI API key provided, embeddings will not work")
            self.openai_client = None
        
        # Initialize Weaviate client
        self.weaviate_client = None
        self._connect_weaviate()
        
        # BM25 index data
        self.bm25_documents: List[Dict[str, Any]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        
        # Document storage for retrieval
        self.document_store: Dict[str, Dict[str, Any]] = {}
    
    def _connect_weaviate(self):
        """Connect to Weaviate and ensure collection exists."""
        try:
            self.weaviate_client = weaviate.connect_to_local(
                host=self.weaviate_url.replace("http://", "").replace("https://", "")
            )
            logger.info(f"Connected to Weaviate at {self.weaviate_url}")
            
            # Create collection if it doesn't exist
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            logger.error("Make sure Weaviate is running. Try: docker run -p 8080:8080 "
                        "semitechnologies/weaviate:latest")
            self.weaviate_client = None
    
    def _ensure_collection(self):
        """Ensure the collection exists with proper schema."""
        if not self.weaviate_client:
            return
            
        try:
            # Check if collection exists
            if self.weaviate_client.collections.exists(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            
            # Create collection with schema
            collection = self.weaviate_client.collections.create(
                name=self.collection_name,
                description="Code function summaries and metadata",
                properties=[
                    Property(name="qualified_name", data_type=DataType.TEXT),
                    Property(name="function_name", data_type=DataType.TEXT),
                    Property(name="summary", data_type=DataType.TEXT),
                    Property(name="purpose", data_type=DataType.TEXT),
                    Property(name="file_path", data_type=DataType.TEXT),
                    Property(name="start_line", data_type=DataType.INT),
                    Property(name="end_line", data_type=DataType.INT),
                    Property(name="complexity", data_type=DataType.TEXT),
                    Property(name="parameters", data_type=DataType.TEXT),  # JSON string
                    Property(name="returns", data_type=DataType.TEXT),
                    Property(name="dependencies", data_type=DataType.TEXT),  # JSON string
                    Property(name="source_code", data_type=DataType.TEXT),
                    Property(name="docstring", data_type=DataType.TEXT),
                    Property(name="labels", data_type=DataType.TEXT),  # JSON string
                ],
                vectorizer_config=Configure.Vectorizer.none()  # We'll provide our own vectors
            )
            
            logger.info(f"Created collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using OpenAI API."""
        if not self.openai_client:
            return None
            
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def _prepare_document_for_indexing(
        self, 
        summary_result: SummaryResult, 
        function_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare a document for indexing with both vector and BM25 indices.
        
        Args:
            summary_result: Generated summary data
            function_metadata: Original function metadata from AST parsing
            
        Returns:
            Prepared document dictionary
        """
        # Combine text fields for embedding and BM25
        searchable_text = f"""
        {summary_result.summary} 
        {summary_result.purpose} 
        {function_metadata.get('name', '')} 
        {function_metadata.get('docstring', '')}
        {' '.join([p.get('name', '') for p in summary_result.parameters])}
        """.strip()
        
        # Prepare document
        doc = {
            # Core identifiers
            'qualified_name': summary_result.qualified_name,
            'function_name': function_metadata.get('name', '').split('.')[-1],
            
            # Summary data
            'summary': summary_result.summary,
            'purpose': summary_result.purpose,
            'complexity': summary_result.complexity,
            'returns': summary_result.returns,
            
            # Metadata
            'file_path': function_metadata.get('file_path', ''),
            'start_line': function_metadata.get('start_line', 0),
            'end_line': function_metadata.get('end_line', 0),
            'source_code': function_metadata.get('source_code', ''),
            'docstring': function_metadata.get('docstring', ''),
            
            # JSON serialized fields
            'parameters': json.dumps(summary_result.parameters),
            'dependencies': json.dumps(summary_result.dependencies),
            'labels': json.dumps(function_metadata.get('labels', [])),
            
            # For search
            'searchable_text': searchable_text,
            'bm25_tokens': searchable_text.lower().split()
        }
        
        return doc
    
    def index_summaries(
        self, 
        summaries: Dict[str, SummaryResult],
        function_metadata: Dict[str, Dict[str, Any]]
    ) -> bool:
        """
        Index all summaries into both vector and BM25 indices.
        
        Args:
            summaries: Dictionary of qualified name -> SummaryResult
            function_metadata: Dictionary of qualified name -> function metadata
            
        Returns:
            True if indexing was successful
        """
        logger.info(f"Indexing {len(summaries)} function summaries...")
        
        documents = []
        successful_embeddings = 0
        
        for qn, summary_result in summaries.items():
            if not summary_result.success:
                logger.warning(f"Skipping failed summary for {qn}")
                continue
                
            func_metadata = function_metadata.get(qn, {})
            doc = self._prepare_document_for_indexing(summary_result, func_metadata)
            
            # Get embedding for vector index
            embedding = self._get_embedding(doc['searchable_text'])
            if embedding:
                doc['vector'] = embedding
                successful_embeddings += 1
            else:
                logger.warning(f"Failed to get embedding for {qn}")
                doc['vector'] = None
            
            documents.append(doc)
            self.document_store[qn] = doc
        
        logger.info(f"Generated embeddings for {successful_embeddings}/{len(documents)} documents")
        
        # Index to Weaviate
        weaviate_success = self._index_to_weaviate(documents)
        
        # Build BM25 index
        bm25_success = self._build_bm25_index(documents)
        
        success = weaviate_success and bm25_success
        
        if success:
            logger.info("Successfully indexed all summaries to both engines")
        else:
            logger.error("Failed to index to one or both engines")
            
        return success
    
    def _index_to_weaviate(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents to Weaviate vector database."""
        if not self.weaviate_client:
            logger.error("Weaviate client not available")
            return False
            
        try:
            collection = self.weaviate_client.collections.get(self.collection_name)
            
            # Clear existing data
            logger.info("Clearing existing Weaviate data...")
            collection.data.delete_many(where=weaviate.classes.query.Filter.by_property("qualified_name").exists())
            
            # Batch insert
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                batch_objects = []
                for doc in batch:
                    if doc.get('vector'):  # Only add documents with embeddings
                        # Prepare properties (exclude vector and search-specific fields)
                        properties = {k: v for k, v in doc.items() 
                                    if k not in ['vector', 'searchable_text', 'bm25_tokens']}
                        
                        batch_objects.append(
                            weaviate.classes.data.DataObject(
                                properties=properties,
                                vector=doc['vector']
                            )
                        )
                
                if batch_objects:
                    collection.data.insert_many(batch_objects)
                    logger.info(f"Indexed batch {i//batch_size + 1} ({len(batch_objects)} docs) to Weaviate")
                
                time.sleep(0.1)  # Rate limiting
            
            logger.info(f"Successfully indexed {len(documents)} documents to Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index to Weaviate: {e}")
            return False
    
    def _build_bm25_index(self, documents: List[Dict[str, Any]]) -> bool:
        """Build BM25 index for keyword search."""
        try:
            # Prepare corpus for BM25
            corpus = [doc['bm25_tokens'] for doc in documents]
            
            if not corpus:
                logger.error("No documents available for BM25 indexing")
                return False
                
            # Build BM25 index
            self.bm25_index = BM25Okapi(corpus)
            self.bm25_documents = documents
            
            logger.info(f"Built BM25 index with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            return False
    
    def vector_search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of (qualified_name, score) tuples
        """
        if not self.weaviate_client:
            logger.error("Weaviate client not available")
            return []
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                logger.error("Failed to get query embedding")
                return []
            
            collection = self.weaviate_client.collections.get(self.collection_name)
            
            # Perform vector search
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
            )
            
            results = []
            for obj in response.objects:
                qualified_name = obj.properties.get('qualified_name', 'unknown')
                # Convert distance to similarity score (lower distance = higher similarity)
                score = 1.0 / (1.0 + obj.metadata.distance) if obj.metadata.distance else 1.0
                results.append((qualified_name, score))
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def bm25_search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of (qualified_name, score) tuples
        """
        if not self.bm25_index:
            logger.error("BM25 index not available")
            return []
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Sort by score and get top results
            scored_docs = [(i, score) for i, score in enumerate(scores)]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for doc_idx, score in scored_docs[:limit]:
                if doc_idx < len(self.bm25_documents):
                    qualified_name = self.bm25_documents[doc_idx]['qualified_name']
                    results.append((qualified_name, float(score)))
            
            logger.debug(f"BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def get_document(self, qualified_name: str) -> Optional[Dict[str, Any]]:
        """Get document by qualified name."""
        return self.document_store.get(qualified_name)
    
    def export_index(self, file_path: str) -> None:
        """Export document store to JSON file."""
        try:
            export_data = {
                'documents': self.document_store,
                'metadata': {
                    'total_documents': len(self.document_store),
                    'collection_name': self.collection_name,
                    'embedding_model': self.embedding_model
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Exported index with {len(self.document_store)} documents to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export index: {e}")
    
    def close(self):
        """Close connections and cleanup."""
        if self.weaviate_client:
            try:
                self.weaviate_client.close()
                logger.info("Closed Weaviate connection")
            except Exception as e:
                logger.error(f"Error closing Weaviate connection: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()