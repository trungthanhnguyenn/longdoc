import logging
import requests
from typing import List

from src.config.config import APIConfig, APISmartChunk

class Reranking():
    """
    Reranking Adapter
    """
    
    def __init__(self, config: APIConfig):
        """
        Initialize pythera reranking adapter
        
        Args:
            api_url: URL of API server
            timeout: Timeout for request
        """
        self.api_url = config.api_url
        self.timeout = config.timeout
        self.logger = logging.getLogger(__name__)
        
        # Validate API URL
        self._validate_api_connection()

    def _validate_api_connection(self):
        """Validate connection to API server"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                self.logger.info(f"Connected to API server at {self.api_url}")
            else:
                self.logger.warning(f"API server returned status {response.status_code}")
        except Exception as e:
            self.logger.error(f"Failed to connect to API server: {e}")
            raise ConnectionError(f"Cannot connect to API server at {self.api_url}: {e}")
        
    def _rerank_chunks(self, query: str, chunks: List[str]) -> List[str]:
        """
        Rerank chunks based on relevance to the query
        
        Args:
            query: User query string
            chunks: List of text chunks
            
        Returns:
            List of reranked chunks
        """
        try:
            payload = {
                "query": query,
                "chunks": chunks
            }
            response = requests.post(f"{self.api_url}/rerank", json=payload, timeout=self.timeout)
            response.raise_for_status()
            ranked_chunks = response.json().get("ranked_chunks", [])
            return ranked_chunks
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Reranking request failed: {e}")
            return chunks  # Return original order on failure
        
    def query_relevant_chunks(self, query: str, chunks: List[APISmartChunk], 
                            threshold: float = 0.1, max_results: int = 20) -> List[tuple]:
        """
        Query relevant chunks from existing chunks
        
        Args:
            query: Query text
            chunks: List chunks available
            threshold: Relevance threshold
            max_results: Maximum results to return
            
        Returns:
            List[tuple]: (chunk, relevance_score) pairs
        """
        try:
            query_items = [{"id": "query_0", "text": query}]
            context_items = []
            
            for chunk in chunks:
                context_items.append({
                    "id": chunk.id,
                    "text": chunk.chunk
                })
            
            rerank_request = {
                "query": query_items,
                "context": context_items,
                "thresh": [threshold, 1.0],
                "limit": max_results
            }
            
            response = requests.post(
                f"{self.api_url}/rerank",
                json=rerank_request,
                timeout=30
            )
            
            if response.status_code == 200:
                rerank_results = response.json()
                
                chunk_score_map = {}
                for result in rerank_results:
                    chunk_score_map[result["context_id"]] = result["score"]
                
                scored_chunks = []
                for chunk in chunks:
                    score = chunk_score_map.get(chunk.id, 0.0)
                    if score >= threshold:
                        scored_chunks.append((chunk, score))
                
                scored_chunks.sort(key=lambda x: x[1], reverse=True)
                return scored_chunks[:max_results]
            
            return []
            
        except Exception as e:
            self.logger.error(f"API query reranking failed: {e}")
            return []