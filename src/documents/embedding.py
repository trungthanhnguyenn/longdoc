from typing import List
import logging
import requests
import time

from requests.exceptions import HTTPError


from src.config.config import APIConfig, APISmartChunk

class Embedding():
    """
    Embedding Adapter
    """
    
    def __init__(self, config: APIConfig):
        """
        Initialize pythera embedding adapter
        
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

    def _process_single_chunk(self, chunk_text: str, chunk_index: int) -> List[APISmartChunk]:
        """
        Process a single chunk through the API with retry logic
        
        Args:
            chunk_text: Individual chunk text to process
            chunk_index: Index of chunk for ID generation
            
        Returns:
            List[APISmartChunk]: API chunks with embeddings
        """
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                params = {"text": chunk_text}

                headers = {
                    'Accept': 'application/json',
                }

                response = requests.post(
                    f"{self.api_url}/context",
                    params=params,
                    headers=headers,
                    timeout=60
                )
                
                self.logger.debug(f"Chunk {chunk_index} - Request URL: {response.url}")
                self.logger.debug(f"Chunk {chunk_index} - Response status: {response.status_code}")
                
                if response.status_code >= 400:
                    self.logger.error(f"Chunk {chunk_index} - Response text: {response.text}")
                
                if response.status_code == 520:
                    self.logger.warning(f"Chunk {chunk_index} - Cloudflare 520 error on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        raise HTTPError(f"Chunk {chunk_index} - Received 520 error from API server after retries")
                
                if response.status_code != 200:
                    raise HTTPError(f"Chunk {chunk_index} - API returned status {response.status_code}: {response.text}")
                
                api_chunks = response.json()
                
                if not api_chunks:
                    self.logger.warning(f"Chunk {chunk_index} - API returned empty response")
                    return []
                
                # Convert to APISmartChunk format with unique IDs
                smart_chunks = []
                for api_chunk_idx, chunk_data in enumerate(api_chunks):
                    # Ensure unique chunk ID across all processed chunks
                    unique_id = f"chunk_{chunk_index}_{api_chunk_idx}_{chunk_data.get('id', 'unknown')}"
                    
                    smart_chunk = APISmartChunk(
                        id=unique_id,
                        chunk=chunk_data["chunk"],
                        embedding=chunk_data["emb"],
                        score=0.0  # Will be set during reranking
                    )
                    smart_chunks.append(smart_chunk)
                
                self.logger.debug(f"Chunk {chunk_index} - Created {len(smart_chunks)} API chunks with embeddings")
                return smart_chunks
                
            except Exception as e:
                self.logger.error(f"Chunk {chunk_index} - Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    # Log specific error details for debugging
                    self.logger.error(f"Chunk {chunk_index} - All retry attempts failed. Final error: {e}")
                    self.logger.error(f"Chunk {chunk_index} - Chunk text length: {len(chunk_text)}")
                    self.logger.error(f"Chunk {chunk_index} - Chunk preview: {chunk_text[:200]}...")
                    raise
        
        return []
    
    def get_query_embeddings(self, query: str) -> List[float]:
        """
        Get embeddings for query
        
        Args:
            query: Query text
            
        Returns:
            List[float]: Query embedding
        """
        try:
            response = requests.post(
                f"{self.api_url}/query",
                json={"text": query, "model_name": "retrieve_query"},
                timeout=60
            )
            
            if response.status_code != 200:
                raise HTTPError(f"Query API returned status {response.status_code}")
            
            query_chunks = response.json()
            
            # Return first query embedding
            if query_chunks:
                return query_chunks[0]["emb"]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Query embedding failed: {e}")
            return []