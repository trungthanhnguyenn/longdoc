"""
Qdrant client wrapper with production-ready features.

This module provides a robust Qdrant client with connection pooling,
retry logic, and error handling.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from qdrant_client import QdrantClient as QdrantNativeClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
# from qdrant_client.qdrant_fastembed import FastembedSparseVectorTextModels  # Not available in current version

from src.config.config import QdrantConfig

logger = logging.getLogger(__name__)


@dataclass
class VectorParams:
    """Parameters for vector configuration."""
    size: int
    distance: qdrant_models.Distance = qdrant_models.Distance.COSINE
    hnsw_config: Optional[qdrant_models.HnswConfigDiff] = None
    quantization_config: Optional[qdrant_models.QuantizationConfig] = None
    on_disk: Optional[bool] = None


@dataclass
class SearchParams:
    """Parameters for vector search."""
    vector: List[float]
    limit: int = 10
    score_threshold: Optional[float] = None
    filter: Optional[qdrant_models.Filter] = None
    with_payload: bool = True
    with_vectors: bool = False
    consistency: Optional[qdrant_models.ReadConsistency] = None


class QdrantClient:
    """
    Production-ready Qdrant client with retry logic and error handling.
    
    This class wraps the native Qdrant client with additional features for
    production environments including connection pooling, retries, and
    comprehensive error handling.
    """
    
    def __init__(self, config: QdrantConfig):
        """
        Initialize Qdrant client.
        
        Args:
            config: Qdrant configuration
        """
        self.config = config
        self.config.validate()
        
        # Initialize native client
        self._client = QdrantNativeClient(
            url=config.url,
            api_key=config.api_key,
            timeout=config.timeout,
            grpc_port=config.grpc_port,
            https=config.https,
            prefix=config.prefix
        )
        
        logger.info(f"Qdrant client initialized for {config.url}")
    
    def _retry_operation(self, operation_func, max_retries: int = 3, backoff_factor: float = 1.0):
        """
        Execute operation with retry logic.
        
        Args:
            operation_func: Function to execute
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
            
        Returns:
            Operation result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return operation_func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
                    raise last_exception
    
    def create_collection(
        self,
        collection_name: str,
        vectors_config: Union[Dict[str, VectorParams], VectorParams],
        force_recreate: bool = False
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vectors_config: Vector configuration
            force_recreate: Whether to recreate if collection exists
            
        Returns:
            True if collection was created successfully
        """
        def _create():
            # Check if collection exists
            if self._client.collection_exists(collection_name):
                if force_recreate:
                    self._client.delete_collection(collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
                else:
                    logger.info(f"Collection already exists: {collection_name}")
                    return False
            
            # Convert vector parameters
            if isinstance(vectors_config, VectorParams):
                vectors_dict = {
                    "size": vectors_config.size,
                    "distance": vectors_config.distance
                }
                
                if vectors_config.hnsw_config is not None:
                    vectors_dict["hnsw_config"] = vectors_config.hnsw_config
                
                if vectors_config.quantization_config is not None:
                    vectors_dict["quantization_config"] = vectors_config.quantization_config
                
                if vectors_config.on_disk is not None:
                    vectors_dict["on_disk"] = vectors_config.on_disk
                
                vectors_config_dict = vectors_dict
            else:
                vectors_config_dict = {}
                for name, params in vectors_config.items():
                    vectors_config_dict[name] = {
                        "size": params.size,
                        "distance": params.distance
                    }
                    if params.hnsw_config is not None:
                        vectors_config_dict[name]["hnsw_config"] = params.hnsw_config
                    if params.quantization_config is not None:
                        vectors_config_dict[name]["quantization_config"] = params.quantization_config
                    if params.on_disk is not None:
                        vectors_config_dict[name]["on_disk"] = params.on_disk
            
            # Create collection
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config_dict
            )
            
            logger.info(f"Created collection: {collection_name}")
            return True
        
        return self._retry_operation(_create)
    
    def upsert_points(
        self,
        collection_name: str,
        points: List[qdrant_models.PointStruct],
        batch_size: int = 100
    ) -> bool:
        """
        Upsert points into collection.
        
        Args:
            collection_name: Name of the collection
            points: List of points to upsert
            batch_size: Batch size for upserting
            
        Returns:
            True if upsert was successful
        """
        def _upsert():
            # Batch processing for large datasets
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self._client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                logger.debug(f"Upserted batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
            logger.info(f"Successfully upserted {len(points)} points into {collection_name}")
            return True
        
        return self._retry_operation(_upsert)
    
    def search_points(
        self,
        collection_name: str,
        search_params: SearchParams
    ) -> List[qdrant_models.ScoredPoint]:
        """
        Search for similar points.
        
        Args:
            collection_name: Name of the collection
            search_params: Search parameters
            
        Returns:
            List of scored points
        """
        def _search():
            return self._client.search(
                collection_name=collection_name,
                query_vector=search_params.vector,
                limit=search_params.limit,
                with_payload=search_params.with_payload,
                with_vectors=search_params.with_vectors,
                score_threshold=search_params.score_threshold,
                query_filter=search_params.filter,
                consistency=search_params.consistency
            )
        
        return self._retry_operation(_search)
    
    def delete_points(
        self,
        collection_name: str,
        points_selector: Union[List[str], qdrant_models.Filter]
    ) -> bool:
        """
        Delete points from collection.
        
        Args:
            collection_name: Name of the collection
            points_selector: Points to delete (either list of IDs or filter)
            
        Returns:
            True if deletion was successful
        """
        def _delete():
            self._client.delete(
                collection_name=collection_name,
                points_selector=points_selector
            )
            logger.info(f"Deleted points from {collection_name}")
            return True
        
        return self._retry_operation(_delete)
    
    def get_collection_info(self, collection_name: str) -> qdrant_models.CollectionInfo:
        """
        Get collection information.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection information
        """
        def _get_info():
            return self._client.get_collection(collection_name)
        
        return self._retry_operation(_get_info)
    
    def count_points(self, collection_name: str, filter: Optional[qdrant_models.Filter] = None) -> int:
        """
        Count points in collection.
        
        Args:
            collection_name: Name of the collection
            filter: Optional filter to apply
            
        Returns:
            Number of points
        """
        def _count():
            return self._client.count(collection_name, filter).count
        
        return self._retry_operation(_count)
    
    def scroll_points(
        self,
        collection_name: str,
        scroll_filter: Optional[qdrant_models.Filter] = None,
        limit: int = 10,
        offset: Optional[qdrant_models.ExtendedPointId] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> tuple[List[qdrant_models.Record], Optional[qdrant_models.ExtendedPointId]]:
        """
        Scroll through points in collection.
        
        Args:
            collection_name: Name of the collection
            scroll_filter: Filter to apply
            limit: Number of points to return
            offset: Offset for pagination
            with_payload: Whether to include payload
            with_vectors: Whether to include vectors
            
        Returns:
            Tuple of (points, next_page_offset)
        """
        def _scroll():
            return self._client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
        
        return self._retry_operation(_scroll)
    
    def close(self) -> None:
        """Close the client connection."""
        if hasattr(self._client, 'close'):
            self._client.close()
        logger.info("Qdrant client closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()