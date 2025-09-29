"""
Qdrant manager for high-level operations.

This module provides high-level abstractions for common Qdrant operations
in a production environment.
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

from qdrant_client.http import models as qdrant_models

from src.base.manager import DatabaseManager
from .client import QdrantClient, VectorParams, SearchParams
from src.config.config import QdrantConfig

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for document storage."""
    doc_id: str
    title: str
    source: str
    document_type: str
    chunk_index: int
    total_chunks: int
    created_at: str
    updated_at: str
    tags: List[str]
    file_size: Optional[int] = None
    file_hash: Optional[str] = None


class QdrantManager(DatabaseManager):
    """
    High-level Qdrant manager for document processing.
    
    This class provides convenient methods for managing documents and
    embeddings in Qdrant, with proper error handling and logging.
    """
    
    def _get_default_config(self) -> QdrantConfig:
        """
        Get default Qdrant configuration.
        
        Returns:
            QdrantConfig instance
        """
        return QdrantConfig.from_env()
    
    def _initialize(self, **kwargs) -> None:
        """
        Initialize Qdrant manager components.
        
        Args:
            **kwargs: Additional initialization parameters
        """
        self.client = QdrantClient(self.config)
        
        # Load vector configuration from environment
        self.vector_size = int(os.getenv('VECTOR_SIZE', 1536))
        self.distance_metric = os.getenv('DISTANCE_METRIC', 'Cosine').upper()
        self.collection_name = os.getenv('COLLECTION_NAME', 'longdoc_collection')
        self.batch_size = int(os.getenv('BATCH_SIZE', 100))
        
        # Initialize collection
        self._initialize_collection()
        
        self.logger.info(f"QdrantManager initialized for collection: {self.collection_name}")
    
    def _initialize_collection(self) -> None:
        """Initialize collection with proper configuration."""
        try:
            # Configure HNSW parameters for production
            hnsw_config = qdrant_models.HnswConfigDiff(
                m=16,  # Number of connections per node
                ef_construct=100,  # Build quality
                full_scan_threshold=10000  # Threshold for full scan
            )
            
            # Configure quantization for memory efficiency
            quantization_config = None
            
            vector_params = VectorParams(
                size=self.vector_size,
                distance=getattr(qdrant_models.Distance, self.distance_metric, qdrant_models.Distance.COSINE),
                hnsw_config=hnsw_config,
                quantization_config=quantization_config,
                on_disk=True  # Store vectors on disk for large collections
            )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params,
                force_recreate=False
            )
            
            logger.info(f"Collection '{self.collection_name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def add_document(
        self,
        embeddings: List[List[float]],
        metadata: List[DocumentMetadata],
        chunk_texts: Optional[List[str]] = None
    ) -> bool:
        """
        Add document embeddings to Qdrant.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata for each chunk
            chunk_texts: Optional list of chunk text content
            
        Returns:
            True if document was added successfully
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        if chunk_texts and len(chunk_texts) != len(embeddings):
            raise ValueError("Number of chunk texts must match number of embeddings")
        
        try:
            # Create points for Qdrant
            points = []
            for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
                payload = asdict(meta)
                
                if chunk_texts:
                    payload['text'] = chunk_texts[i]
                
                point = qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upsert points in batches
            success = self.client.upsert_points(
                collection_name=self.collection_name,
                points=points,
                batch_size=self.batch_size
            )
            
            logger.info(f"Added {len(points)} chunks for document {metadata[0].doc_id}")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")
            return False

    def is_collection_exists(self, collection_name: str) -> bool:
        try:
            # Check if collection exists
            collection_info = self.client.get_collection_info(collection_name)
            return collection_info.status == qdrant_models.CollectionStatus.GREEN
        except Exception as e:
            self.logger.error(f"Failed to check collection existence: {e}")
            return False

    def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            limit: Number of results to return
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter conditions
            
        Returns:
            List of search results with metadata
        """
        try:
            # Build filter if provided
            qdrant_filter = None
            if filter_conditions:
                filter_conditions_list = []
                for key, value in filter_conditions.items():
                    if isinstance(value, (list, tuple)):
                        filter_conditions_list.append(
                            qdrant_models.FieldCondition(
                                key=key,
                                match=qdrant_models.MatchAny(any=value)
                            )
                        )
                    else:
                        filter_conditions_list.append(
                            qdrant_models.FieldCondition(
                                key=key,
                                match=qdrant_models.MatchValue(value=value)
                            )
                        )
                
                qdrant_filter = qdrant_models.Filter(must=filter_conditions_list)
            
            # Create search parameters
            search_params = SearchParams(
                vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filter=qdrant_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Perform search
            results = self.client.search_points(
                collection_name=self.collection_name,
                search_params=search_params
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    'id': str(result.id),
                    'score': result.score,
                    'payload': result.payload
                }
                formatted_results.append(formatted_result)
            
            self.logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of document chunks
        """
        try:
            # Create filter for document ID
            qdrant_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="doc_id",
                        match=qdrant_models.MatchValue(value=doc_id)
                    )
                ]
            )
            
            # Get all chunks for the document
            points, _ = self.client.scroll_points(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=1000,  # Large limit to get all chunks
                with_payload=True,
                with_vectors=False
            )
            
            # Sort by chunk_index
            points.sort(key=lambda x: x.payload.get('chunk_index', 0))
            
            # Format results
            chunks = []
            for point in points:
                chunk = {
                    'id': str(point.id),
                    'chunk_index': point.payload.get('chunk_index', 0),
                    'text': point.payload.get('text', ''),
                    'metadata': {k: v for k, v in point.payload.items() if k != 'text'}
                }
                chunks.append(chunk)
            
            self.logger.info(f"Retrieved {len(chunks)} chunks for document {doc_id}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to get document chunks: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            # Create filter for document ID
            qdrant_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="doc_id",
                        match=qdrant_models.MatchValue(value=doc_id)
                    )
                ]
            )
            
            # Delete points
            success = self.client.delete_points(
                collection_name=self.collection_name,
                points_selector=qdrant_filter
            )
            
            self.logger.info(f"Deleted document {doc_id}")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete document: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Collection statistics
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection_info(self.collection_name)
            
            # Get total point count
            total_points = self.client.count_points(self.collection_name)
            
            # Get unique document count
            doc_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="doc_id",
                        match=qdrant_models.MatchAny(any=[])
                    )
                ]
            )
            
            stats = {
                'collection_name': self.collection_name,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance,
                'total_points': total_points,
                'status': collection_info.status,
                'optimizer_status': collection_info.optimizer_status,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all documents in the collection.
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of document metadata
        """
        try:
            # Get all points with doc_id field
            points, _ = self.client.scroll_points(
                collection_name=self.collection_name,
                limit=limit * 10,  # Get more points to account for chunks
                with_payload=True,
                with_vectors=False
            )
            
            # Group by document ID and get unique documents
            documents = {}
            for point in points:
                doc_id = point.payload.get('doc_id')
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        'doc_id': doc_id,
                        'title': point.payload.get('title', ''),
                        'source': point.payload.get('source', ''),
                        'document_type': point.payload.get('document_type', ''),
                        'created_at': point.payload.get('created_at', ''),
                        'updated_at': point.payload.get('updated_at', ''),
                        'tags': point.payload.get('tags', []),
                        'chunk_count': 1
                    }
                elif doc_id:
                    documents[doc_id]['chunk_count'] += 1
            
            # Convert to list and limit results
            doc_list = list(documents.values())
            return doc_list[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to list documents: {e}")
            return []
    
    def update_document_metadata(self, doc_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """
        Update metadata for a document.
        
        Args:
            doc_id: Document ID
            metadata_updates: Metadata fields to update
            
        Returns:
            True if update was successful
        """
        try:
            # Create filter for document ID
            qdrant_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="doc_id",
                        match=qdrant_models.MatchValue(value=doc_id)
                    )
                ]
            )
            
            # Get all points for the document
            points, _ = self.client.scroll_points(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            # Update payload for each point
            for point in points:
                updated_payload = point.payload.copy()
                updated_payload.update(metadata_updates)
                updated_payload['updated_at'] = self._get_current_timestamp()
                
                self.client._client.set_payload(
                    collection_name=self.collection_name,
                    points=[point.id],
                    payload=updated_payload
                )
            
            self.logger.info(f"Updated metadata for document {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update document metadata: {e}")
            return False
    
    def connect(self) -> bool:
        """
        Connect to Qdrant.
        
        Returns:
            True if connection was successful
        """
        try:
            # Test connection by getting collection info
            self.client.get_collection_info(self.collection_name)
            self.logger.info("Successfully connected to Qdrant")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Qdrant.
        
        Returns:
            True if disconnection was successful
        """
        try:
            self.client.close()
            self.logger.info("Successfully disconnected from Qdrant")
            return True
        except Exception as e:
            self.logger.error(f"Failed to disconnect from Qdrant: {e}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if Qdrant connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            self.client.get_collection_info(self.collection_name)
            return True
        except Exception:
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of Qdrant manager.
        
        Returns:
            Health check status dictionary
        """
        try:
            # Check connection
            connection_status = self.is_connected()
            
            # Get collection stats
            collection_stats = self.get_collection_stats()
            
            return {
                'status': 'healthy' if connection_status else 'unhealthy',
                'connected': connection_status,
                'collection_name': self.collection_name,
                'stats': collection_stats,
                'timestamp': self._get_current_timestamp()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': self._get_current_timestamp()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about Qdrant manager operations.
        
        Returns:
            Statistics dictionary
        """
        try:
            collection_stats = self.get_collection_stats()
            
            return {
                'manager_type': 'QdrantManager',
                'collection_name': self.collection_name,
                'vector_size': self.vector_size,
                'distance_metric': self.distance_metric,
                'batch_size': self.batch_size,
                'config': self.config.get_safe_config(),
                'collection_stats': collection_stats,
                'timestamp': self._get_current_timestamp()
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}
    
    def close(self) -> None:
        """Close the manager and underlying client."""
        self.disconnect()