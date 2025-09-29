"""
Long Document Processing System - Main Package.

This package provides a comprehensive system for processing and analyzing
long documents with vector database integration.
"""

from .base import BaseModelConfig, BaseModelManager, DatabaseManager, DocumentManager
from .qdrant import QdrantClient, QdrantConfig, QdrantManager, DocumentMetadata

__all__ = [
    'BaseModelConfig', 
    'BaseModelManager', 
    'DatabaseManager', 
    'DocumentManager',
    'QdrantClient',
    'QdrantConfig', 
    'QdrantManager',
    'DocumentMetadata'
]