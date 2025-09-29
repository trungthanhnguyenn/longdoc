"""
Qdrant vector database management module.

This package provides production-ready Qdrant integration for the Long Document Processing System.
"""

from .client import QdrantClient
from src.config.config import QdrantConfig
from .manager import QdrantManager, DocumentMetadata

__all__ = ['QdrantClient', 'QdrantConfig', 'QdrantManager', 'DocumentMetadata']