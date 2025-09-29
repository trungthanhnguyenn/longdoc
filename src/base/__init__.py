"""
Base classes and utilities for the Long Document Processing System.

This package provides abstract base classes for configuration and management
components across different modules (Qdrant, Documents, Agents, etc.).
"""

from .config import BaseModelConfig
from .manager import BaseModelManager, DatabaseManager, DocumentManager

__all__ = [
    'BaseModelConfig', 
    'BaseModelManager', 
    'DatabaseManager', 
    'DocumentManager'
]