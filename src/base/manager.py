"""
Base manager classes for production-ready application modules.

This module provides abstract base classes for management components
across different modules (Qdrant, Documents, Agents, etc.).
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .config import BaseModelConfig

logger = logging.getLogger(__name__)


class BaseModelManager(ABC):
    """
    Abstract base class for manager classes.
    
    Provides common manager functionality including initialization,
    logging, and utility methods that can be inherited by specific managers.
    """
    
    def __init__(self, config: Optional[BaseModelConfig] = None, **kwargs):
        """
        Initialize manager with configuration.
        
        Args:
            config: Configuration instance. If None, loads from environment.
            **kwargs: Additional configuration parameters
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # Validate configuration
        self.config.validate()
        
        # Initialize any additional components
        self._initialize(**kwargs)
        
        self.logger.info(f"{self.__class__.__name__} initialized successfully")
    
    @abstractmethod
    def _get_default_config(self) -> BaseModelConfig:
        """
        Get default configuration for the manager.
        
        Returns:
            Default configuration instance
        """
        pass
    
    @abstractmethod
    def _initialize(self, **kwargs) -> None:
        """
        Initialize manager-specific components.
        
        Args:
            **kwargs: Additional initialization parameters
        """
        pass
    
    def _setup_logger(self) -> logging.Logger:
        """
        Setup logger for the manager.
        
        Returns:
            Configured logger instance
        """
        logger_name = self.__class__.__name__.lower()
        return logging.getLogger(f"longdoc.{logger_name}")
    
    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            Current timestamp as string
        """
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    
    def _retry_operation(
        self, 
        operation_func, 
        max_retries: int = 3, 
        backoff_factor: float = 1.0,
        operation_name: str = "operation"
    ):
        """
        Execute operation with retry logic.
        
        Args:
            operation_func: Function to execute
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
            operation_name: Name of the operation for logging
            
        Returns:
            Operation result
            
        Raises:
            Exception: If all retries fail
        """
        import time
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return operation_func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    self.logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"{operation_name} failed after {max_retries + 1} attempts: {e}")
                    raise last_exception
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the manager and its components.
        
        Returns:
            Health check status dictionary
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the manager and its operations.
        
        Returns:
            Statistics dictionary
        """
        pass
    
    def close(self) -> None:
        """
        Close the manager and cleanup resources.
        
        This method should be overridden by subclasses to cleanup
        any specific resources (connections, files, etc.).
        """
        self.logger.info(f"{self.__class__.__name__} closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class DatabaseManager(BaseModelManager):
    """
    Specialized base class for database managers.
    
    Provides common database operations and connection management.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the database.
        
        Returns:
            True if connection was successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the database.
        
        Returns:
            True if disconnection was successful
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if database connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    def close(self) -> None:
        """Close database connection."""
        self.disconnect()


class DocumentManager(BaseModelManager):
    """
    Specialized base class for document managers.
    
    Provides common document processing operations.
    """
    
    @abstractmethod
    def process_document(self, document_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process a document.
        
        Args:
            document_path: Path to the document
            **kwargs: Additional processing parameters
            
        Returns:
            Processing results
        """
        pass
    
    @abstractmethod
    def list_documents(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List available documents.
        
        Args:
            **kwargs: Filtering and pagination parameters
            
        Returns:
            List of documents
        """
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str, **kwargs) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: Document identifier
            **kwargs: Additional deletion parameters
            
        Returns:
            True if deletion was successful
        """
        pass