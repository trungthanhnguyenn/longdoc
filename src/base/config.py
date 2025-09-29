"""
Base configuration models for production-ready application modules.

This module provides abstract base classes for configuration components
across different modules (Qdrant, Documents, Agents, etc.).
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Type, TypeVar

import dotenv

# Type variable for generic typing
TConfig = TypeVar('TConfig', bound='BaseModelConfig')

logger = logging.getLogger(__name__)


@dataclass
class BaseModelConfig(ABC):
    """
    Abstract base class for configuration models.
    
    Provides common configuration loading, validation, and utility methods
    that can be inherited by specific module configurations.
    """
    
    @classmethod
    @abstractmethod
    def from_env(cls: Type[TConfig], env_file: Optional[str] = None) -> TConfig:
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in current directory.
            
        Returns:
            Configuration instance
            
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        pass
    
    @classmethod
    def _load_env_variables(cls, env_file: Optional[str] = None) -> None:
        """
        Load environment variables from file.
        
        Args:
            env_file: Path to .env file
        """
        if env_file:
            dotenv.load_dotenv(env_file)
        else:
            dotenv.load_dotenv()
    
    @classmethod
    def _get_env_var(
        cls, 
        key: str, 
        default: Any = None, 
        required: bool = False,
        var_type: Optional[Type] = None
    ) -> Any:
        """
        Get environment variable with type conversion and validation.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            required: Whether this variable is required
            var_type: Expected type for the variable
            
        Returns:
            Environment variable value
            
        Raises:
            ValueError: If required variable is missing or type conversion fails
        """
        value = os.getenv(key, default)
        
        if required and value is None:
            raise ValueError(f"Required environment variable {key} is missing")
        
        if value is not None and var_type is not None:
            try:
                if var_type == bool:
                    value = str(value).lower() in ('true', '1', 'yes', 'on')
                else:
                    value = var_type(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid type for {key}: expected {var_type.__name__}, got {value}") from e
        
        return value
    
    @abstractmethod
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        # Check if instance is a dataclass
        if hasattr(self, '__dataclass_fields__'):
            return {f.name: getattr(self, f.name) for f in fields(self)}
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_safe_config(self) -> Dict[str, Any]:
        """
        Get configuration safe for logging (hides sensitive values).
        
        Returns:
            Configuration dictionary with sensitive values masked
        """
        config_dict = self.to_dict()
        
        # Common sensitive keys to mask
        sensitive_keys = [
            'password', 'key', 'secret', 'token', 'api_key', 
            'private_key', 'access_key', 'secret_key'
        ]
        
        safe_config = {}
        for key, value in config_dict.items():
            is_sensitive = any(sensitive in key.lower() for sensitive in sensitive_keys)
            safe_config[key] = '***MASKED***' if is_sensitive and value else value
        
        return safe_config