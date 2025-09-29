"""
Agent module for document processing and analysis.

This module provides specialized agents for:
- Document Read Agent: Analyzes large chunks to create report frameworks
- Document Write Agent: Uses RAG to generate content from frameworks
"""

from .read import DocumentReadAgent
from .write import DocumentWriteAgent

__all__ = ['DocumentReadAgent', 'DocumentWriteAgent']