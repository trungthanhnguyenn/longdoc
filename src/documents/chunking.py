import logging
import requests
import re
import time

from requests.exceptions import HTTPError
from typing import List

from src.config.config import APIConfig
from src.documents.preprocess import load_and_split_text

class Chunking():
    """
    API Chunking Adapter
    """
    
    def __init__(self, config: APIConfig):
        """
        Initialize Pythera API chunking adapter
        
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
        
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns"""
        # Vietnamese and English sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỬỮỰÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ])'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str, chunk_size: int) -> List[str]:
        """Split very long sentences by word boundaries"""
        words = sentence.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            potential_chunk = current_chunk + (" " if current_chunk else "") + word
            
            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
                
                # If single word is still too long, truncate it
                if len(current_chunk) > chunk_size:
                    chunks.append(current_chunk[:chunk_size])
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """Apply overlap between consecutive chunks"""
        if len(chunks) <= 1 or overlap <= 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # First chunk unchanged
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Get overlap from end of previous chunk
            overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
            
            # Find word boundary for cleaner overlap
            if overlap_text and not overlap_text.startswith(' '):
                space_idx = overlap_text.find(' ')
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx:]
            
            # Combine overlap with current chunk
            overlapped_chunk = overlap_text.strip() + " " + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks

    def _semantic_chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Perform semantic chunking on input text before API processing.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size for each chunk (characters)
            overlap: Overlap between consecutive chunks (characters)
        
        Returns:
            List[str]: List of semantically meaningful text chunks
        
        Raises:
            ValueError: If text is empty or parameters are invalid
            Exception: For other processing errors with detailed logging
        """
        try:
            # Input validation
            if not text or not text.strip():
                self.logger.error("Input text is empty or whitespace-only")
                raise ValueError("Input text cannot be empty or whitespace-only")
            
            if chunk_size <= 0:
                self.logger.error(f"Invalid chunk_size: {chunk_size}")
                raise ValueError("chunk_size must be positive")
                
            if overlap < 0 or overlap >= chunk_size:
                self.logger.error(f"Invalid overlap: {overlap} (must be 0 <= overlap < chunk_size)")
                raise ValueError("overlap must be non-negative and less than chunk_size")
            
            text = text.strip()
            
            # If text is smaller than chunk_size, return as single chunk
            if len(text) <= chunk_size:
                self.logger.info(f"Text length ({len(text)}) <= chunk_size ({chunk_size}), returning single chunk")
                return [text]
            
            # Semantic chunking implementation
            chunks = []
            
            # Step 1: Split by paragraphs first (double newlines)
            paragraphs = re.split(r'\n\s*\n', text)
            
            current_chunk = ""
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # If paragraph alone exceeds chunk_size, need to split sentences
                if len(paragraph) > chunk_size:
                    # Save current chunk if exists
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    
                    # Split long paragraph by sentences
                    sentences = self._split_sentences(paragraph)
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        # If single sentence is too long, split by character with word boundaries
                        if len(sentence) > chunk_size:
                            # Save current chunk if exists
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""
                            
                            # Split long sentence by words, respecting chunk_size
                            word_chunks = self._split_long_sentence(sentence, chunk_size)
                            chunks.extend(word_chunks)
                        else:
                            # Check if adding sentence would exceed chunk_size
                            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
                            
                            if len(potential_chunk) <= chunk_size:
                                current_chunk = potential_chunk
                            else:
                                # Save current chunk and start new one
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = sentence
                else:
                    # Check if adding paragraph would exceed chunk_size
                    potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
                    
                    if len(potential_chunk) <= chunk_size:
                        current_chunk = potential_chunk
                    else:
                        # Save current chunk and start new one
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = paragraph
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Apply overlap between chunks
            if overlap > 0 and len(chunks) > 1:
                chunks = self._apply_overlap(chunks, overlap)
            
            # Filter out empty chunks
            chunks = [chunk for chunk in chunks if chunk.strip()]
            
            if not chunks:
                self.logger.error("No valid chunks created from input text")
                raise Exception("Failed to create any valid chunks from input text")
            
            self.logger.info(f"Successfully created {len(chunks)} semantic chunks from text of length {len(text)}")
            return chunks
            
        except ValueError as e:
            self.logger.error(f"Validation error in semantic chunking: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in semantic chunking: {e}")
            raise Exception(f"Semantic chunking failed: {e}")
