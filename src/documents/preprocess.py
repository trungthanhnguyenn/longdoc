import os
from docx import Document

def load_and_split_text(file_path: str) -> str:
        """Load text file and split into chunks, with smart chunking option"""
        # Read file (supports both txt and docx)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        elif file_extension == '.docx':
            if Document is None:
                raise ImportError("python-docx library is required to read .docx files. Install with: pip install python-docx")
            doc = Document(file_path)
            raw_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Only .txt and .docx files are supported.")
        
        return raw_text