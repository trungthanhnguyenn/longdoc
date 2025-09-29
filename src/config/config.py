from dataclasses import dataclass
from typing import Optional, List

from src.base.config import BaseModelConfig


@dataclass
class QdrantConfig(BaseModelConfig):
    """Configuration class for Qdrant connection settings."""
    
    host: str
    port: int
    grpc_port: int
    api_key: Optional[str] = None
    timeout: int = 30
    https: bool = False
    prefix: Optional[str] = None
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'QdrantConfig':
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in current directory.
            
        Returns:
            QdrantConfig instance
        """
        # Load environment variables
        cls._load_env_variables(env_file)
        
        return cls(
            host=cls._get_env_var('QDRANT_HOST', 'localhost', required=True),
            port=cls._get_env_var('QDRANT_PORT', 6333, required=True, var_type=int),
            grpc_port=cls._get_env_var('QDRANT_GRPC_PORT', 6334, required=True, var_type=int),
            api_key=cls._get_env_var('QDRANT_API_KEY'),
            timeout=cls._get_env_var('TIMEOUT', 30, var_type=int),
            https=cls._get_env_var('QDRANT_HTTPS', False, var_type=bool),
            prefix=cls._get_env_var('QDRANT_PREFIX')
        )
    
    @property
    def url(self) -> str:
        """Get the HTTP URL for Qdrant."""
        scheme = 'https' if self.https else 'http'
        url = f"{scheme}://{self.host}:{self.port}"
        if self.prefix:
            url = f"{url}/{self.prefix}"
        return url
    
    @property
    def grpc_url(self) -> str:
        """Get the gRPC URL for Qdrant."""
        return f"{self.host}:{self.grpc_port}"
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.host:
            raise ValueError("QDRANT_HOST is required")
        
        if not (1 <= self.port <= 65535):
            raise ValueError("QDRANT_PORT must be between 1 and 65535")
        
        if not (1 <= self.grpc_port <= 65535):
            raise ValueError("QDRANT_GRPC_PORT must be between 1 and 65535")
        
        if self.timeout <= 0:
            raise ValueError("TIMEOUT must be positive")
        
@dataclass
class APIConfig(BaseModelConfig):
    """Configuration class for API connection settings."""
    
    api_url: str
    timeout: int = 30
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'APIConfig':
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in current directory.
            
        Returns:
            APIConfig instance
        """
        # Load environment variables
        cls._load_env_variables(env_file)
        
        return cls(
            api_url=cls._get_env_var('API_URL', 'http://localhost:8000', required=True),
            timeout=cls._get_env_var('API_TIMEOUT', 30, var_type=int)
        )
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.api_url:
            raise ValueError("API_URL is required")
        
        if not self.api_url.startswith(('http://', 'https://')):
            raise ValueError("API_URL must start with http:// or https://")
        
        if self.timeout <= 0:
            raise ValueError("API_TIMEOUT must be positive")

@dataclass
class APISmartChunk:
    """Data structure cho API smart chunk result"""
    id: str
    chunk: str
    embedding: List[float]
    score: float = 0.0

@dataclass
class AgentConfig(BaseModelConfig):
    """Configuration class for Agent settings."""
    
    api_config: APIConfig
    qdrant_config: QdrantConfig
    max_retries: int = 3
    batch_size: int = 10
    enable_smart_chunking: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'AgentConfig':
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in current directory.
            
        Returns:
            AgentConfig instance
        """
        # Load environment variables
        cls._load_env_variables(env_file)
        
        api_config = APIConfig.from_env(env_file)
        qdrant_config = QdrantConfig.from_env(env_file)
        
        return cls(
            api_config=api_config,
            qdrant_config=qdrant_config,
            max_retries=cls._get_env_var('MAX_RETRIES', 3, var_type=int),
            batch_size=cls._get_env_var('BATCH_SIZE', 10, var_type=int),
            enable_smart_chunking=cls._get_env_var('ENABLE_SMART_CHUNKING', True, var_type=bool),
            chunk_size=cls._get_env_var('CHUNK_SIZE', 1000, var_type=int),
            chunk_overlap=cls._get_env_var('CHUNK_OVERLAP', 200, var_type=int)
        )
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.max_retries <= 0:
            raise ValueError("MAX_RETRIES must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("BATCH_SIZE must be positive")
        
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be non-negative and less than CHUNK_SIZE")

@dataclass
class LLMAgentConfig(BaseModelConfig):
    """Configuration class for LLM Agent settings."""
    
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.3
    max_tokens: int = 1500
    top_p: float = 0.9
    large_chunk_size: int = 5000
    system_prompt_template: str = """
    Bạn là một trợ lý AI chuyên phân tích tài liệu dài. Nhiệm vụ của bạn là:
    1. Đọc và hiểu nội dung các phần tài liệu được cung cấp
    2. Tạo khung xương báo cáo có cấu trúc
    3. Xác định các section và sub-section quan trọng
    4. Đề xuất câu hỏi để hệ thống RAG có thể truy vấn thông tin chi tiết
    
    Hãy phân tích kỹ lưỡng và tạo output theo định dạng JSON rõ ràng.
    """
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'LLMAgentConfig':
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in current directory.
            
        Returns:
            LLMAgentConfig instance
        """
        # Load environment variables
        cls._load_env_variables(env_file)
        
        return cls(
            model_name=cls._get_env_var('OPENAI_MODEL', 'deepseek/deepseek-r1-0528-qwen3-8b:free'),
            temperature=cls._get_env_var('LLM_TEMPERATURE', 0.3, var_type=float),
            max_tokens=cls._get_env_var('LLM_MAX_TOKENS', 4096, var_type=int),
            top_p=cls._get_env_var('LLM_TOP_P', 0.9, var_type=float),
            large_chunk_size=cls._get_env_var('LARGE_CHUNK_SIZE', 5000, var_type=int)
        )
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.model_name:
            raise ValueError("LLM_MODEL_NAME is required")
        
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("LLM_TEMPERATURE must be between 0.0 and 2.0")
        
        if self.max_tokens <= 0:
            raise ValueError("LLM_MAX_TOKENS must be positive")
        
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("LLM_TOP_P must be between 0.0 and 1.0")
        
        if self.large_chunk_size <= 0:
            raise ValueError("LARGE_CHUNK_SIZE must be positive")

@dataclass 
class DocumentSection:
    """Data structure for document section"""
    section_id: str
    title: str
    description: str
    parent_section: Optional[str] = None
    order: int = 0
    content: Optional[str] = None
    questions: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.questions is None:
            self.questions = []

@dataclass
class ReportSkeleton:
    """Data structure for report skeleton"""
    document_id: str
    title: str
    main_sections: List[DocumentSection]
    created_at: str
    updated_at: str
    version: int = 1
    
    def __post_init__(self):
        if self.main_sections is None:
            self.main_sections = []