"""
Document Read Agent for analyzing large document chunks and creating report skeletons.

This agent processes large document chunks (~5k characters) to create structured
report frameworks, identify sections, and generate questions for RAG system.
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

from src.base.manager import BaseModelManager
from src.config.config import LLMAgentConfig, ReportSkeleton, DocumentSection
from src.qdrant import QdrantManager


class DocumentReadAgent(BaseModelManager):
    """
    Agent for reading large document chunks and creating report frameworks.
    
    Processes large text chunks to identify document structure, create report
    skeletons, and generate targeted questions for the RAG system.
    """
    
    def _get_default_config(self):
        """Get default configuration for the agent."""
        return LLMAgentConfig.from_env()
    
    def _initialize(self, **kwargs) -> None:
        """Initialize agent components."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            openai_api_base=os.getenv('OPENAI_BASE_URL'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Initialize Qdrant manager for context
        self.qdrant_manager = QdrantManager(kwargs.get('qdrant_config'))
        
        # Statistics
        self.processed_chunks = 0
        self.created_skeletons = 0
        self.generated_questions = 0
        
        self.logger.info(f"DocumentReadAgent initialized with {self.config.model_name}")
    
    def analyze_document_chunk(
        self, 
        chunk_text: str, 
        document_id: str,
        chunk_index: int,
        existing_skeleton: Optional[ReportSkeleton] = None,
        **kwargs
    ) -> ReportSkeleton:
        """
        Analyze a large document chunk and create/update report skeleton.
        
        Args:
            chunk_text: Large text chunk (~5k characters) to analyze
            document_id: Unique document identifier
            chunk_index: Index of this chunk in the document
            existing_skeleton: Existing skeleton to update (for subsequent chunks)
            **kwargs: Additional analysis options
            
        Returns:
            Updated or created ReportSkeleton
        """
        try:
            self.logger.info(f"Analyzing chunk {chunk_index} for document {document_id}")
            
            # Prepare analysis prompt
            if existing_skeleton:
                prompt = self._create_update_prompt(chunk_text, existing_skeleton)
                operation_type = "update"
            else:
                prompt = self._create_initial_prompt(chunk_text)
                operation_type = "create"
            
            # Call LLM for analysis
            with get_openai_callback() as cb:
                system_prompt = getattr(self.config, 'system_prompt_template', 
                    "Bạn là một trợ lý AI chuyên phân tích tài liệu dài. Nhiệm vụ của bạn là phân tích và tạo báo cáo có cấu trúc.")
                
                response = self.llm.generate([[SystemMessage(content=system_prompt), HumanMessage(content=prompt)]])
                self.logger.debug(f"LLM call completed: {cb.total_tokens} tokens, ${cb.total_cost:.4f}")
            
            # Parse LLM response
            try:
                response_text = response.generations[0][0].text
            except AttributeError:
                # Handle different response structure
                if hasattr(response, 'generations') and response.generations:
                    generation = response.generations[0][0]
                    if hasattr(generation, 'text'):
                        response_text = generation.text
                    else:
                        response_text = str(generation)
                else:
                    response_text = str(response)
            
            self.logger.debug(f"LLM response type: {type(response)}")
            self.logger.debug(f"LLM response text: {response_text[:200]}...")
            
            analysis_result = self._parse_llm_response(response_text)
            
            # Create or update skeleton
            if operation_type == "create":
                skeleton = self._create_skeleton_from_analysis(
                    analysis_result, document_id, chunk_text
                )
                self.created_skeletons += 1
            else:
                skeleton = self._update_skeleton_from_analysis(
                    existing_skeleton, analysis_result, chunk_text
                )
            
            # Update statistics
            self.processed_chunks += 1
            self.generated_questions += sum(
                len(section.questions) for section in skeleton.main_sections
            )
            
            self.logger.info(f"Successfully analyzed chunk {chunk_index}: "
                           f"{len(skeleton.main_sections)} sections, "
                           f"{sum(len(s.questions) for s in skeleton.main_sections)} questions")
            
            return skeleton
            
        except Exception as e:
            self.logger.error(f"Error analyzing chunk {chunk_index}: {e}")
            raise
    
    def process_document_in_chunks(
        self,
        large_chunks: List[str],
        document_id: str,
        document_title: Optional[str] = None,
        **kwargs
    ) -> ReportSkeleton:
        """
        Process entire document by analyzing large chunks sequentially.
        
        Args:
            large_chunks: List of large text chunks (~5k chars each)
            document_id: Unique document identifier
            document_title: Optional document title
            **kwargs: Additional processing options
            
        Returns:
            Complete ReportSkeleton with all sections and questions
        """
        current_skeleton = None
        
        for i, chunk in enumerate(large_chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(large_chunks)}")
            
            try:
                current_skeleton = self.analyze_document_chunk(
                    chunk_text=chunk,
                    document_id=document_id,
                    chunk_index=i,
                    existing_skeleton=current_skeleton,
                    **kwargs
                )
                
                # Set document title if provided and this is the first chunk
                if document_title and i == 0 and current_skeleton:
                    current_skeleton.title = document_title
                
            except Exception as e:
                self.logger.error(f"Failed to process chunk {i}: {e}")
                # Continue with next chunk or re-raise based on configuration
                if kwargs.get("fail_fast", False):
                    raise
        
        return current_skeleton
    
    def _create_initial_prompt(self, chunk_text: str) -> str:
        """Create prompt for initial document analysis."""
        return f"""
        Phân tích đoạn văn bản sau và tạo khung xương báo cáo. Tập trung vào nội dung thực tế có trong văn bản.

        VĂN BẢN:
        {chunk_text}

        Hãy thực hiện các nhiệm vụ sau:
        1. Xác định loại tài liệu và đề xuất tiêu đề chính dựa trên nội dung
        2. Xác định CÁC SECTION CHÍNH có thực sự được đề cập trong văn bản
        3. Với mỗi section, tạo 2-3 câu hỏi TRỰC TIẾP dựa trên thông tin có sẵn trong văn bản

        QUAN TRỌNG:
        - Chỉ tạo section nếu có thực sự đề cập trong văn bản
        - Câu hỏi phải dựa trên thông tin cụ thể, không tự suy diễn
        - Giữ câu hỏi ngắn gọn, tập trung vào nội dung chính

        Trả lời theo định dạng JSON:
        {{
            "document_type": "loại tài liệu",
            "suggested_title": "tiêu đề đề xuất", 
            "main_sections": [
                {{
                    "title": "tên section",
                    "description": "mô tả ngắn",
                    "order": 1,
                    "questions": [
                        "câu hỏi trực tiếp từ văn bản 1",
                        "câu hỏi trực tiếp từ văn bản 2"
                    ]
                }}
            ]
        }}
        """
    
    def _create_update_prompt(self, chunk_text: str, existing_skeleton: ReportSkeleton) -> str:
        """Create prompt for updating existing skeleton."""
        skeleton_summary = self._summarize_skeleton(existing_skeleton)
        
        return f"""
        CẬP NHẬT khung xương báo cáo dựa trên nội dung mới. Tập trung vào thông tin thực tế.

        KHUNG XƯƠNG HIỆN TẠI:
        {skeleton_summary}

        VĂN BẢN MỚI:
        {chunk_text}

        Nhiệm vụ của bạn:
        1. Chỉ thêm section mới nếu thực sự có nội dung mới trong văn bản
        2. Với section mới, tạo 2-3 câu hỏi trực tiếp từ nội dung
        3. Cập nhật mô tả section hiện có nếu có thông tin mới
        4. Không thêm câu hỏi quá nhiều

        Trả lời theo định dạng JSON:
        {{
            "should_update_structure": true,
            "new_sections": [
                {{
                    "title": "tên section mới",
                    "description": "mô tả từ văn bản",
                    "order": 1,
                    "questions": [
                        "câu hỏi trực tiếp từ văn bản 1",
                        "câu hỏi trực tiếp từ văn bản 2"
                    ]
                }}
            ],
            "updated_sections": [
                {{
                    "title": "tên section cần cập nhật",
                    "updated_description": "mô tả cập nhật từ văn bản",
                    "additional_questions": [
                        "câu hỏi bổ sung từ văn bản"
                    ]
                }}
            ]
        }}
        """
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate LLM JSON response."""
        try:
            # Handle None response
            if not response_text:
                self.logger.error("LLM response is None or empty")
                raise ValueError("LLM response is None or empty")
            
            # Extract JSON from response (handle markdown code blocks)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Handle case where response_text is still empty after stripping
            if not response_text.strip():
                self.logger.error("LLM response is empty after processing")
                raise ValueError("LLM response is empty after processing")
            
            result = json.loads(response_text.strip())
            
            # Basic validation
            if not isinstance(result, dict):
                raise ValueError("LLM response is not a valid JSON object")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            self.logger.error(f"Raw response: {response_text}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            self.logger.error(f"Raw response: {response_text}")
            raise ValueError(f"Error parsing LLM response: {e}")
    
    def _create_skeleton_from_analysis(self, analysis: Dict[str, Any], document_id: str, chunk_text: str) -> ReportSkeleton:
        """Create new skeleton from LLM analysis."""
        main_sections = []
        
        for section_data in analysis.get("main_sections", []):
            # Create main section with questions directly
            main_section = DocumentSection(
                section_id=str(uuid.uuid4()),
                title=section_data.get("title", "Untitled Section"),
                description=section_data.get("description", ""),
                order=section_data.get("order", 0),
                questions=section_data.get("questions", [])
            )
            
            main_sections.append(main_section)
        
        return ReportSkeleton(
            document_id=document_id,
            title=analysis.get("suggested_title", "Untitled Document"),
            main_sections=main_sections,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    
    def _update_skeleton_from_analysis(self, skeleton: ReportSkeleton, analysis: Dict[str, Any], chunk_text: str) -> ReportSkeleton:
        """Update existing skeleton with new analysis."""
        # Update timestamp
        skeleton.updated_at = datetime.now().isoformat()
        skeleton.version += 1
        
        # Add new sections if needed
        for new_section_data in analysis.get("new_sections", []):
            new_section = DocumentSection(
                section_id=str(uuid.uuid4()),
                title=new_section_data.get("title", "New Section"),
                description=new_section_data.get("description", ""),
                order=len(skeleton.main_sections) + 1,
                questions=new_section_data.get("questions", [])
            )
            skeleton.main_sections.append(new_section)
        
        # Update existing sections
        for update_data in analysis.get("updated_sections", []):
            # Find and update matching section
            for section in skeleton.main_sections:
                if update_data.get("title") in section.title:
                    section.description = update_data.get("updated_description", section.description)
                    section.questions.extend(update_data.get("additional_questions", []))
                    break
        
        return skeleton
    
    def _summarize_skeleton(self, skeleton: ReportSkeleton) -> str:
        """Create a summary of existing skeleton for LLM context."""
        summary = f"Document: {skeleton.title}\n"
        summary += f"Sections: {len(skeleton.main_sections)}\n\n"
        
        for section in skeleton.main_sections:
            summary += f"- {section.title}: {section.description}\n"
            summary += f"  Questions: {len(section.questions)}\n"
        
        return summary
    
    def get_section_questions(self, skeleton: ReportSkeleton, section_id: str) -> List[str]:
        """Get all questions for a specific section and its sub-sections."""
        questions = []
        
        for section in skeleton.main_sections:
            if section.section_id == section_id:
                questions.extend(section.questions)
                
                # Find sub-sections
                for sub_section in skeleton.main_sections:
                    if sub_section.parent_section == section_id:
                        questions.extend(sub_section.questions)
                break
        
        return questions
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the agent."""
        try:
            # Test LLM connection with simple prompt
            test_response = self.llm.generate([[SystemMessage(content="You are a helpful assistant."), HumanMessage(content="Respond with 'OK' if you can read this.")]])
            
            # Extract health check response with error handling
            try:
                response_text = test_response.generations[0][0].text
            except AttributeError:
                # Handle different response structure
                if hasattr(test_response, 'generations') and test_response.generations:
                    generation = test_response.generations[0][0]
                    if hasattr(generation, 'text'):
                        response_text = generation.text
                    else:
                        response_text = str(generation)
                else:
                    response_text = str(test_response)
            
            llm_status = "healthy" if "OK" in response_text else "degraded"
            
        except Exception as e:
            llm_status = f"error: {e}"
        
        return {
            "agent": "healthy" if llm_status == "healthy" else "degraded",
            "components": {
                "llm": llm_status,
                "qdrant": "connected" if hasattr(self, 'qdrant_manager') else "not_initialized"
            },
            "statistics": {
                "processed_chunks": self.processed_chunks,
                "created_skeletons": self.created_skeletons,
                "generated_questions": self.generated_questions
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "processed_chunks": self.processed_chunks,
            "created_skeletons": self.created_skeletons,
            "generated_questions": self.generated_questions,
            "config": {
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "large_chunk_size": self.config.large_chunk_size
            }
        }