import uuid
import os
import requests
from typing import List, Dict, Any

from src.qdrant.client import QdrantClient, SearchParams
from src.qdrant.manager import QdrantManager
from src.qdrant.manager import DocumentMetadata
from src.agent.read import DocumentReadAgent
from src.agent.write import DocumentWriteAgent
from src.documents.preprocess import load_and_split_text
from src.documents.chunking import Chunking
from src.documents.embedding import Embedding
from src.config.config import APIConfig, QdrantConfig, LLMAgentConfig
from datetime import datetime


def create_batches_from_chunks(chunks: List[str], max_batch_size: int = 5000) -> List[str]:
    """
    Create batches of chunks with total size <= max_batch_size.
    
    Args:
        chunks: List of text chunks
        max_batch_size: Maximum characters per batch
        
    Returns:
        List of batched text strings
    """
    batches = []
    current_batch = []
    current_length = 0
    
    for chunk in chunks:
        chunk_length = len(chunk)
        
        # If adding this chunk exceeds max size, finalize current batch
        if current_length + chunk_length > max_batch_size and current_batch:
            batches.append(" ".join(current_batch))
            current_batch = [chunk]
            current_length = chunk_length
        else:
            current_batch.append(chunk)
            current_length += chunk_length
    
    # Add remaining chunks
    if current_batch:
        batches.append(" ".join(current_batch))
    
    return batches


def rag_with_rerank(query: str, embedding_api: Embedding, collection_name: str, 
                   qdrant_client: QdrantClient, top_k: int = 20, rerank_top_k: int = 5) -> List[str]:
    """
    Perform RAG with reranking.
    
    Args:
        query: Query text
        embedding_api: Embedding API instance
        collection_name: Qdrant collection name
        qdrant_client: Qdrant client instance
        top_k: Number of documents to retrieve initially
        rerank_top_k: Number of documents to return after reranking
        
    Returns:
        List of top reranked context chunks
    """
    try:
        # Step 1: Get query embedding
        query_embedding = embedding_api.get_query_embeddings(query)
        if not query_embedding:
            print(f"Failed to get query embedding for: {query}")
            return []
        
        
        search_params = SearchParams(
            vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        search_results = qdrant_client.search_points(
            collection_name=collection_name,
            search_params=search_params
        )
        
        if not search_results:
            print(f"No results found for query: {query}")
            return []
        
        # Step 3: Prepare for reranking
        contexts = []
        for result in search_results:
            payload = result.payload
            if payload and 'text' in payload:
                contexts.append({
                    'id': result.id,
                    'text': payload['text'],
                    'score': result.score
                })
        
        # Step 4: Rerank using API
        if len(contexts) > rerank_top_k:
            rerank_data = {
                'query': query,
                'contexts': [ctx['text'] for ctx in contexts]
            }
            
            try:
                rerank_response = requests.post(
                    f"{embedding_api.api_url}/rerank",
                    json=rerank_data,
                    timeout=embedding_api.timeout
                )
                
                if rerank_response.status_code == 200:
                    rerank_results = rerank_response.json()
                    # Return top rerank_top_k contexts
                    return [ctx['text'] for ctx in rerank_results[:rerank_top_k]]
                else:
                    print(f"Rerank failed with status {rerank_response.status_code}")
                    # Fallback to top rerank_top_k by original score
                    return [ctx['text'] for ctx in contexts[:rerank_top_k]]
                    
            except Exception as e:
                print(f"Rerank error: {e}")
                # Fallback to top rerank_top_k by original score
                return [ctx['text'] for ctx in contexts[:rerank_top_k]]
        else:
            return [ctx['text'] for ctx in contexts]
    
    except Exception as e:
        print(f"RAG with rerank error: {e}")
        return []


def run(file_path: str):
    """Main processing pipeline."""
    try:
        print(f"🚀 Starting document processing for: {file_path}")
        
        # Initialize configurations
        api_config = APIConfig.from_env()
        qdrant_config = QdrantConfig.from_env()
        llm_config = LLMAgentConfig.from_env()
        
        # Initialize Qdrant client and manager
        qdrant_client = QdrantClient(qdrant_config)
        qdrant_manager = QdrantManager(qdrant_config)
        
        # Initialize processing modules
        chunking = Chunking(api_config)
        embedding = Embedding(api_config)
        
        # Initialize agents
        read_agent = DocumentReadAgent(llm_config, qdrant_config=qdrant_config)
        write_agent = DocumentWriteAgent(llm_config, qdrant_config=qdrant_config)
        
        print("✅ All modules initialized successfully")
        
        # Step 1: Load and preprocess document
        print("\nLoading document...")
        raw_doc = load_and_split_text(file_path)
        print(f"Document loaded ({len(raw_doc)} characters)")
        
        # Step 2: Create large chunks using semantic chunking
        print("\nCreating semantic chunks...")
        large_chunks = chunking._semantic_chunk_text(raw_doc)
        print(f"Created {len(large_chunks)} large chunks")
        
        # Step 3: Create collection name based on file name
        file_name = os.path.basename(file_path)
        file_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_name))
        collection_name = f"doc_{file_uuid.replace('-', '')[:16]}"
        
        print(f"\nCollection name: {collection_name}")
        
        # Step 4: Check if collection exists, create and populate if not
        if not qdrant_manager.is_collection_exists(collection_name):
            print(f"Creating new collection and embedding documents...")
            
            # Create collection using client directly
            from qdrant_client.http import models as qdrant_models
            from src.qdrant.client import VectorParams
            
            vector_params = VectorParams(
                size=768,
                distance=qdrant_models.Distance.COSINE,
                on_disk=True
            )
            
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_params,
                force_recreate=False
            )
            
            # Embed and upload chunks to Qdrant
            all_smart_chunks = []
            for i, chunk in enumerate(large_chunks):
                print(f"Embedding chunk {i+1}/{len(large_chunks)}...")
                smart_chunks = embedding._process_single_chunk(chunk, i)
                all_smart_chunks.extend(smart_chunks)
            
            # Prepare metadata for Qdrant
            embeddings = [chunk.embedding for chunk in all_smart_chunks]
            metadata_list = []
            chunk_texts = []
            
            for i, smart_chunk in enumerate(all_smart_chunks):
                metadata = DocumentMetadata(
                    doc_id=file_uuid,
                    title=f"Document {file_uuid}",
                    source=file_path,
                    document_type="text",
                    chunk_index=i,
                    total_chunks=len(all_smart_chunks),
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    tags=["test", "document"]
                )
                metadata_list.append(metadata)
                chunk_texts.append(smart_chunk.chunk)
            
            # Upload to Qdrant - update manager collection name and reuse
            qdrant_manager.collection_name = collection_name
            success = qdrant_manager.add_document(embeddings, metadata_list, chunk_texts)
            if success:
                print(f"Successfully uploaded {len(all_smart_chunks)} chunks to Qdrant")
            else:
                raise Exception("Failed to upload chunks to Qdrant")
                
        else:
            print(f"Collection {collection_name} already exists, skipping embedding")
        
        # Step 5: Create batches for read agent (max 5000 chars per batch)
        print(f"\nCreating batches for read agent...")
        batches = create_batches_from_chunks(large_chunks, max_batch_size=5000)
        print(f"Created {len(batches)} batches for processing")
        
        # Step 6: Process batches with read agent (sequential with skeleton updates)
        print(f"\nProcessing with Read Agent...")
        document_id = str(uuid.uuid4())
        skeleton = None
        
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} characters)...")
            
            try:
                skeleton = read_agent.analyze_document_chunk(
                    chunk_text=batch,
                    document_id=document_id,
                    chunk_index=i,
                    existing_skeleton=skeleton
                )
                
                if i == 0:
                    print(f"Initial skeleton created with {len(skeleton.main_sections)} sections")
                else:
                    print(f"Skeleton updated (version {skeleton.version})")
                    
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
                if i == 0:  # If first batch fails, we can't continue
                    raise
        
        if not skeleton:
            raise Exception("Failed to create report skeleton")
        
        print(f"Read Agent completed. Final skeleton has {len(skeleton.main_sections)} sections")
        
        # Save skeleton for debugging
        _save_skeleton_for_debug(skeleton)
        
        # Step 7: Process with write agent using RAG with reranking
        print(f"\nProcessing with Write Agent (RAG with reranking)...")
        
        # Debug: Check collection data before processing
        print(f"\nDebug: Checking collection data...")
        try:
            collection_stats = qdrant_manager.get_collection_stats()
            print(f"Collection stats: {collection_stats}")
            
            # Try to get a few sample points
            sample_points = qdrant_client.search_points(
                collection_name=collection_name,
                search_params=SearchParams(
                    vector=[0.1] * 768,  # Dummy vector
                    limit=3,
                    score_threshold=None,
                    with_payload=True,
                    with_vectors=False
                )
            )
            print(f"Sample points found: {len(sample_points)}")
            for i, point in enumerate(sample_points[:2]):
                if point.payload and point.payload.get('text'):
                    text_preview = point.payload['text'][:100] + "..." if len(point.payload['text']) > 100 else point.payload['text']
                    print(f"Sample {i+1}: {text_preview}")
        except Exception as e:
            print(f"Debug check failed: {e}")

        try:
            # Generate output filename based on input file
            base_file_name = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"report_{base_file_name}"
            
            print(f"🎯 Using collection name: {collection_name}")
            complete_report = write_agent.write_complete_report(
                skeleton=skeleton,
                collection_name=collection_name,
                context_limit=5,
                output_filename=output_filename
            )
            
            print(f"Write Agent completed successfully")
            
            # Display results
            print(f"\nFinal Report Summary:")
            print(f"   Title: {complete_report.title}")
            print(f"   Sections: {len(complete_report.main_sections)}")
            print(f"   Version: {complete_report.version}")
            print(f"   Created: {complete_report.created_at}")
            print(f"   Updated: {complete_report.updated_at}")
            
            # Display section information
            for i, section in enumerate(complete_report.main_sections, 1):
                content_length = len(section.content) if section.content else 0
                print(f"   {i}. {section.title} ({content_length} chars)")
            
            return complete_report
            
        except Exception as e:
            print(f"Error in Write Agent processing: {e}")
            raise
            
    except Exception as e:
        print(f"Processing failed: {e}")
        raise


def _save_skeleton_for_debug(skeleton):
    """Save skeleton to JSON file for debugging."""
    import json
    
    # Create debug directory
    debug_dir = "./debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Convert skeleton to dict
    skeleton_dict = {
        "document_id": skeleton.document_id,
        "title": skeleton.title,
        "version": skeleton.version,
        "created_at": skeleton.created_at,
        "updated_at": skeleton.updated_at,
        "main_sections": []
    }
    
    for section in skeleton.main_sections:
        section_dict = {
            "section_id": section.section_id,
            "title": section.title,
            "description": section.description,
            "order": section.order,
            "parent_section": section.parent_section,
            "questions": section.questions if section.questions else [],
            "content": section.content if section.content else None
        }
        skeleton_dict["main_sections"].append(section_dict)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"skeleton_debug_{timestamp}.json"
    filepath = os.path.join(debug_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(skeleton_dict, f, ensure_ascii=False, indent=2)
    
    print(f"📝 Skeleton saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python main.py <document_path>")
        sys.exit(1)
    
    document_path = sys.argv[1]
    
    if not os.path.exists(document_path):
        print(f"Error: Document file not found: {document_path}")
        sys.exit(1)
    
    try:
        result = run(document_path)
        print(f"\nDocument processing completed successfully!")
        
    except Exception as e:
        print(f"\nProcessing failed: {e}")
        sys.exit(1)
