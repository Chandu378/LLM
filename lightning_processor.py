#!/usr/bin/env python3
"""
Lightning-Fast Document Processor
Back to basics: Extract → Process → Upload → Answer (FAST!)
"""

import os
import tempfile
import hashlib
import time
from typing import List, Dict, Any
import google.generativeai as genai
from config import config
import PyPDF2
import pdfplumber
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

class LightningProcessor:
    def __init__(self):
        """Initialize lightning-fast processor"""
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.embedding_model = "models/embedding-001"
        print("⚡ Lightning Processor initialized - SPEED MODE")
    
    async def process_lightning_fast(self, pdf_url: str, questions: List[str], 
                                   pinecone_service, redis_service=None, supabase_db_service=None) -> List[str]:
        """
        LIGHTNING FAST: Process PDF and answer questions in minimal time
        Strategy: Extract → Chunk → Embed → Upload → Search → Answer
        """
        start_time = time.time()
        document_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:12]
        
        print(f"⚡ LIGHTNING MODE: Processing {document_hash}")
        
        # STEP 1: Quick cache check (Redis first, then Supabase)
        cached_answers = await self._quick_cache_check(questions, document_hash, redis_service, supabase_db_service)
        if cached_answers:
            print(f"🚀 INSTANT: All answers from cache ({time.time() - start_time:.2f}s)")
            return cached_answers
        
        # STEP 2: Fast PDF download and extraction
        print("📥 Fast download...")
        pdf_content = await self._fast_download(pdf_url)
        
        print("📄 Fast extraction...")
        text_chunks = await self._fast_extract_and_chunk(pdf_content)
        
        if not text_chunks:
            return ["Document could not be processed"] * len(questions)
        
        print(f"✅ Extracted {len(text_chunks)} chunks in {time.time() - start_time:.2f}s")
        
        # STEP 3: Lightning-fast embedding and upload
        print("🚀 Lightning upload...")
        clause_ids = await self._lightning_upload(text_chunks, document_hash, pinecone_service)
        
        print(f"✅ Uploaded {len(clause_ids)} clauses in {time.time() - start_time:.2f}s")
        
        # STEP 4: Fast question answering
        print("🤖 Fast answering...")
        answers = await self._fast_answer_questions(questions, document_hash, pinecone_service)
        
        # STEP 5: Quick cache storage for future speed
        if redis_service or supabase_db_service:
            asyncio.create_task(self._quick_cache_store(questions, answers, document_hash, redis_service, supabase_db_service))
        
        total_time = time.time() - start_time
        print(f"🎉 LIGHTNING COMPLETE: {total_time:.2f}s total")
        
        return answers
    
    async def _quick_cache_check(self, questions: List[str], document_hash: str, 
                                redis_service, supabase_db_service) -> List[str]:
        """Quick cache check - Redis first, then Supabase"""
        try:
            # Check Redis first (fastest)
            if redis_service:
                cached_answers = []
                all_cached = True
                
                for question in questions:
                    cache_key = f"fast_answer:{document_hash}:{hashlib.md5(question.encode()).hexdigest()[:8]}"
                    cached = redis_service.redis_client.get(cache_key)
                    if cached:
                        cached_answers.append(cached)
                    else:
                        all_cached = False
                        break
                
                if all_cached:
                    print("⚡ Redis cache hit!")
                    return cached_answers
            
            # Check Supabase if Redis miss
            if supabase_db_service and supabase_db_service.is_available():
                cached_answers = []
                all_cached = True
                
                for question in questions:
                    cached_answer = await supabase_db_service.get_cached_answer(question, document_hash)
                    if cached_answer:
                        cached_answers.append(cached_answer["answer"])
                    else:
                        all_cached = False
                        break
                
                if all_cached:
                    print("💾 Supabase cache hit!")
                    return cached_answers
            
        except Exception as e:
            print(f"⚠️ Cache check failed: {e}")
        
        return None
    
    async def _fast_download(self, pdf_url: str) -> bytes:
        """Fast PDF download"""
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(pdf_url)
            response.raise_for_status()
            return response.content
    
    async def _fast_extract_and_chunk(self, pdf_content: bytes) -> List[Dict[str, Any]]:
        """Fast extraction and chunking in one step"""
        try:
            # Write to temp file with unique name to avoid conflicts
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', 
                                                  prefix=f'lightning_{int(time.time())}_')
            temp_file.write(pdf_content)
            temp_file.close()
            
            try:
                # Fast extraction with pdfplumber (better quality)
                chunks = []
                with pdfplumber.open(temp_file.name) as pdf:
                    text_content = ""
                    
                    # Extract all pages quickly
                    for page in pdf.pages[:50]:  # Limit to first 50 pages for speed
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n\n"
                    
                    # Fast chunking - split into meaningful pieces
                    if text_content:
                        chunks = self._smart_chunk(text_content)
                
                return chunks
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                    
        except Exception as e:
            print(f"⚠️ Fast extraction failed: {e}")
            return []
    
    def _smart_chunk(self, text: str) -> List[Dict[str, Any]]:
        """Smart chunking with Gemini API size limits (max 30KB per chunk)"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Split on natural boundaries
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 1
        MAX_CHUNK_SIZE = 25000  # 25KB limit (well under 36KB Gemini limit)
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed size limit
            potential_chunk = current_chunk + ("\n\n" + paragraph if current_chunk else paragraph)
            
            if len(potential_chunk.encode('utf-8')) > MAX_CHUNK_SIZE and current_chunk:
                # Save current chunk
                chunks.append({
                    "id": f"chunk_{chunk_id:03d}",
                    "content": current_chunk.strip(),
                    "word_count": len(current_chunk.split()),
                    "char_count": len(current_chunk),
                    "byte_size": len(current_chunk.encode('utf-8'))
                })
                current_chunk = paragraph
                chunk_id += 1
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "id": f"chunk_{chunk_id:03d}",
                "content": current_chunk.strip(),
                "word_count": len(current_chunk.split()),
                "char_count": len(current_chunk),
                "byte_size": len(current_chunk.encode('utf-8'))
            })
        
        # Additional safety check - split any chunks that are still too large
        safe_chunks = []
        for chunk in chunks:
            if chunk["byte_size"] > MAX_CHUNK_SIZE:
                # Split large chunk into smaller pieces
                content = chunk["content"]
                sentences = content.split('. ')
                
                sub_chunk = ""
                sub_id = 1
                
                for sentence in sentences:
                    potential_sub = sub_chunk + (". " + sentence if sub_chunk else sentence)
                    
                    if len(potential_sub.encode('utf-8')) > MAX_CHUNK_SIZE and sub_chunk:
                        # Save sub-chunk
                        safe_chunks.append({
                            "id": f"{chunk['id']}_sub{sub_id}",
                            "content": sub_chunk.strip(),
                            "word_count": len(sub_chunk.split()),
                            "char_count": len(sub_chunk),
                            "byte_size": len(sub_chunk.encode('utf-8'))
                        })
                        sub_chunk = sentence
                        sub_id += 1
                    else:
                        sub_chunk = potential_sub
                
                # Add final sub-chunk
                if sub_chunk:
                    safe_chunks.append({
                        "id": f"{chunk['id']}_sub{sub_id}",
                        "content": sub_chunk.strip(),
                        "word_count": len(sub_chunk.split()),
                        "char_count": len(sub_chunk),
                        "byte_size": len(sub_chunk.encode('utf-8'))
                    })
            else:
                safe_chunks.append(chunk)
        
        print(f"📊 Smart chunking: {len(safe_chunks)} chunks created")
        for i, chunk in enumerate(safe_chunks[:3]):  # Show first 3 chunks info
            print(f"   Chunk {i+1}: {chunk['byte_size']} bytes, {chunk['word_count']} words")
        
        return safe_chunks
    
    async def _lightning_upload(self, chunks: List[Dict[str, Any]], document_hash: str, 
                               pinecone_service) -> List[str]:
        """Lightning-fast parallel upload to Pinecone"""
        if not pinecone_service or not pinecone_service.index:
            return []
        
        vectors = []
        
        # Generate embeddings in parallel (max 5 concurrent for speed)
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all embedding tasks
            future_to_chunk = {
                executor.submit(self._generate_embedding, chunk["content"]): chunk 
                for chunk in chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    embedding = future.result(timeout=15)  # 15 second timeout per embedding
                    if embedding:
                        vector = {
                            "id": f"{document_hash}_{chunk['id']}",
                            "values": embedding,
                            "metadata": {
                                "content": chunk["content"],
                                "document": document_hash,
                                "word_count": chunk["word_count"],
                                "char_count": chunk["char_count"]
                            }
                        }
                        vectors.append(vector)
                except Exception as e:
                    print(f"⚠️ Embedding failed for {chunk['id']}: {e}")
        
        # Batch upload to Pinecone
        if vectors:
            try:
                pinecone_service.index.upsert(vectors=vectors)
                return [v["id"] for v in vectors]
            except Exception as e:
                print(f"❌ Pinecone upload failed: {e}")
        
        return []
    
    def _generate_embedding(self, content: str) -> List[float]:
        """Generate single embedding with size validation"""
        try:
            # Safety check for content size
            content_bytes = len(content.encode('utf-8'))
            if content_bytes > 30000:  # 30KB safety limit
                print(f"⚠️ Content too large ({content_bytes} bytes), truncating...")
                # Truncate to safe size
                content = content[:20000]  # Truncate to ~20KB
            
            result = genai.embed_content(
                model=self.embedding_model,
                content=content,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
            # If still failing, try with even smaller content
            if "payload size exceeds" in str(e) and len(content) > 10000:
                print("🔄 Retrying with smaller content...")
                try:
                    smaller_content = content[:8000]  # Much smaller
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=smaller_content,
                        task_type="retrieval_document"
                    )
                    return result['embedding']
                except Exception as retry_error:
                    print(f"❌ Retry failed: {retry_error}")
            return None
    
    async def _fast_answer_questions(self, questions: List[str], document_hash: str, 
                                   pinecone_service) -> List[str]:
        """Fast question answering using Pinecone search"""
        answers = []
        
        for question in questions:
            try:
                # Generate question embedding
                question_embedding = self._generate_embedding(question)
                if not question_embedding:
                    answers.append("Could not process question")
                    continue
                
                # Search Pinecone
                search_results = pinecone_service.index.query(
                    vector=question_embedding,
                    top_k=5,  # Get top 5 most relevant chunks
                    include_metadata=True,
                    filter={"document": document_hash}
                )
                
                if not search_results.matches:
                    answers.append("Information not found in document")
                    continue
                
                # Generate answer from top matches
                relevant_content = []
                for match in search_results.matches:
                    if match.score > 0.5:  # Lower threshold for better recall
                        relevant_content.append(match.metadata.get("content", ""))
                
                if relevant_content:
                    answer = await self._generate_fast_answer(question, relevant_content)
                    answers.append(answer)
                else:
                    answers.append("No relevant information found")
                    
            except Exception as e:
                print(f"❌ Question answering failed: {e}")
                answers.append("Error processing question")
        
        return answers
    
    async def _generate_fast_answer(self, question: str, relevant_content: List[str]) -> str:
        """Generate fast answer using Gemini"""
        try:
            # Combine relevant content
            context = "\n\n".join(relevant_content[:3])  # Use top 3 chunks
            
            prompt = f"""Based on the following content, answer the question directly and concisely.

Content:
{context}

Question: {question}

Answer (1-2 sentences maximum):"""

            # Fast generation with minimal parameters
            from services import gemini_service
            response = gemini_service.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'top_k': 1,
                    'max_output_tokens': 100
                }
            )
            
            answer = response.text.strip()
            return answer if answer else "Information not clearly specified"
            
        except Exception as e:
            print(f"❌ Answer generation failed: {e}")
            return "Could not generate answer"
    
    async def _quick_cache_store(self, questions: List[str], answers: List[str], 
                               document_hash: str, redis_service, supabase_db_service):
        """Quick cache storage for future speed"""
        try:
            # Store in Redis for immediate future requests
            if redis_service:
                for question, answer in zip(questions, answers):
                    cache_key = f"fast_answer:{document_hash}:{hashlib.md5(question.encode()).hexdigest()[:8]}"
                    redis_service.redis_client.setex(cache_key, 3600, answer)  # 1 hour TTL
            
            # Store in Supabase for long-term caching
            if supabase_db_service and supabase_db_service.is_available():
                for question, answer in zip(questions, answers):
                    await supabase_db_service.cache_answer(
                        question, document_hash, answer, [], 0.5, confidence_score=0.9
                    )
            
        except Exception as e:
            print(f"⚠️ Cache storage failed: {e}")

# Global lightning processor
lightning_processor = LightningProcessor()