#!/usr/bin/env python3
"""
High-Performance Supabase Database Service
Dramatically improves performance through intelligent caching and storage
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import hashlib
import time

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️ Supabase not installed. Install with: pip install supabase")

class HighPerformanceSupabaseService:
    def __init__(self):
        """Initialize high-performance Supabase service"""
        self.client = None
        self._tables_initialized = False
        
        if not SUPABASE_AVAILABLE:
            print("⚠️ Supabase service disabled - package not available")
            return
        
        # Get Supabase credentials from environment
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            print("⚠️ Supabase credentials not found in environment variables")
            print("   Set SUPABASE_URL and SUPABASE_ANON_KEY to enable high-performance storage")
            return
        
        try:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            print("🚀 High-Performance Supabase service initialized")
            
        except Exception as e:
            self.client = None
            print(f"❌ Supabase initialization failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Supabase service is available"""
        return self.client is not None
    
    async def ensure_tables_exist(self):
        """Ensure all required tables exist with proper schema"""
        if not self.is_available() or self._tables_initialized:
            return
        
        print("📋 Supabase Tables Schema (create via Supabase Dashboard):")
        print("""
-- High-Performance Documents Table
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    url_hash TEXT UNIQUE NOT NULL,
    document_name TEXT NOT NULL,
    original_url TEXT,
    domain_type TEXT,
    complexity_level TEXT,
    total_clauses INTEGER DEFAULT 0,
    file_size INTEGER,
    processing_status TEXT DEFAULT 'completed',
    intelligence_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- High-Performance Clauses Table
CREATE TABLE IF NOT EXISTS clauses (
    id BIGSERIAL PRIMARY KEY,
    clause_id TEXT UNIQUE NOT NULL,
    document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    section INTEGER,
    section_title TEXT,
    word_count INTEGER,
    char_count INTEGER,
    document_type TEXT,
    relevance_score REAL DEFAULT 1.0,
    embedding_vector JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Query Cache Table for Instant Responses
CREATE TABLE IF NOT EXISTS query_cache (
    id BIGSERIAL PRIMARY KEY,
    query_hash TEXT UNIQUE NOT NULL,
    document_hash TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    confidence_score REAL DEFAULT 1.0,
    clause_ids JSONB,
    processing_time REAL,
    hit_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Semantic Cache for Similar Questions
CREATE TABLE IF NOT EXISTS semantic_cache (
    id BIGSERIAL PRIMARY KEY,
    question_embedding JSONB NOT NULL,
    question_text TEXT NOT NULL,
    document_hash TEXT NOT NULL,
    answer TEXT NOT NULL,
    similarity_threshold REAL DEFAULT 0.85,
    usage_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_documents_url_hash ON documents(url_hash);
CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain_type);
CREATE INDEX IF NOT EXISTS idx_documents_accessed ON documents(last_accessed);

CREATE INDEX IF NOT EXISTS idx_clauses_document ON clauses(document_id);
CREATE INDEX IF NOT EXISTS idx_clauses_hash ON clauses(content_hash);
CREATE INDEX IF NOT EXISTS idx_clauses_type ON clauses(document_type);
CREATE INDEX IF NOT EXISTS idx_clauses_score ON clauses(relevance_score);

CREATE INDEX IF NOT EXISTS idx_query_cache_hash ON query_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_query_cache_doc ON query_cache(document_hash);
CREATE INDEX IF NOT EXISTS idx_query_cache_used ON query_cache(last_used);

CREATE INDEX IF NOT EXISTS idx_semantic_doc ON semantic_cache(document_hash);
CREATE INDEX IF NOT EXISTS idx_semantic_usage ON semantic_cache(usage_count);
        """)
        
        self._tables_initialized = True
    
    def generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def generate_query_hash(self, question: str, document_hash: str) -> str:
        """Generate hash for query caching"""
        combined = f"{question.lower().strip()}:{document_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def is_document_cached(self, url_hash: str) -> bool:
        """Check if document is already processed and cached"""
        if not self.is_available():
            return False
        
        await self.ensure_tables_exist()
        
        try:
            result = self.client.table("documents").select("id").eq("url_hash", url_hash).eq("processing_status", "completed").execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"❌ Document cache check failed: {e}")
            return False
    
    async def get_cached_document_info(self, url_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached document information with intelligence data"""
        if not self.is_available():
            return None
        
        await self.ensure_tables_exist()
        
        try:
            result = self.client.table("documents").select(
                "document_name, domain_type, complexity_level, total_clauses, intelligence_data, created_at"
            ).eq("url_hash", url_hash).eq("processing_status", "completed").execute()
            
            if result.data:
                doc = result.data[0]
                
                # Update last accessed time
                self.client.table("documents").update({
                    "last_accessed": datetime.now().isoformat()
                }).eq("url_hash", url_hash).execute()
                
                return {
                    "document_name": doc["document_name"],
                    "domain_type": doc["domain_type"],
                    "complexity_level": doc["complexity_level"],
                    "total_clauses": doc["total_clauses"],
                    "intelligence": doc["intelligence_data"] or {},
                    "created_at": doc["created_at"]
                }
            return None
            
        except Exception as e:
            print(f"❌ Document info retrieval failed: {e}")
            return None
    
    async def get_cached_clauses(self, url_hash: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get cached clause metadata (references only)
        Full content should be retrieved from Pinecone when needed
        """
        if not self.is_available():
            return []
        
        await self.ensure_tables_exist()
        
        try:
            # Get document ID first
            doc_result = self.client.table("documents").select("id").eq("url_hash", url_hash).execute()
            
            if not doc_result.data:
                return []
            
            document_id = doc_result.data[0]["id"]
            
            # Get clause metadata (not full content)
            result = self.client.table("clauses").select("""
                clause_id, content, section, section_title, word_count, 
                char_count, document_type, relevance_score, metadata
            """).eq("document_id", document_id).order("relevance_score", desc=True).order("section", desc=False).limit(limit).execute()
            
            clauses = []
            for row in result.data:
                metadata = row["metadata"] or {}
                
                # Create clause with reference info
                clause = {
                    "id": row["clause_id"],
                    "content": metadata.get("content_preview", f"[Content in Pinecone: {row['clause_id']}]"),  # Preview or reference
                    "section": row["section"],
                    "section_title": row["section_title"],
                    "metadata": {
                        "word_count": row["word_count"],
                        "char_count": row["char_count"],
                        "document_type": row["document_type"],
                        "storage_location": metadata.get("storage_location", "pinecone"),
                        **metadata
                    },
                    "score": row["relevance_score"]
                }
                clauses.append(clause)
            
            print(f"✅ Retrieved {len(clauses)} clause references from Supabase")
            return clauses
            
        except Exception as e:
            print(f"❌ Clause metadata retrieval failed: {e}")
            return []
    
    async def store_document_with_clauses(self, document_data: Dict[str, Any], 
                                        clauses: List[Dict[str, Any]]) -> bool:
        """
        Store document metadata and clause references (NOT full content)
        Full content is stored in Pinecone for vector search
        """
        if not self.is_available():
            return False
        
        await self.ensure_tables_exist()
        
        try:
            # Store document metadata only
            doc_record = {
                "url_hash": document_data["url_hash"],
                "document_name": document_data["document_name"],
                "original_url": document_data.get("original_url"),
                "domain_type": document_data.get("domain_type"),
                "complexity_level": document_data.get("complexity_level"),
                "total_clauses": len(clauses),
                "file_size": document_data.get("file_size"),
                "intelligence_data": document_data.get("intelligence", {}),
                "processing_status": "completed"
            }
            
            doc_result = self.client.table("documents").upsert(doc_record, on_conflict="url_hash").execute()
            
            if not doc_result.data:
                print("❌ Document metadata storage failed")
                return False
            
            document_id = doc_result.data[0]["id"]
            
            # Store ONLY clause metadata and references (not full content)
            clause_records = []
            for clause in clauses:
                # Generate hash for deduplication
                content_preview = clause.get("content", "")[:100]  # Only first 100 chars
                content_hash = self.generate_content_hash(content_preview)
                
                clause_record = {
                    "clause_id": clause["id"],
                    "document_id": document_id,
                    "content": f"[Stored in Pinecone: {clause['id']}]",  # Reference only, not full content
                    "content_hash": content_hash,
                    "section": clause.get("section", 0),
                    "section_title": clause.get("section_title", ""),
                    "word_count": clause.get("metadata", {}).get("word_count", 0),
                    "char_count": clause.get("metadata", {}).get("char_count", 0),
                    "document_type": clause.get("metadata", {}).get("document_type", "general"),
                    "relevance_score": clause.get("score", 1.0),
                    "metadata": {
                        **clause.get("metadata", {}),
                        "storage_location": "pinecone",  # Indicates where full content is stored
                        "content_preview": content_preview  # Small preview only
                    }
                }
                clause_records.append(clause_record)
            
            # Batch insert clause metadata (efficient storage)
            if clause_records:
                clause_result = self.client.table("clauses").upsert(clause_records, on_conflict="clause_id").execute()
                
                if not clause_result.data:
                    print("⚠️ Some clause metadata may not have been stored")
            
            print(f"✅ Stored metadata for {len(clauses)} clauses in Supabase (content in Pinecone)")
            return True
            
        except Exception as e:
            print(f"❌ Supabase metadata storage failed: {e}")
            return False
    
    async def get_cached_answer(self, question: str, document_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached answer for instant response"""
        if not self.is_available():
            return None
        
        await self.ensure_tables_exist()
        
        query_hash = self.generate_query_hash(question, document_hash)
        
        try:
            result = self.client.table("query_cache").select(
                "answer, confidence_score, clause_ids, processing_time, hit_count"
            ).eq("query_hash", query_hash).execute()
            
            if result.data:
                cache_entry = result.data[0]
                
                # Update usage statistics
                self.client.table("query_cache").update({
                    "hit_count": cache_entry["hit_count"] + 1,
                    "last_used": datetime.now().isoformat()
                }).eq("query_hash", query_hash).execute()
                
                return {
                    "answer": cache_entry["answer"],
                    "confidence_score": cache_entry["confidence_score"],
                    "clause_ids": cache_entry["clause_ids"] or [],
                    "processing_time": cache_entry["processing_time"],
                    "hit_count": cache_entry["hit_count"] + 1,
                    "cached": True
                }
            return None
            
        except Exception as e:
            print(f"❌ Cache retrieval failed: {e}")
            return None
    
    async def cache_answer(self, question: str, document_hash: str, answer: str,
                          clause_ids: List[str], processing_time: float, 
                          confidence_score: float = 1.0) -> bool:
        """Cache answer for future instant responses"""
        if not self.is_available():
            return False
        
        await self.ensure_tables_exist()
        
        query_hash = self.generate_query_hash(question, document_hash)
        
        try:
            cache_record = {
                "query_hash": query_hash,
                "document_hash": document_hash,
                "question": question,
                "answer": answer,
                "confidence_score": confidence_score,
                "clause_ids": clause_ids,
                "processing_time": processing_time
            }
            
            result = self.client.table("query_cache").upsert(cache_record, on_conflict="query_hash").execute()
            return bool(result.data)
            
        except Exception as e:
            print(f"❌ Answer caching failed: {e}")
            return False
    
    async def find_similar_cached_answers(self, question: str, document_hash: str, 
                                        similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar cached answers for semantic matching"""
        if not self.is_available():
            return []
        
        await self.ensure_tables_exist()
        
        try:
            # Get popular cached answers for this document
            result = self.client.table("query_cache").select(
                "question, answer, confidence_score, hit_count"
            ).eq("document_hash", document_hash).gt("hit_count", 1).order("hit_count", desc=True).order("last_used", desc=True).limit(10).execute()
            
            similar_answers = []
            question_words = set(question.lower().split())
            
            for row in result.data:
                cached_question = row["question"]
                cached_words = set(cached_question.lower().split())
                
                # Calculate word overlap similarity
                intersection = question_words & cached_words
                union = question_words | cached_words
                similarity = len(intersection) / len(union) if union else 0
                
                if similarity >= similarity_threshold:
                    similar_answers.append({
                        "question": cached_question,
                        "answer": row["answer"],
                        "confidence_score": row["confidence_score"],
                        "hit_count": row["hit_count"],
                        "similarity": similarity
                    })
            
            return sorted(similar_answers, key=lambda x: x["similarity"], reverse=True)
            
        except Exception as e:
            print(f"❌ Similar answers search failed: {e}")
            return []
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get Supabase database performance statistics"""
        if not self.is_available():
            return {"error": "Supabase not available"}
        
        await self.ensure_tables_exist()
        
        try:
            # Document stats
            doc_result = self.client.table("documents").select("id, total_clauses").execute()
            doc_count = len(doc_result.data)
            avg_clauses = sum(doc.get("total_clauses", 0) for doc in doc_result.data) / max(doc_count, 1)
            
            # Clause stats
            clause_result = self.client.table("clauses").select("id, word_count").execute()
            clause_count = len(clause_result.data)
            avg_words = sum(clause.get("word_count", 0) for clause in clause_result.data) / max(clause_count, 1)
            
            # Cache stats
            cache_result = self.client.table("query_cache").select("hit_count").gt("hit_count", 1).execute()
            cached_queries = len(cache_result.data)
            total_hits = sum(cache.get("hit_count", 0) for cache in cache_result.data)
            avg_hits = total_hits / max(cached_queries, 1)
            
            # Recent documents
            recent_result = self.client.table("documents").select("document_name, last_accessed").order("last_accessed", desc=True).limit(5).execute()
            
            return {
                "documents": {
                    "total": doc_count,
                    "avg_clauses": round(avg_clauses, 2)
                },
                "clauses": {
                    "total": clause_count,
                    "avg_words": round(avg_words, 2)
                },
                "cache": {
                    "cached_queries": cached_queries,
                    "avg_hits": round(avg_hits, 2),
                    "total_cache_hits": total_hits,
                    "cache_hit_rate": f"{((total_hits / max(cached_queries, 1)) - 1) * 100:.1f}%"
                },
                "recent_documents": [
                    {"name": doc["document_name"], "accessed": doc["last_accessed"]} 
                    for doc in recent_result.data
                ]
            }
            
        except Exception as e:
            print(f"❌ Stats retrieval failed: {e}")
            return {"error": str(e)}
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Optimize Supabase database for better performance"""
        if not self.is_available():
            return {"status": "failed", "error": "Supabase not available"}
        
        await self.ensure_tables_exist()
        
        try:
            # Clean old cache entries (older than 7 days with low hit count)
            cutoff_date = (datetime.now() - timedelta(days=7)).isoformat()
            
            # Delete old low-usage cache entries
            delete_result = self.client.table("query_cache").delete().lt("created_at", cutoff_date).lte("hit_count", 2).execute()
            deleted_count = len(delete_result.data) if delete_result.data else 0
            
            # Clean old semantic cache
            self.client.table("semantic_cache").delete().lt("created_at", cutoff_date).lte("usage_count", 1).execute()
            
            return {
                "status": "optimized",
                "deleted_cache_entries": deleted_count,
                "optimization_time": datetime.now().isoformat(),
                "database": "Supabase"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def get_smart_clause_recommendations(self, question: str, document_hash: str, 
                                             limit: int = 10) -> List[Dict[str, Any]]:
        """Get smart clause recommendations based on question and past queries"""
        if not self.is_available():
            return []
        
        await self.ensure_tables_exist()
        
        try:
            # Complex query to get clauses from successful past queries
            # Note: This would be more efficient with a proper SQL join in Supabase
            
            # First get successful queries for this document
            cache_result = self.client.table("query_cache").select(
                "clause_ids, confidence_score, hit_count"
            ).eq("document_hash", document_hash).gt("confidence_score", 0.7).order("hit_count", desc=True).limit(20).execute()
            
            # Collect clause IDs from successful queries
            popular_clause_ids = set()
            for cache_entry in cache_result.data:
                clause_ids = cache_entry.get("clause_ids", [])
                if isinstance(clause_ids, list):
                    popular_clause_ids.update(clause_ids)
            
            if not popular_clause_ids:
                return []
            
            # Get details for popular clauses
            clause_result = self.client.table("clauses").select(
                "clause_id, content, relevance_score, section_title"
            ).in_("clause_id", list(popular_clause_ids)).order("relevance_score", desc=True).limit(limit).execute()
            
            recommendations = []
            for clause in clause_result.data:
                recommendations.append({
                    "clause_id": clause["clause_id"],
                    "content": clause["content"][:200] + "..." if len(clause["content"]) > 200 else clause["content"],
                    "relevance_score": clause["relevance_score"],
                    "section_title": clause["section_title"],
                    "popularity": "high"  # Since these came from successful queries
                })
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Recommendations retrieval failed: {e}")
            return []
    
    async def store_semantic_cache(self, question_embedding: List[float], question_text: str,
                                 document_hash: str, answer: str) -> bool:
        """Store semantic cache for similar question matching"""
        if not self.is_available():
            return False
        
        await self.ensure_tables_exist()
        
        try:
            semantic_record = {
                "question_embedding": question_embedding,
                "question_text": question_text,
                "document_hash": document_hash,
                "answer": answer,
                "similarity_threshold": 0.85
            }
            
            result = self.client.table("semantic_cache").insert(semantic_record).execute()
            return bool(result.data)
            
        except Exception as e:
            print(f"❌ Semantic cache storage failed: {e}")
            return False

# Global high-performance Supabase service
supabase_db_service = HighPerformanceSupabaseService()