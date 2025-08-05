#!/usr/bin/env python3
"""
Supabase Service for efficient document metadata storage
Reduces memory usage by storing document information in database
Optional service - system works without it
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import hashlib

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

class SupabaseService:
    def __init__(self):
        """Initialize Supabase client"""
        if not SUPABASE_AVAILABLE:
            self.client = None
            print("❌ Supabase service disabled - package not available")
            return
        
        # Get Supabase credentials from environment
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            self.client = None
            print("⚠️ Supabase credentials not found in environment variables")
            print("   Set SUPABASE_URL and SUPABASE_ANON_KEY to enable Supabase storage")
            return
        
        try:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            print("✅ Supabase service initialized")
            
            # Note: Table initialization will be done on first use
            self._tables_initialized = False
            
        except Exception as e:
            self.client = None
            print(f"❌ Supabase initialization failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Supabase service is available"""
        return self.client is not None
    
    async def _ensure_tables_initialized(self):
        """Ensure database tables are initialized (called on first use)"""
        if not self.is_available() or self._tables_initialized:
            return
        
        try:
            # In production, create these tables via Supabase dashboard SQL editor:
            
            print("📋 Database tables should be created via Supabase dashboard:")
            print("   1. Go to Supabase dashboard > SQL Editor")
            print("   2. Create 'documents' table for document metadata")
            print("   3. Create 'clause_metadata' table for clause information")
            print("   4. Set up proper indexes and constraints")
            
            self._tables_initialized = True
            
        except Exception as e:
            print(f"⚠️ Table initialization check: {e}")
    
    async def store_document_info(self, document_data: Dict[str, Any]) -> bool:
        """Store document information"""
        if not self.is_available():
            return False
        
        # Ensure tables are initialized
        await self._ensure_tables_initialized()
        
        try:
            # Prepare document record
            doc_record = {
                "document_name": document_data.get("document_name"),
                "url_hash": document_data.get("url_hash"),
                "original_url": document_data.get("original_url"),
                "domain_type": document_data.get("domain_type"),
                "complexity_level": document_data.get("complexity_level"),
                "total_clauses": document_data.get("total_clauses", 0),
                "file_size": document_data.get("file_size"),
                "processing_status": "completed",
                "updated_at": datetime.now().isoformat()
            }
            
            # Insert or update document record
            result = self.client.table("documents").upsert(doc_record, on_conflict="url_hash").execute()
            
            if result.data:
                print(f"✅ Document info stored: {document_data.get('document_name')}")
                return True
            else:
                print(f"⚠️ Document storage returned no data")
                return False
                
        except Exception as e:
            print(f"❌ Document info storage failed: {e}")
            return False
    
    async def store_clause_metadata(self, clause_records: List[Dict[str, Any]]) -> bool:
        """Store clause metadata in batch"""
        if not self.is_available():
            return False
        
        # Ensure tables are initialized
        await self._ensure_tables_initialized()
        
        try:
            # Prepare records with timestamps
            for record in clause_records:
                record["created_at"] = datetime.now().isoformat()
                record["embedding_uploaded"] = True  # Assume uploaded if we're storing
            
            # Batch insert clause metadata
            result = self.client.table("clause_metadata").upsert(clause_records, on_conflict="clause_id").execute()
            
            if result.data:
                print(f"✅ Stored metadata for {len(clause_records)} clauses")
                return True
            else:
                print(f"⚠️ Clause metadata storage returned no data")
                return False
                
        except Exception as e:
            print(f"❌ Clause metadata storage failed: {e}")
            return False
    
    async def get_document_info(self, url_hash: str) -> Optional[Dict[str, Any]]:
        """Get document information by URL hash"""
        if not self.is_available():
            return None
        
        try:
            result = self.client.table("documents").select("*").eq("url_hash", url_hash).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            print(f"❌ Document info retrieval failed: {e}")
            return None
    
    async def get_clause_metadata(self, document_name: str) -> List[Dict[str, Any]]:
        """Get clause metadata for a document"""
        if not self.is_available():
            return []
        
        try:
            result = self.client.table("clause_metadata").select("*").eq("document_name", document_name).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"❌ Clause metadata retrieval failed: {e}")
            return []
    
    async def is_document_processed(self, url_hash: str) -> bool:
        """Check if document has been processed"""
        doc_info = await self.get_document_info(url_hash)
        return doc_info is not None and doc_info.get("processing_status") == "completed"
    
    async def mark_processing_started(self, url_hash: str, document_name: str) -> bool:
        """Mark document processing as started"""
        if not self.is_available():
            return False
        
        try:
            doc_record = {
                "url_hash": url_hash,
                "document_name": document_name,
                "processing_status": "processing",
                "created_at": datetime.now().isoformat()
            }
            
            result = self.client.table("documents").upsert(doc_record, on_conflict="url_hash").execute()
            return bool(result.data)
            
        except Exception as e:
            print(f"❌ Processing status update failed: {e}")
            return False
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if not self.is_available():
            return {"error": "Supabase not available"}
        
        try:
            # Get document count
            doc_result = self.client.table("documents").select("count", count="exact").execute()
            doc_count = doc_result.count if doc_result.count else 0
            
            # Get clause count
            clause_result = self.client.table("clause_metadata").select("count", count="exact").execute()
            clause_count = clause_result.count if clause_result.count else 0
            
            # Get recent documents
            recent_result = self.client.table("documents").select("document_name, created_at, total_clauses").order("created_at", desc=True).limit(5).execute()
            recent_docs = recent_result.data if recent_result.data else []
            
            return {
                "total_documents": doc_count,
                "total_clauses": clause_count,
                "recent_documents": recent_docs
            }
            
        except Exception as e:
            print(f"❌ Stats retrieval failed: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_records(self, days_old: int = 30) -> bool:
        """Clean up old records to save space"""
        if not self.is_available():
            return False
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now().replace(day=datetime.now().day - days_old)
            
            # Delete old clause metadata
            clause_result = self.client.table("clause_metadata").delete().lt("created_at", cutoff_date.isoformat()).execute()
            
            # Delete old documents
            doc_result = self.client.table("documents").delete().lt("created_at", cutoff_date.isoformat()).execute()
            
            print(f"✅ Cleaned up records older than {days_old} days")
            return True
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return False

# Global service instance
supabase_service = SupabaseService() if SUPABASE_AVAILABLE else None