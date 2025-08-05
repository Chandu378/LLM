-- High-Performance Supabase Tables for Document Query System
-- Copy and paste this entire SQL into your Supabase SQL Editor

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

-- Performance Indexes for Fast Queries
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

-- Insert a test record to verify tables are working
INSERT INTO documents (url_hash, document_name, domain_type, total_clauses) 
VALUES ('test_hash', 'Test Document', 'general', 0) 
ON CONFLICT (url_hash) DO NOTHING;

-- Show success message
SELECT 'High-Performance Supabase Tables Created Successfully! 🚀' as status;