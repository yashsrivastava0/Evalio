-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Confirm extension loaded
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
