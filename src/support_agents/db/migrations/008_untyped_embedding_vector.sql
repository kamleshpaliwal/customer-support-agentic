-- Let one schema serve every embedding model: a local Ollama database (768 dimensions) and a
-- Bedrock one (e.g. 1024). Each row already records its ``embedding_model`` and searches filter
-- on it, so vectors of different sizes are never compared. Existing rows are kept as they are.
ALTER TABLE academic_documents
    ALTER COLUMN embedding TYPE vector;

-- An HNSW index needs a fixed dimension. The corpus is small enough to scan; add a per-model
-- expression index (embedding::vector(N)) if it grows.
DROP INDEX IF EXISTS academic_documents_embedding_hnsw_idx;
