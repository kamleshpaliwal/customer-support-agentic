-- Hash of each chunk's text, so re-ingestion can reuse vectors for unchanged chunks.
-- Equals Python's hashlib.sha256(content.encode("utf-8")).hexdigest().
ALTER TABLE academic_documents
    ADD COLUMN IF NOT EXISTS content_hash CHAR(64);

UPDATE academic_documents
    SET content_hash = encode(sha256(convert_to(content, 'UTF8')), 'hex')
    WHERE content_hash IS NULL;

CREATE INDEX IF NOT EXISTS academic_documents_source_hash_idx
    ON academic_documents (source_id, content_hash);
