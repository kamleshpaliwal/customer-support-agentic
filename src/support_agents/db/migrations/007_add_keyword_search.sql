-- Full-text index for the keyword half of hybrid retrieval. The column is generated, so ingestion
-- needs no change and existing rows are filled when the column is added.
ALTER TABLE academic_documents
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS academic_documents_content_tsv_idx
    ON academic_documents USING GIN (content_tsv);
