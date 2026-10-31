-- Maps each support request (graph thread) to the authenticated user who started it.
CREATE TABLE IF NOT EXISTS request_owners (
    request_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS request_owners_created_at_idx ON request_owners (created_at);
