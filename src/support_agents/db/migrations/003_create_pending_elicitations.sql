-- Questions the MCP tools ask the user mid-request (for example, book language).
-- Shared across API workers so any worker can serve the poll and answer requests.
CREATE TABLE IF NOT EXISTS pending_elicitations (
    request_id UUID PRIMARY KEY,
    message TEXT NOT NULL,
    field_name TEXT NOT NULL,
    options JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'cancelled')),
    answer TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS pending_elicitations_expires_at_idx
    ON pending_elicitations (expires_at);
