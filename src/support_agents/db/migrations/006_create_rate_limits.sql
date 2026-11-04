-- Per-user request counts for fixed time windows, shared by all API workers.
CREATE TABLE IF NOT EXISTS rate_limits (
    user_id TEXT NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, window_start)
);

CREATE INDEX IF NOT EXISTS rate_limits_window_start_idx ON rate_limits (window_start);
