-- Sample DuckDB queries for Claude Code JSONL data
-- Usage: duckdb -c ".read queries.sql"

-- 1. Load all messages from all sessions
SELECT
    filename AS file_path,
    type,
    timestamp,
    sessionId AS session_id,
    uuid,
    parentUuid AS parent_uuid,
    json_extract_string(message, '$.role') AS role,
    json_extract_string(message, '$.model') AS model,
FROM read_json_auto(
    '~/.claude/projects/**/*.jsonl',
    filename=true,
    format='newline_delimited',
    union_by_name=true,
    ignore_errors=true
)
WHERE type IN ('user', 'assistant')
ORDER BY timestamp
LIMIT 50;

-- 2. List all sessions with start/end times and message counts
SELECT
    sessionId AS session_id,
    MIN(timestamp) AS started_at,
    MAX(timestamp) AS ended_at,
    age(MAX(timestamp)::TIMESTAMP, MIN(timestamp)::TIMESTAMP) AS duration,
    COUNT(*) FILTER (WHERE type = 'user' AND json_extract_string(message, '$.role') = 'user') AS user_messages,
    COUNT(*) FILTER (WHERE type = 'assistant') AS assistant_messages,
    ANY_VALUE(json_extract_string(message, '$.model')) AS model,
FROM read_json_auto(
    '~/.claude/projects/**/*.jsonl',
    filename=true,
    format='newline_delimited',
    union_by_name=true,
    ignore_errors=true
)
WHERE type IN ('user', 'assistant')
GROUP BY sessionId
ORDER BY started_at DESC;

-- 3. Extract tool calls with their results
WITH tools AS (
    SELECT
        sessionId AS session_id,
        timestamp,
        uuid,
        json_extract_string(message, '$.content[0].name') AS tool_name,
        json_extract_string(message, '$.content[0].id') AS tool_use_id,
        json_extract_string(message, '$.content[0].input') AS tool_input,
    FROM read_json_auto(
        '~/.claude/projects/**/*.jsonl',
        format='newline_delimited',
        union_by_name=true,
        ignore_errors=true
    )
    WHERE type = 'assistant'
      AND json_extract_string(message, '$.content[0].type') = 'tool_use'
),
results AS (
    SELECT
        json_extract_string(message, '$.content[0].tool_use_id') AS tool_use_id,
        COALESCE(
            json_extract_string(toolUseResult, '$.stderr'),
            ''
        ) AS stderr,
        json_extract(message, '$.content[0].is_error') AS is_error,
    FROM read_json_auto(
        '~/.claude/projects/**/*.jsonl',
        format='newline_delimited',
        union_by_name=true,
        ignore_errors=true
    )
    WHERE type = 'user'
      AND json_extract_string(message, '$.content[0].type') = 'tool_result'
)
SELECT
    t.session_id,
    t.timestamp,
    t.tool_name,
    t.tool_use_id,
    r.is_error,
    LEFT(r.stderr, 200) AS stderr_preview,
FROM tools t
LEFT JOIN results r ON t.tool_use_id = r.tool_use_id
ORDER BY t.timestamp DESC
LIMIT 50;

-- 4. Token usage per session (cost proxy)
SELECT
    sessionId AS session_id,
    SUM(CAST(json_extract(message, '$.usage.input_tokens') AS INTEGER)) AS total_input_tokens,
    SUM(CAST(json_extract(message, '$.usage.output_tokens') AS INTEGER)) AS total_output_tokens,
    SUM(CAST(json_extract(message, '$.usage.cache_read_input_tokens') AS INTEGER)) AS cache_read_tokens,
    SUM(CAST(json_extract(message, '$.usage.cache_creation_input_tokens') AS INTEGER)) AS cache_create_tokens,
    COUNT(*) AS assistant_turns,
FROM read_json_auto(
    '~/.claude/projects/**/*.jsonl',
    format='newline_delimited',
    union_by_name=true,
    ignore_errors=true
)
WHERE type = 'assistant'
  AND json_extract(message, '$.usage') IS NOT NULL
GROUP BY sessionId
ORDER BY total_input_tokens DESC;

-- 5. Tool usage frequency breakdown
SELECT
    json_extract_string(message, '$.content[0].name') AS tool_name,
    COUNT(*) AS call_count,
FROM read_json_auto(
    '~/.claude/projects/**/*.jsonl',
    format='newline_delimited',
    union_by_name=true,
    ignore_errors=true
)
WHERE type = 'assistant'
  AND json_extract_string(message, '$.content[0].type') = 'tool_use'
GROUP BY tool_name
ORDER BY call_count DESC;
