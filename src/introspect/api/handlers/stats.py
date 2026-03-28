"""Stats route handler."""

import logging

from fastapi import Request
from fastapi.responses import HTMLResponse

from ._helpers import conn, parent, templates


async def stats(request: Request) -> HTMLResponse:
    """Stats and insights page."""
    db = conn(request)

    # Summary metrics
    summary = db.execute("""
        SELECT
            COUNT(*) AS total_sessions,
            MIN(started_at) AS earliest_session,
            SUM(user_messages) AS total_user_messages,
            SUM(user_messages + assistant_messages) AS total_turns
        FROM logical_sessions
    """).fetchone()
    total_sessions = summary[0]
    earliest_session = summary[1]
    total_user_messages = summary[2] or 0
    total_turns = summary[3] or 0

    tool_summary = db.execute("""
        SELECT
            COUNT(*) AS total_tool_calls,
            COUNT(*) FILTER (WHERE is_error = 'true') AS total_failed
        FROM tool_calls
    """).fetchone()
    total_tool_calls = tool_summary[0]
    total_failed = tool_summary[1]

    # Session duration distribution
    duration_buckets = db.execute("""
        SELECT
            bucket,
            COUNT(*) AS cnt
        FROM (
            SELECT
                CASE
                    WHEN duration < INTERVAL '1 minute' THEN '< 1 min'
                    WHEN duration < INTERVAL '5 minutes' THEN '1-5 min'
                    WHEN duration < INTERVAL '15 minutes' THEN '5-15 min'
                    WHEN duration < INTERVAL '30 minutes' THEN '15-30 min'
                    ELSE '30+ min'
                END AS bucket,
                CASE
                    WHEN duration < INTERVAL '1 minute' THEN 1
                    WHEN duration < INTERVAL '5 minutes' THEN 2
                    WHEN duration < INTERVAL '15 minutes' THEN 3
                    WHEN duration < INTERVAL '30 minutes' THEN 4
                    ELSE 5
                END AS sort_order
            FROM logical_sessions
        ) sub
        GROUP BY bucket, sort_order
        ORDER BY sort_order
    """).fetchall()

    # Turns per session distribution
    turns_buckets = db.execute("""
        SELECT
            bucket,
            COUNT(*) AS cnt
        FROM (
            SELECT
                CASE
                    WHEN (user_messages + assistant_messages) <= 2 THEN '1-2 turns'
                    WHEN (user_messages + assistant_messages) <= 5 THEN '3-5 turns'
                    WHEN (user_messages + assistant_messages) <= 10 THEN '6-10 turns'
                    WHEN (user_messages + assistant_messages) <= 20 THEN '11-20 turns'
                    WHEN (user_messages + assistant_messages) <= 50 THEN '21-50 turns'
                    WHEN (user_messages + assistant_messages) <= 100 THEN '51-100 turns'
                    ELSE '100+ turns'
                END AS bucket,
                CASE
                    WHEN (user_messages + assistant_messages) <= 2 THEN 1
                    WHEN (user_messages + assistant_messages) <= 5 THEN 2
                    WHEN (user_messages + assistant_messages) <= 10 THEN 3
                    WHEN (user_messages + assistant_messages) <= 20 THEN 4
                    WHEN (user_messages + assistant_messages) <= 50 THEN 5
                    WHEN (user_messages + assistant_messages) <= 100 THEN 6
                    ELSE 7
                END AS sort_order
            FROM logical_sessions
        ) sub
        GROUP BY bucket, sort_order
        ORDER BY sort_order
    """).fetchall()

    # Tool usage breakdown (non-MCP tools only, with percentage)
    tool_breakdown = db.execute("""
        SELECT
            tool_name,
            COUNT(*) AS cnt,
            100.0 * COUNT(*) FILTER (WHERE is_error IS DISTINCT FROM 'true')
                / COUNT(*) AS success_rate,
            100.0 * COUNT(*) / (SELECT COUNT(*) FROM tool_calls) AS pct_of_total
        FROM tool_calls
        WHERE tool_name IS NOT NULL
          AND tool_name NOT LIKE 'mcp__%'
        GROUP BY tool_name
        ORDER BY cnt DESC
    """).fetchall()

    # MCP usage: servers and their commands
    mcp_breakdown = db.execute("""
        SELECT
            tool_name,
            COUNT(*) AS cnt,
            100.0 * COUNT(*) FILTER (WHERE is_error IS DISTINCT FROM 'true')
                / COUNT(*) AS success_rate,
            100.0 * COUNT(*) / (SELECT COUNT(*) FROM tool_calls) AS pct_of_total
        FROM tool_calls
        WHERE tool_name LIKE 'mcp__%'
        GROUP BY tool_name
        ORDER BY cnt DESC
    """).fetchall()

    mcp_servers = db.execute("""
        SELECT
            split_part(tool_name, '__', 2) AS server_name,
            COUNT(*) AS cnt,
            COUNT(DISTINCT split_part(tool_name, '__', 3)) AS command_count
        FROM tool_calls
        WHERE tool_name LIKE 'mcp__%'
        GROUP BY server_name
        ORDER BY cnt DESC
    """).fetchall()

    # Bash command breakdown
    bash_breakdown = db.execute("""
        SELECT
            split_part(
                trim(json_extract_string(tool_input, '$.command')),
                ' ', 1
            ) AS first_word,
            COUNT(*) AS cnt
        FROM tool_calls
        WHERE tool_name = 'Bash'
          AND json_extract_string(tool_input, '$.command') IS NOT NULL
        GROUP BY first_word
        ORDER BY cnt DESC
    """).fetchall()

    bash_two_words = db.execute("""
        WITH cmds AS (
            SELECT trim(
                json_extract_string(tool_input, '$.command')
            ) AS cmd
            FROM tool_calls
            WHERE tool_name = 'Bash'
              AND json_extract_string(
                  tool_input, '$.command'
              ) IS NOT NULL
        )
        SELECT
            CASE
                WHEN array_length(
                    string_split(cmd, ' ')
                ) >= 2
                THEN split_part(cmd, ' ', 1)
                    || ' '
                    || split_part(cmd, ' ', 2)
                ELSE cmd
            END AS first_two_words,
            COUNT(*) AS cnt
        FROM cmds
        GROUP BY first_two_words
        ORDER BY cnt DESC
        LIMIT 30
    """).fetchall()

    bash_chained = db.execute("""
        SELECT
            COUNT(*) FILTER (
                WHERE json_extract_string(tool_input, '$.command') LIKE '%&&%'
            ) AS chained_count,
            COUNT(*) AS total_bash
        FROM tool_calls
        WHERE tool_name = 'Bash'
          AND json_extract_string(tool_input, '$.command') IS NOT NULL
    """).fetchone()

    # Longest sessions top 15 with all metrics
    longest_sessions = db.execute("""
        SELECT
            ls.session_id, ls.started_at, ls.duration, ls.model,
            ls.user_messages, ls.assistant_messages,
            (ls.user_messages + ls.assistant_messages) AS total_turns,
            COALESCE(tc.tool_count, 0) AS tool_count,
            COALESCE(tc.failed_count, 0) AS failed_count
        FROM logical_sessions ls
        LEFT JOIN (
            SELECT
                session_id,
                COUNT(*) AS tool_count,
                COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
            FROM tool_calls
            GROUP BY session_id
        ) tc ON ls.session_id = tc.session_id
        ORDER BY ls.duration DESC
        LIMIT 15
    """).fetchall()

    # Most tool calls sessions top 15 with all metrics
    most_tools_sessions = db.execute("""
        SELECT
            ls.session_id,
            COALESCE(tc.tool_count, 0) AS tool_count,
            COALESCE(tc.failed_count, 0) AS failed_count,
            ls.started_at, ls.duration, ls.model,
            ls.user_messages, ls.assistant_messages,
            (ls.user_messages + ls.assistant_messages) AS total_turns
        FROM logical_sessions ls
        LEFT JOIN (
            SELECT
                session_id,
                COUNT(*) AS tool_count,
                COUNT(*) FILTER (WHERE is_error = 'true') AS failed_count
            FROM tool_calls
            GROUP BY session_id
        ) tc ON ls.session_id = tc.session_id
        ORDER BY tc.tool_count DESC NULLS LAST
        LIMIT 15
    """).fetchall()

    # Sessions per day
    sessions_per_day = db.execute("""
        SELECT
            CAST(started_at AS DATE) AS day,
            COUNT(*) AS cnt
        FROM logical_sessions
        GROUP BY day
        ORDER BY day DESC
        LIMIT 30
    """).fetchall()

    # Token usage summary (best effort)
    try:
        token_usage = db.execute("""
            SELECT
                SUM(CAST(json_extract(message, '$.usage.input_tokens') AS BIGINT)),
                SUM(CAST(json_extract(message, '$.usage.output_tokens') AS BIGINT))
            FROM raw_messages
            WHERE type = 'assistant'
              AND json_extract(message, '$.usage.input_tokens') IS NOT NULL
        """).fetchone()
    except Exception:
        logging.getLogger(__name__).debug("token usage query failed", exc_info=True)
        token_usage = None

    return templates.TemplateResponse(
        request,
        "stats.html",
        {
            "parent": parent(request),
            "total_sessions": total_sessions,
            "total_tool_calls": total_tool_calls,
            "total_failed": total_failed,
            "total_user_messages": total_user_messages,
            "total_turns": total_turns,
            "earliest_session": earliest_session,
            "duration_buckets": duration_buckets,
            "turns_buckets": turns_buckets,
            "tool_breakdown": tool_breakdown,
            "mcp_breakdown": mcp_breakdown,
            "mcp_servers": mcp_servers,
            "bash_breakdown": bash_breakdown,
            "bash_two_words": bash_two_words,
            "bash_chained": bash_chained,
            "longest_sessions": longest_sessions,
            "most_tools_sessions": most_tools_sessions,
            "sessions_per_day": sessions_per_day,
            "token_usage": token_usage,
        },
    )
