# JSONL schema

Reference notes on the shape of the Claude Code conversation logs that
Introspect reads. These describe the raw JSONL on disk, before it is
materialized into the DuckDB views documented in [Architecture](architecture.md).

## File Location
- Conversations stored at `~/.claude/projects/<project-slug>/<session-uuid>.jsonl`
- One file per session, one JSON object per line
- Project slug is derived from the project path (e.g., `-home-user-introspect`)

## Top-Level Record Types (`type` field)

### 1. `queue-operation`
Internal queue management. Two operations:
- `enqueue`: Has `content` (the user prompt text), `sessionId`, `timestamp`
- `dequeue`: Has `sessionId`, `timestamp`

### 2. `user`
User messages and tool results.

**Common fields:**
- `uuid` — unique message ID
- `parentUuid` — links to parent in conversation tree (null for first message)
- `sessionId` — session UUID (matches filename)
- `timestamp` — ISO 8601
- `type` — "user"
- `isSidechain` — boolean
- `cwd` — working directory at time of message
- `version` — Claude Code version string
- `entrypoint` — "remote" | "cli" | etc.
- `userType` — "external"
- `gitBranch` — current git branch

**Variants:**
- **Human prompt**: Has `message.role = "user"`, `message.content` is a string, plus `promptId`, `permissionMode`
- **Tool result**: Has `toolUseResult` object with keys: `stdout`, `stderr`, `interrupted`, `isImage`, `noOutputExpected`. Also has `sourceToolAssistantUUID` linking to the assistant message that made the tool call. The `message.content` is a list with `tool_result` blocks containing `tool_use_id`, `content`, `is_error`.

### 3. `assistant`
Model responses.

**Common fields:** Same as user (`uuid`, `parentUuid`, `sessionId`, `timestamp`, etc.) plus `requestId`.

**`message` object:**
- `model` — model ID (e.g., "claude-opus-4-6")
- `id` — API message ID
- `role` — "assistant"
- `stop_reason` — "end_turn" | "tool_use" | null (streaming)
- `usage` — token counts (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`)
- `content` — array of content blocks

**Content block types:**
- `text`: `{ type: "text", text: "..." }`
- `thinking`: `{ type: "thinking", thinking: "...", signature: "..." }`
- `tool_use`: `{ type: "tool_use", id: "toolu_...", name: "Bash"|"Read"|"Edit"|..., input: {...} }`

### 4. `attachment`
Harness-injected context — not model/user text. Every derived view over
`raw_messages` drops these (it filters to `type IN ('user','assistant')`), so
they survive only in `raw_data`. The `session_context_loads` view reads them
back and classifies the useful subtypes.

**Common fields:** `uuid`, `parentUuid`, `sessionId`, `timestamp`, `cwd`, plus
an `attachment` object (materialized as `MAP(VARCHAR, JSON)`) whose `type` key
names the subtype.

**Observed `attachment.type` subtypes** (▶ = surfaced in `session_context_loads`):
- ▶ `nested_memory` — nested CLAUDE.md / `.claude/rules/*` loaded on directory
  entry. Keys: `path`, `displayPath`, `content`. (The root project CLAUDE.md and
  global `~/.claude/CLAUDE.md` arrive inline as a first-message
  `<system-reminder>`, *not* as an attachment.)
- ▶ `file` — `@`-file expansion. Keys: `filename`, `displayPath`, `content`.
- ▶ `skill_listing` — the skill menu. Keys: `content`, `skillCount`, `names`,
  `isInitial`.
- ▶ `mcp_instructions_delta` — MCP server instruction blocks. Keys:
  `addedNames`, `addedBlocks`, `removedNames`.
- ▶ `hook_success` / `hook_non_blocking_error` — hook output. Keys: `hookName`,
  `content`, `stdout`, `stderr`, `exitCode`, `command`, `durationMs`.
- Dropped as chatter: `output_style`, `total_tokens_reminder`, `task_reminder`,
  `deferred_tools_delta`, `agent_listing_delta`, `queued_command`,
  `edited_text_file`, `command_permissions`, `opened_file_in_ide`,
  `plan_mode` / `plan_mode_exit`, `date_change`, …

### Tool Input Schemas (observed)
- **Bash**: `{ command, description }` — optional: `timeout`, `run_in_background`
- **Read**: `{ file_path }` — optional: `offset`, `limit`, `pages`
- **Edit**: `{ file_path, old_string, new_string }` — optional: `replace_all`
- **Write**: `{ file_path, content }`
- **Glob**: `{ pattern }` — optional: `path`
- **Grep**: `{ pattern }` — many optional params
- **ToolSearch**: `{ query, max_results }`
- **TodoWrite**: `{ todos: [...] }`
- **Agent**: `{ description, prompt }` — optional: `subagent_type`, `isolation`, `run_in_background`

## Key Relationships
- `parentUuid` → `uuid` forms a tree (branching possible via sidechains)
- `tool_use.id` → `tool_result.tool_use_id` links tool calls to results
- `sourceToolAssistantUUID` → assistant `uuid` links tool results to the requesting turn
- `sessionId` is constant per file and matches the filename

## Notes
- Multiple assistant lines can share the same `requestId` (streaming chunks / parallel tool calls)
- `isSidechain: true` marks agent sub-conversations
- `usage` block contains cost-relevant token data
- The `queue-operation` lines always appear as the first 2 lines (enqueue/dequeue pair)
