---
name: python-review
context: fork
description: Deep Python code quality review. Auto-invoke when finishing a task, before marking work complete, when the user asks to review code, or when preparing a PR. Focuses on design judgment, naming, performance, and test quality — things that ruff and ty cannot catch.
---

# Deep Code Review

Review the changes for design quality — what automated tools miss. Report findings as 🔴 Must Fix, 🟡 Should Fix, 🟢 Suggestion.

First, confirm `uv run poe check` passes (ruff, ty, tests). Fix those before proceeding.

---

## 1. Function Design

Prefer pure functions: data in, data out, no side effects. When side effects are necessary (I/O, logging, state mutation), isolate them — push them to the edges, keep the core logic pure and testable.

- Functions that both compute and mutate → split into a pure computation and a thin side-effecting wrapper.
- Over ~30 lines → smell. Justify or split.
- Mixed abstraction levels (HTTP parsing interleaved with business logic).
- \>5 parameters → group into a dataclass or split.
- Boolean params make call sites unreadable (`process(data, True, False)`) → enum or separate functions.
- `@staticmethod` on a single-method class, or classes with only `__init__` + one method → just a function.
- Inheritance for code reuse where composition is simpler.

## 2. Naming

- Single-letter variables outside comprehensions/lambdas. `d`, `x`, `r` — name the thing.
- Generic names that mean nothing: `data`, `result`, `info`, `item`, `obj`, `tmp`, `val`, `manager`, `handler`, `processor`, `helper`, `utils`. Name the *what*.
- Misleading: `users` that's a count, `get_` that mutates, `status` that's a bool.
- Inconsistent: `user_id` / `userId` / `uid` across functions.
- Unnecessary abbreviation. `cfg` is fine. `proc_dat_xfrm` is not.
- Negated booleans (`not_found`, `is_not_valid`) → invert the name.

## 3. Comments

Comments exist for *why*, never *what*.

- Restating code (`# increment counter` above `counter += 1`) → delete.
- Commented-out code → it's in git, delete.
- TODOs without a ticket reference → link to tracker or delete.
- *Missing* why-comments on magic numbers, workarounds, performance hacks, non-obvious decisions.
- Docstrings that parrot the signature (`"""Gets the user."""` on `get_user()`) → explain behavior/constraints/edge cases or remove.
- Excessive inline comments on straightforward code → the code is too clever, simplify it.

## 4. Error Handling Design

Strategy, not syntax (ruff handles syntax).

- `try` blocks wrapping 20 lines → wrap only what can throw.
- `except Exception as e: raise Exception(str(e))` → destroys traceback/type. `raise` or wrap in domain exception.
- `raise OtherError("failed")` discarding original → chain with `from e`.
- Error types that are `str` or bare `Exception` → domain-specific exceptions.
- `assert` for runtime validation in non-test code → stripped by `-O`, use `if`/`raise`.
- String-based error discrimination (`if "not found" in str(e)`) → exception types.
- Inconsistent strategy within a module: some functions return `None`, others raise. Pick one.

## 5. Performance

Patterns that cause real problems, not micro-optimization.

- `in` on a `list` that should be a `set` (>~20 elements: O(n) vs O(1)).
- `+` string concatenation in a loop → `"".join()`.
- `await` in a loop → `asyncio.gather()` / `TaskGroup`.
- N+1: querying a DB or API per item instead of batching.
- `f.read()` on large files when line-by-line streaming works.
- Repeated computation inside a loop → hoist or `@cache`.
- Building a list only to immediately iterate it → use the iterator.
- Unnecessary `.copy()` / `deepcopy` → restructure ownership.

## 6. Test Quality

- No assertions — calling code without checking results is not a test.
- Weak assertions: `assert result is not None` without checking actual value.
- Mocking 5 things → testing setup, not behavior. Restructure for testability.
- 3+ tests differing only in input/output → `@pytest.mark.parametrize`.
- Missing edge cases: empty input, error paths, boundary conditions.
- Useless names: `test_process_data` → `test_returns_error_on_empty_input`.
- 20 lines of setup for 2 lines of test → extract fixtures.

## 7. Duplicated Logic

- Functions/blocks doing the same thing with minor variations → extract with concrete params.
- Watch for: HTTP handling, validation, serialization boilerplate, similar conditional chains.
- Copy-paste with slight modifications — the second copy will drift into a bug.

## 8. Module Design

- God modules (>500 lines, mixed responsibilities) → suggest split.
- In-function imports for lazy loading must have a `# lazy:` comment. Uncommented → why?
- Side effects at import time (starting servers, DB connections, thread spawning at module level) → 🔴. Import must be inert.
- Public names (`no _` prefix) that aren't intended API.
- CLI entrypoints containing logic instead of parsing args and delegating.

## 9. Dependencies

- Mixed HTTP clients (`requests` + `httpx`) → pick one.
- Mixed serialization (`json` + `orjson`/`msgspec`) → use what's in the dep tree.
- Vendored code that should be a dep, or trivial deps that could be a 10-line util.

---

## Output

Group by severity, then area. Each finding: **what** + **where** (file:line) + **why** + **concrete fix**.

End with: X must-fix, Y should-fix, Z suggestions.

$ARGUMENTS
