# Databricks VDI tool-envelope A/B probe

A readable, Python-standard-library-only probe comparing these served entities on
**the same `/serving-endpoints/{endpointName}/invocations` surface**:

- Claude: `system.ai.databricks-claude-sonnet-5`
- GPT: `system.ai.gpt-oss-120b` (normally endpoint `databricks-gpt-oss-120b`)

GPT already works in Readable Studio on the corporate VDI; Claude alone rejects
`tools`. This version therefore replaces the old ai-gateway prefix sweep with a
paired model comparison. It does not assume those prefixes exist on the VDI.

## Run

Requires Python 3.9+ and an authenticated Databricks CLI supporting `auth describe
--output json` and `auth token`. No pip install, Node dependency, or configuration
editing is needed. Only `test.py` needs to be brought onto the VDI.

```console
python test.py
```

Explicit profile (works in PowerShell, cmd.exe, and bash):

```console
python test.py --profile YOUR_PROFILE
```

`DATABRICKS_PROFILE` is also supported. The CLI resolves the host and obtains the
token. No token or raw CLI output is printed or written to a file. Errors are
one-line messages rather than tracebacks. Gateway rejection messages appear only
in the local human-readable table, with the current credential redacted and
whitespace normalized. No telemetry or result upload exists.

The probe calls `GET /api/2.0/serving-endpoints` and matches the **served entity**
(`config.served_entities[].entity_name`, or legacy `served_models[].model_name`),
not a guessed endpoint name. When several endpoints match, it prefers the
canonical name, then alphabetical order; the chosen name is printed locally.
It does not select pending configurations or silently fall back to UC APIs.
A missing entity skips that model, still producing a complete code. Failed
endpoint discovery also produces a code, with its failure in the discovery slot.
Authentication/CLI failures instead exit nonzero with a one-line error.

Each matched endpoint receives 11 tiny POST requests, sequentially per model;
the two model tracks run concurrently. Each HTTP operation has a 20-second
socket timeout (not a total wall-clock deadline). Requests can incur normal
inference charges. No endpoint is changed, and no function is executed.

## Retype only the coded result

The output ends with **one 60-character line**, format:

```text
V2:dd:cccc/cccc/cccc/cccc/cccc/cccc/cccc/cccc/cccc/cccc/cccc
```

- `V2` identifies this format (not compatible with the old A/C/R format).
- `dd` is the two-character outcome of endpoint discovery.
- There are exactly **11 slash-separated positions**, in the table order below.
- Each four-character position is **Claude's two characters, then GPT's two**.
- Example `4T2.` means Claude rejected `tools` with a non-404 4xx, GPT accepted.
- Example `--2.` means no matching Claude endpoint, GPT accepted.
- The local table also prints exact HTTP statuses, named paths, gateway rejection
  messages, and whether the two codes differ. Identical codes are not proof of
  identical full error messages.

### Every position

All requests use the same `messages` and `max_tokens: 32`. The endpoint selects
the model; there is no body `model` field. The prompt asks for `OK`, so acceptance
means HTTP success, **not proof of tool execution or correct tool-call output**.

| Position | Label | Request | Compare against |
| --- | --- | --- | --- |
| 1 | B | Baseline, no `tools` or tool controls | Other model's B |
| 2 | O | B + one OpenAI function tool | B |
| 3 | A | B + one Anthropic tool | B and O |
| 4 | OC | O + `tool_choice: "auto"` | O |
| 5 | OP | O + `parallel_tool_calls: false` | O |
| 6 | OS | O + `tools[0].function.strict: true` | O |
| 7 | AC | A + `tool_choice: {"type":"auto"}` | A |
| 8 | AP | A + `parallel_tool_calls: false` | A |
| 9 | AS | A + `tools[0].strict: true` | A |
| 10 | OT | O + top-level `strict: true` | O |
| 11 | AT | A + top-level `strict: true` | A |

Variants are independent, never cumulative. Each control probe adds exactly one
field to its indicated parent; OS and AS change only the nested `strict` field.
`tool_choice` uses the native form for its tool family. Cross-family choice forms,
forced calls, streaming, and combinations of controls are not tested. AP, AS, OT,
and AT are intentional compatibility tests, not claims that these fields are
valid on every protocol. If A already fails, failures in AC/AP/AS/AT cannot by
themselves implicate the newly added field; the earlier error may mask it.

The tools share the same name, description, and strict-compatible schema:

```json
{"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false}
```

OpenAI envelope: `{"type":"function","function":{"name":"echo","description":"Return x.","parameters":SCHEMA}}`.
Anthropic envelope: `{"name":"echo","description":"Return x.","input_schema":SCHEMA}`.
`SCHEMA` above is a documentation placeholder for the actual schema object.

### Every two-character outcome

| Code / first character | Meaning |
| --- | --- |
| `--` | No matching endpoint, or discovery failed; no POST attempted |
| `0N` | Transport failure / timeout (not an HTTP response) |
| `2.` | HTTP 2xx, request accepted |
| `N` + field character | HTTP **404** (distinct from other 4xx) |
| `4` + field character | Other HTTP 4xx, including 400/401/403/429 |
| `5` + field character | HTTP 5xx |
| `3` + field character | Unresolved HTTP 3xx |

For an HTTP error, the second character identifies the named rejection field:

| Character | Field |
| --- | --- |
| `.` | No explicitly named rejection field |
| `T` | `tools` |
| `F` | `function` |
| `N` | `name` (so `4N` is a rejected name, **not** a network failure) |
| `J` | `parameters` |
| `I` | `input_schema` |
| `C` | `tool_choice` |
| `P` | `parallel_tool_calls` |
| `S` | `strict` |
| `M` | `max_tokens` |
| `G` | `messages` |
| `D` | `model` |
| `X` | Another explicitly named field; full name is in the local table |

**Lowercase means the gateway explicitly supplied a nested path**, e.g.
`Rejected parameter: tools[0].function.strict` -> `4s`,
`tools.0.input_schema` -> `4i`. Uppercase means it supplied a bare field; it does
not prove that field was top-level. For example `unknown field "name"` -> `4N`,
even if the request's name was nested inside a tool.

Extraction prioritizes `Rejected parameter: PATH`, then structured `error.param`,
then explicit unknown/unsupported/invalid-field validation messages. A mere
mention of `tools` is **not** labeled a tools rejection. `Rejected parameter:
tools` -> `4T`; `Rejected parameter: tools[0].function` -> `4f`. The short code
necessarily reduces prose and HTTP status detail; the local table retains them.

## How to interpret the A/B

1. Compare B: both `2.` means both minimal requests work on this surface. If
   Claude B fails, do not attribute its later errors to tools alone.
2. Compare O and A: Claude O rejected but Claude A accepted, with GPT O accepted,
   isolates a model-specific tool-envelope mismatch. The reverse identifies an
   OpenAI-only route. Both failing can be policy or model-adapter validation;
   this probe does not infer which without additional gateway evidence.
3. Compare each control variant with its parent. O accepted but OP rejected
   isolates `parallel_tool_calls`; O accepted but OS rejected isolates nested
   strict. OT separately tests top-level strict. A rejected on its own makes its
   control variants inconclusive about the added control.
4. Read lowercase field codes as explicit nested-field evidence rather than a
   blanket ban on tools. 404 (`N.`) is not the same as a working baseline followed
   by a 400 tools rejection (`4T`). Other 4xx may still be auth, permissions, rate
   limits, or validation; exact statuses are visible locally.

## Verified personal-workspace baseline

Run on 2026-09-15 using:

```console
python test.py --profile dbc-48383de7-db32
```

Exact output:

```text
V2:2.:--2./--2./--4N/--2./--4P/--2./--4N/--4N/--4N/--4S/--4N
```

Discovery was HTTP 200. This workspace has `databricks-gpt-oss-120b` serving
`system.ai.gpt-oss-120b`, but **no matching Claude serving endpoint**. Its Claude
is available as a UC model service, which is intentionally not substituted: that
would change both the model and the surface and invalidate the controlled A/B.
Consequently every Claude slot is `--`; this baseline verifies GPT and graceful
asymmetry, not corporate Claude behavior.

GPT results: B/O/OC/OS HTTP 200; A/AC/AP/AS/AT HTTP 400 with
`Bad request: json: unknown field "name"`; OP HTTP 400 naming
`parallel_tool_calls`; OT HTTP 400 naming `strict`.

## Maintainer verification

```console
python -m unittest discover -p test_probe.py -v
python -m py_compile test.py test_probe.py
```

Tests cover endpoint matching, exact one-field variants, rejection-path coding,
credential-safe CLI failures, missing endpoints, and the real urllib/HTTP request
and printed-code integration using a local HTTP server. No corporate credential
or timing sleeps are used. The live command above is the workspace verification.
