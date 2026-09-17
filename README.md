# readable-studio-vdi-probe

Read-only diagnostic for a Databricks serving endpoint that rejects tool calls.
Standard library Python only. The workspace host and token are never printed.

## Run

```bash
python test.py --selftest

python test.py --endpoint system.ai.databricks-claude-sonnet-5 \
               --control system.ai.gpt-oss-120b \
               --host https://<workspace>.cloud.databricks.com \
               --token <pat>
```

`--selftest` runs offline with no credentials and must print `SELFTEST PASSED`
before you trust a run. It exercises every classifier, every request-body
builder and the result encoding.

Optional: `--dump detail.json` keeps a full local transcript for your own
reading, `--insecure` skips TLS verification behind a corporate proxy,
`--verbose` shows per-cell progress, `--only A,C,E` restricts paths.

## What to relay

Only the block printed under `RESULT CODE`. It is **uppercase letters**,
grouped in fours, roughly 35-80 characters. Nothing else needs copying.

```
=== RESULT CODE (letters only - relay exactly this) ===
IDXU VPUQ ZPLA ZAMB LPVZ ...
```

Two trailing check letters detect a mistyped character, so a bad copy is
reported rather than silently misread. Spaces and lower case are ignored on
decode. `python test.py --decode "<code>"` expands it back to the full matrix.

A line reading `FAILCODE` means the probe itself broke - that is not a result.
Inside the matrix, `X1`/`X2`/`X3` mark cells the script could not attempt.
Script faults and findings are never mixed.

## What it decides

For each of eight candidate request paths it sends twenty single-variable
requests: minimal, system prompt, three tool-envelope shapes, empty tools,
three tool_choice modes, parallel-off, reasoning effort, alternate max-token
field, streaming, stream options, two tools, strict schema, a tool-result round
trip, anthropic_version, block-form system, and reasoning without tools. It
also records endpoint type, task, entity kind, AI Gateway presence and the
declared supported APIs, plus any parameter name the server names as rejected.

That isolates whether the rejection is the path, the tool envelope shape, one
specific parameter, or the combination - in a single run.

## Finding the endpoint name

A Unity Catalog model name such as `system.ai.<model>` is not necessarily the
serving-endpoint name, so probing it can return 404 on every path including the
metadata API. That means no endpoint carries that name - not that the model
rejects tools.

The probe now lists the endpoints the token can see, matches the requested name
against them, and probes the resolved name, so one run still answers the
question. The relayed code records how the name was obtained (exact, resolved,
ambiguous, or unlistable) and how many endpoints were visible, so a wrong-name
run can never be mistaken for a working endpoint that rejects tools.

Run `python test.py --list` to print the visible endpoint names on screen.
