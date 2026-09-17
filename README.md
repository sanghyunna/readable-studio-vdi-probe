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

`--selftest` runs offline and must print `SELFTEST PASSED` before you trust a run.

Optional: `--dump detail.json` keeps a full local transcript for your own reading,
`--insecure` skips TLS verification behind a corporate proxy, `--verbose` shows
progress, `--only A,C,E` restricts which paths are probed.

## What to relay

Only the final line. It starts with `V4#`.
A line starting with `FAIL#` means the probe itself broke, not that the endpoint failed.

## What it decides

For each of eight candidate request paths it sends twenty single-variable
requests: minimal, system prompt, three tool-envelope shapes, empty tools,
three tool_choice modes, parallel-off, reasoning effort, alternate max-token
field, streaming, stream options, two tools, strict schema, a tool-result round
trip, anthropic_version, block-form system, and reasoning without tools.
That isolates whether the rejection is the path, the tool envelope shape, a
specific parameter, or the combination.
