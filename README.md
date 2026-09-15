# Databricks AI Gateway VDI Probe

A single-file, standard-library-only probe for a corporate VDI that diagnoses why
`system.ai.databricks-claude-sonnet-5` returns HTTP 400 with `rejected parameter: tools`.

## What it does

1. Pulls a bearer token from the Databricks CLI (`databricks auth token`) and the
   workspace host from `~/.databrickscfg`.
2. POSTs tiny requests to three AI Gateway surfaces:
   - `A` Anthropic Messages: `/ai-gateway/anthropic/v1/messages`
   - `C` OpenAI Chat Completions: `/ai-gateway/openai/v1/chat/completions`
   - `R` OpenAI Responses: `/ai-gateway/openai/v1/responses`
3. For each surface it tries four request shapes: baseline, with four tools,
   with a large `max_tokens`, and with `reasoning_effort`.
4. Prints a human-readable table and a short coded string.

## Run it

```bash
python test.py
```

To use a non-default profile:

```bash
set DATABRICKS_PROFILE=your-profile
python test.py
```

## Send the result back

The script prints a single short line like:

```
A:2.2T4M2R/C:2.2T4M2R/R:2.2T2M2R
```

Type that exact string back to us. It is the only thing that needs to leave the VDI.

## Code key

Each surface has four probes. Every probe is two characters:

- First character: HTTP status bucket (`2` = 2xx, `4` = 4xx, `5` = 5xx, `0` = network/error)
- Second character: offending parameter (`T` = tools, `M` = max_tokens,
  `R` = reasoning_effort, `X` = other named parameter, `.` = none, `N` = network error)

Order within each surface: `base`, `tools`, `large`, `reason`.
