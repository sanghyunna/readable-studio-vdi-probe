#!/usr/bin/env python3
"""Readable Studio VDI probe v4 - find exactly why a Databricks endpoint rejects tools.

Standard library only. Python 3.8+. Read-only: it never writes to the workspace.

The host and token are read from arguments or the environment and are NEVER
printed. Standard output is one short result code so it can be retyped by hand
out of a restricted environment.

SCRIPT FAILURE vs RESULT
    Probe bugs and real answers are never mixed.
      - a cell that could not even be attempted is `X<n>` (script/build fault)
      - a cell that reached the server is a status code (`2`, `4t`, `N`, ...)
      - if the script itself breaks, the last line starts with `FAIL#`
        instead of `V4#`, so a broken run can never be read as a finding.
    Run `python test.py --selftest` first: it exercises every classifier and
    body builder offline with no network and no credentials.

USAGE
    python test.py --selftest
    python test.py --endpoint system.ai.databricks-claude-sonnet-5 \
                   --control system.ai.gpt-oss-120b

CREDENTIALS (first match wins; you are prompted otherwise)
    --host  / DATABRICKS_HOST     https://your-workspace.cloud.databricks.com
    --token / DATABRICKS_TOKEN    personal access token (prompt is hidden)

OPTIONS
    --dump detail.json   full local transcript for your own eyes; keep it local
    --insecure           skip TLS verification (corporate proxy)
    --verbose            per-cell progress on stderr
    --only A,C           restrict to some paths
"""

import argparse
import getpass
import hashlib
import json
import os
import re
import ssl
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request

VERSION = "V4"
TIMEOUT = 90
USER_TEXT = "Reply with the single word ok."
SYSTEM_TEXT = "You are terse."
TOOL_NAME = "get_weather"
TOOL_DESC = "Look up the current weather for a city."
TOOL_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City name."}},
    "required": ["city"],
}

# --------------------------------------------------------------- classification
#
# status class
#   2  200 ok                4x 4xx bad request (letter says which parameter)
#   N  404 not found         A  401/403 auth or permission
#   R  429 rate limited      P  413 payload too large
#   5  5xx upstream          T  transport/timeout/TLS/proxy
#   ?  reached server but unclassified
#   X<n> the probe could not attempt this cell - a SCRIPT fault, not a finding
#
# 4xx detail letter
#   t tools                  c tool_choice           p parallel_tool_calls
#   r reasoning/thinking     m max-token field       s stream/stream_options
#   n tool/function message  j schema rejected       b unsupported combination
#   v generic validation     o other named parameter

# Exact parameter name -> detail letter. Order matters: the longest, most
# specific name must come before any name that is its prefix.
PARAM_LETTERS = (
    ("parallel_tool_calls", "p"),
    ("tool_choice", "c"),
    ("tool_call_id", "n"),
    ("tool_calls", "n"),
    ("tools", "t"),
    ("reasoning_effort", "r"),
    ("output_config", "r"),
    ("reasoning", "r"),
    ("thinking", "r"),
    ("max_completion_tokens", "m"),
    ("max_output_tokens", "m"),
    ("max_new_tokens", "m"),
    ("max_tokens", "m"),
    ("stream_options", "s"),
    ("stream", "s"),
    ("anthropic_version", "o"),
    ("input_schema", "j"),
    ("additionalproperties", "j"),
    ("parameters", "j"),
)

# Generic English words that also appear as JSON keys. They are trusted only
# when upstream names them explicitly, never from a loose substring scan -
# otherwise "unsupported combination of parameters" reads as a schema fault.
AMBIGUOUS = frozenset({"parameters", "stream", "reasoning", "thinking"})

# Names upstream may legitimately reject. A captured token outside this set is
# only accepted when it looks like snake_case, so prose fragments and URL
# scheme words can never be reported as a parameter.
KNOWN_PARAMS = frozenset([name for name, _ in PARAM_LETTERS] + [
    "messages", "model", "input", "system", "instructions", "temperature",
    "top_p", "top_k", "n", "stop", "seed", "user", "logprobs", "functions",
    "function_call", "response_format", "metadata", "store", "prompt",
])

SAFE_NAME = re.compile(r"^[a-z0-9_]{1,40}$")
REJECTED = re.compile(r"reject(?:ed)?\s+parameter[s]?:?\s*([a-z0-9_ ,\.]+)", re.I)
UNRECOGNIZED = re.compile(r"(?:unrecognized|unexpected|unknown|invalid|unsupported)\s+"
                          r"(?:request\s+)?(?:field|parameter|key|argument|property)[s]?:?\s*"
                          r"['\"]?([a-z0-9_]+)", re.I)


SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")


def named_parameter(text):
    """Extract a rejected parameter name from upstream prose, or None.

    A capture is accepted only when it is a parameter name we know, or is
    unambiguously snake_case. Anything else - a stray English word, a URL
    scheme, a host fragment - is discarded rather than reported.
    """
    for pattern in (REJECTED, UNRECOGNIZED):
        match = pattern.search(text or "")
        if not match:
            continue
        candidate = match.group(1).strip().replace(" ", "").split(",")[0].strip(".").lower()
        if not SAFE_NAME.match(candidate):
            continue
        if candidate in KNOWN_PARAMS or SNAKE_CASE.match(candidate):
            return candidate
    return None


def classify(status, text):
    """Map one HTTP outcome to (cell, rejected-parameter-or-None).

    Parameter identification always wins over generic phrasing, because
    'max_completion_tokens is not supported' is a max-token finding, not a
    generic 'unsupported combination'.
    """
    if status == 0:
        return "T", None
    if 200 <= status < 300:
        return "2", None
    if status == 404:
        return "N", None
    if status in (401, 403):
        return "A", None
    if status == 413:
        return "P", None
    if status == 429:
        return "R", None
    if status >= 500:
        return "5", None
    if not 400 <= status < 500:
        return "?", None

    body = text or ""
    low = body.lower()

    # 1. An explicitly named parameter is the strongest signal.
    named = named_parameter(body)
    if named:
        for key, letter in PARAM_LETTERS:
            if named == key:
                return "4" + letter, named
        return "4o", named

    # 2. A combination complaint names no single parameter, so settle it here
    #    before any substring guessing.
    if "unsupported combination" in low or "invalid combination" in low:
        return "4b", None

    # 3. Unambiguous parameter names appearing anywhere in the prose.
    for key, letter in PARAM_LETTERS:
        if key in AMBIGUOUS:
            continue
        if key in low:
            return "4" + letter, key

    # 4. Remaining generic rejections.
    if "not supported" in low or "unsupported" in low:
        return "4b", None
    return "4v", None


# ------------------------------------------------------------------- transport


def make_opener(insecure):
    context = ssl.create_default_context()
    if insecure:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))


def call(opener, host, token, path, body=None, method=None):
    """Return (status, text). HTTP errors are results, not exceptions."""
    url = host.rstrip("/") + path
    data = None
    headers = {"Authorization": "Bearer " + token, "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method=method or ("POST" if data is not None else "GET"))
    try:
        with opener.open(request, timeout=TIMEOUT) as response:
            return response.status, response.read(200000).decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        try:
            payload = error.read(200000).decode("utf-8", "replace")
        except Exception:
            payload = ""
        return error.code, payload
    except Exception as error:
        return 0, type(error).__name__


# ------------------------------------------------------------------ body shapes


def tools_openai():
    return [{"type": "function",
             "function": {"name": TOOL_NAME, "description": TOOL_DESC, "parameters": TOOL_SCHEMA}}]


def tools_flat():
    return [{"name": TOOL_NAME, "description": TOOL_DESC, "parameters": TOOL_SCHEMA}]


def tools_anthropic():
    return [{"name": TOOL_NAME, "description": TOOL_DESC, "input_schema": TOOL_SCHEMA}]


KINDS = ("chat", "chat_model", "messages", "messages_model", "responses")


def base_body(kind, endpoint):
    user = [{"role": "user", "content": USER_TEXT}]
    if kind == "chat":
        return {"messages": user, "max_tokens": 64}
    if kind == "chat_model":
        return {"model": endpoint, "messages": user, "max_tokens": 64}
    if kind == "messages":
        return {"messages": user, "max_tokens": 64}
    if kind == "messages_model":
        return {"model": endpoint, "messages": user, "max_tokens": 64}
    if kind == "responses":
        return {"model": endpoint, "input": USER_TEXT, "max_output_tokens": 64}
    raise ValueError("unknown kind: " + str(kind))


# Fixed cell order. A relayed code is decoded positionally against this list.
CELL_LABELS = (
    "min", "sys", "t_oa", "t_fl", "t_an", "t_none", "tc_a", "tc_r", "tc_x", "ptc",
    "eff", "mct", "str", "sopt", "two", "strict", "round", "avers", "sysarr", "eff_only",
)


def variants(kind, endpoint):
    """Build the fixed ablation list. Each entry changes one thing from minimal."""
    native = kind in ("messages", "messages_model")
    responses = kind == "responses"
    preferred = tools_anthropic() if native else tools_openai()

    def fresh():
        return json.loads(json.dumps(base_body(kind, endpoint)))

    def with_tools(extra=None, tools=None):
        body = fresh()
        body["tools"] = json.loads(json.dumps(preferred if tools is None else tools))
        if extra:
            body.update(json.loads(json.dumps(extra)))
        return body

    out = [("min", fresh())]

    body = fresh()
    if native:
        body["system"] = SYSTEM_TEXT
    elif responses:
        body["instructions"] = SYSTEM_TEXT
    else:
        body["messages"] = [{"role": "system", "content": SYSTEM_TEXT}] + body["messages"]
    out.append(("sys", body))

    out.append(("t_oa", with_tools(tools=tools_openai())))
    out.append(("t_fl", with_tools(tools=tools_flat())))
    out.append(("t_an", with_tools(tools=tools_anthropic())))
    out.append(("t_none", with_tools(tools=[])))

    out.append(("tc_a", with_tools({"tool_choice": {"type": "auto"} if native else "auto"})))
    out.append(("tc_r", with_tools({"tool_choice": {"type": "any"} if native else "required"})))
    out.append(("tc_x", with_tools({"tool_choice": {"type": "none"} if native else "none"})))

    if native:
        out.append(("ptc", with_tools({"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}})))
    else:
        out.append(("ptc", with_tools({"parallel_tool_calls": False})))

    if native:
        out.append(("eff", with_tools({"output_config": {"effort": "high"}})))
    elif responses:
        out.append(("eff", with_tools({"reasoning": {"effort": "high"}})))
    else:
        out.append(("eff", with_tools({"reasoning_effort": "high"})))

    body = with_tools()
    body.pop("max_tokens", None)
    body.pop("max_output_tokens", None)
    body["max_completion_tokens"] = 64
    out.append(("mct", body))

    out.append(("str", with_tools({"stream": True})))
    out.append(("sopt", with_tools({"stream": True, "stream_options": {"include_usage": True}})))

    second = json.loads(json.dumps(preferred[0]))
    if native:
        second["name"] = "get_time"
    else:
        second["function"]["name"] = "get_time"
    out.append(("two", with_tools(tools=preferred + [second])))

    strict_schema = json.loads(json.dumps(TOOL_SCHEMA))
    strict_schema["additionalProperties"] = False
    if native:
        strict_tools = [{"name": TOOL_NAME, "description": TOOL_DESC, "input_schema": strict_schema}]
    else:
        strict_tools = [{"type": "function", "function": {"name": TOOL_NAME, "description": TOOL_DESC,
                                                          "strict": True, "parameters": strict_schema}}]
    out.append(("strict", with_tools(tools=strict_tools)))

    # Tool-result round trip: the turn that follows a tool call.
    body = with_tools()
    if native:
        body["messages"] = [
            {"role": "user", "content": "Weather in Seoul?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_probe1", "name": TOOL_NAME, "input": {"city": "Seoul"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_probe1", "content": "18C clear"}]},
        ]
    elif responses:
        body["input"] = [
            {"role": "user", "content": "Weather in Seoul?"},
            {"type": "function_call", "call_id": "call_probe1", "name": TOOL_NAME,
             "arguments": "{\"city\":\"Seoul\"}"},
            {"type": "function_call_output", "call_id": "call_probe1", "output": "18C clear"},
        ]
    else:
        body["messages"] = [
            {"role": "user", "content": "Weather in Seoul?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_probe1", "type": "function",
                 "function": {"name": TOOL_NAME, "arguments": "{\"city\":\"Seoul\"}"}}]},
            {"role": "tool", "tool_call_id": "call_probe1", "content": "18C clear"},
        ]
    out.append(("round", body))

    body = with_tools()
    body["anthropic_version"] = "bedrock-2023-05-31"
    out.append(("avers", body))

    body = with_tools()
    if native:
        body["system"] = [{"type": "text", "text": SYSTEM_TEXT}]
    elif responses:
        body["instructions"] = SYSTEM_TEXT
    else:
        body["messages"] = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]}] + body["messages"]
    out.append(("sysarr", body))

    # Reasoning effort WITHOUT tools, to separate an effort fault from a tools fault.
    body = fresh()
    if native:
        body["output_config"] = {"effort": "high"}
    elif responses:
        body["reasoning"] = {"effort": "high"}
    else:
        body["reasoning_effort"] = "high"
    out.append(("eff_only", body))

    labels = tuple(label for label, _ in out)
    if labels != CELL_LABELS:
        raise AssertionError("cell order drift: {} != {}".format(labels, CELL_LABELS))
    return out


PATHS = (
    ("A", "/serving-endpoints/{name}/invocations", "chat"),
    ("B", "/serving-endpoints/{name}/invocations", "messages"),
    ("C", "/ai-gateway/mlflow/v1/chat/completions", "chat_model"),
    ("D", "/ai-gateway/openai/v1/chat/completions", "chat_model"),
    ("E", "/ai-gateway/anthropic/v1/messages", "messages_model"),
    ("F", "/ai-gateway/mlflow/v1/responses", "responses"),
    ("G", "/ai-gateway/openai/v1/responses", "responses"),
    ("H", "/serving-endpoints/chat/completions", "chat_model"),
)


# -------------------------------------------------------------------- metadata


def metadata(opener, host, token, endpoint, dump):
    """Return (4-char code, supported-api list). Never raises."""
    status, text = call(opener, host, token,
                        "/api/2.0/serving-endpoints/" + urllib.parse.quote(endpoint))
    record(dump, "meta", endpoint, status, text)
    if status != 200:
        letter = "A" if status in (401, 403) else "N" if status == 404 else "T" if status == 0 else "?"
        return "x" + letter + "--", []
    try:
        data = json.loads(text)
    except Exception:
        return "xJ--", []

    etype = {"FOUNDATION_MODEL_API": "f", "EXTERNAL_MODEL": "x",
             "PROVISIONED_THROUGHPUT": "p", "CUSTOM_MODEL": "c",
             "FEATURE_STORE": "s"}.get(str(data.get("endpoint_type") or "").upper(), "?")
    task = str(data.get("task") or "")
    task_letter = ("c" if task == "llm/v1/chat"
                   else "m" if task.endswith("completions")
                   else "e" if "embed" in task
                   else "-" if not task else "o")

    entities = ((data.get("config") or {}).get("served_entities")
                or (data.get("pending_config") or {}).get("served_entities") or [])
    entity_letter, supported = "-", []
    if entities:
        first = entities[0] or {}
        if first.get("external_model"):
            entity_letter = "x"
        elif first.get("foundation_model"):
            entity_letter = "f"
        elif first.get("entity_name"):
            entity_letter = "c"
        else:
            entity_letter = "?"
        for key in ("supported_apis", "openai_api_compatible", "supported_api_formats"):
            value = first.get(key) or data.get(key)
            if isinstance(value, list):
                supported = [str(v) for v in value]
                break
    gateway = "g" if data.get("ai_gateway") else "-"
    return etype + task_letter + entity_letter + gateway, supported


def record(dump, phase, key, status, text):
    if dump is None:
        return
    dump.append({"phase": phase, "key": key, "status": status, "body": (text or "")[:6000]})


# ----------------------------------------------------------------------- probe


def probe_endpoint(opener, host, token, endpoint, dump, verbose, only):
    meta, supported = metadata(opener, host, token, endpoint, dump)
    rejected, digests, path_codes = [], [], []

    for letter, template, kind in PATHS:
        if only and letter not in only:
            continue
        path = template.replace("{name}", urllib.parse.quote(endpoint))
        try:
            plan = variants(kind, endpoint)
        except Exception:
            if verbose:
                traceback.print_exc()
            path_codes.append(letter + "=" + ".".join(["X1"] * len(CELL_LABELS)))
            continue

        cells = []
        for index, (label, body) in enumerate(plan, start=1):
            try:
                status, text = call(opener, host, token, path, body)
            except Exception:
                if verbose:
                    traceback.print_exc()
                cells.append("X2")
                continue
            record(dump, letter + "{:02d}".format(index) + "-" + label, path, status, text)
            try:
                cell, named = classify(status, text)
            except Exception:
                if verbose:
                    traceback.print_exc()
                cells.append("X3")
                continue
            cells.append(cell)
            if named and named not in rejected:
                rejected.append(named)
            if cell.startswith("4") or cell in ("?", "5"):
                digest = hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()[:4]
                if digest not in digests:
                    digests.append(digest)
            if verbose:
                sys.stderr.write("  {}{:02d} {:<8} -> {}\n".format(letter, index, label, cell))
            if index == 1 and cell in ("N", "A", "T"):
                cells.extend(["-"] * (len(CELL_LABELS) - 1))
                break
        path_codes.append(letter + "=" + ".".join(cells))

    parts = ["M=" + meta]
    if supported:
        joined = " ".join(supported).lower()
        parts.append("S=" + "".join([
            "a" if "anthropic" in joined else "-",
            "o" if ("openai" in joined and "responses" not in joined) else "-",
            "m" if ("mlflow" in joined and "responses" not in joined) else "-",
            "r" if "responses" in joined else "-",
        ]))
    parts.extend(path_codes)
    if rejected:
        parts.append("R=" + ",".join(rejected[:4]))
    if digests:
        parts.append("H=" + ",".join(digests[:5]))
    return "/".join(parts)


LEGEND = """
--- legend (no need to relay this) -------------------------------------------
Cells, in order: 01 min 02 sys 03 tools-openai 04 tools-flat 05 tools-anthropic
  06 tools-empty 07 tool_choice-auto 08 tool_choice-forced 09 tool_choice-none
  10 parallel-off 11 reasoning+tools 12 max_completion_tokens 13 stream+tools
  14 stream_options 15 two-tools 16 strict-schema 17 tool-result-roundtrip
  18 anthropic_version 19 system-as-blocks 20 reasoning-without-tools
Paths: A invocations(chat) B invocations(anthropic) C mlflow-chat D openai-chat
       E anthropic-messages F mlflow-responses G openai-responses H legacy-chat
Status: 2 ok | 4x bad-request | N not-found | A auth | R rate | P too-large
        5 upstream | T transport | ? unclassified | - skipped | X# SCRIPT FAULT
4x letter: t tools c tool_choice p parallel r reasoning m max-tokens s stream
           n tool-message j schema b unsupported-combination v generic o other
M=<type><task><entity><gateway>  S=<anthropic><openai><mlflow><responses>
R=rejected parameter names   H=distinct error fingerprints
Only a line starting with V4# is a result. FAIL# means the probe itself broke.
------------------------------------------------------------------------------
"""


# -------------------------------------------------------------------- selftest


def selftest():
    """Offline checks: no network, no credentials. Prints PASS/FAIL per check."""
    failures = []

    def check(name, condition, detail=""):
        print("{:<46} {}".format(name, "PASS" if condition else "FAIL " + detail))
        if not condition:
            failures.append(name)

    check("classify 200", classify(200, "")[0] == "2")
    check("classify 204", classify(204, "")[0] == "2")
    check("classify 404", classify(404, "")[0] == "N")
    check("classify 401", classify(401, "")[0] == "A")
    check("classify 403", classify(403, "")[0] == "A")
    check("classify 429", classify(429, "")[0] == "R")
    check("classify 413", classify(413, "")[0] == "P")
    check("classify 503", classify(503, "")[0] == "5")
    check("classify transport", classify(0, "TimeoutError")[0] == "T")
    check("classify 418 unclassified-4xx", classify(418, "teapot")[0] == "4v")

    cases = (
        ('{"message":"Rejected parameter: tools"}', "4t", "tools"),
        ('{"message":"Rejected parameter: tool_choice, tools"}', "4c", "tool_choice"),
        ("Rejected parameters: parallel_tool_calls", "4p", "parallel_tool_calls"),
        ("max_completion_tokens is not supported", "4m", None),
        ("Unsupported combination of parameters", "4b", None),
        ("Unrecognized request field: reasoning_effort", "4r", "reasoning_effort"),
        ('Invalid parameter: "anthropic_version"', "4o", "anthropic_version"),
        ("input_schema must be an object", "4j", None),
        ("tool_call_id missing for role tool", "4n", None),
        ("stream_options not allowed", "4s", None),
        ("something went sideways", "4v", None),
    )
    for text, expect_cell, expect_named in cases:
        cell, named = classify(400, text)
        ok = cell == expect_cell and (expect_named is None or named == expect_named)
        check("classify 400: " + text[:34], ok, "got {} {}".format(cell, named))

    leaks = (
        "Rejected parameter: https://secret.host",
        "Rejected parameter: dapi1234567890abcdef",
        "Unrecognized request field: workspace",
    )
    for text in leaks:
        got = named_parameter(text)
        check("no prose/secret leak: " + text[:30], got is None, "got " + repr(got))
    check("named accepts known bare word", named_parameter("Rejected parameter: tools") == "tools")
    check("named accepts snake_case", named_parameter("Rejected parameter: some_new_field") == "some_new_field")

    for kind in KINDS:
        try:
            plan = variants(kind, "probe.endpoint")
            labels = tuple(label for label, _ in plan)
            serializable = all(json.dumps(body) for _, body in plan)
            check("variants " + kind, labels == CELL_LABELS and serializable)
        except Exception as error:
            check("variants " + kind, False, repr(error))

    try:
        plan = dict(variants("chat_model", "e"))
        checks = (
            ("min has no tools", "tools" not in plan["min"]),
            ("t_oa openai envelope", plan["t_oa"]["tools"][0].get("type") == "function"),
            ("t_fl flat envelope", "name" in plan["t_fl"]["tools"][0]),
            ("t_an input_schema", "input_schema" in plan["t_an"]["tools"][0]),
            ("t_none empty list", plan["t_none"]["tools"] == []),
            ("mct drops max_tokens", "max_tokens" not in plan["mct"] and "max_completion_tokens" in plan["mct"]),
            ("two has 2 tools", len(plan["two"]["tools"]) == 2),
            ("two distinct names", plan["two"]["tools"][0]["function"]["name"]
             != plan["two"]["tools"][1]["function"]["name"]),
            ("round has tool message", any(m.get("role") == "tool" for m in plan["round"]["messages"])),
            ("eff_only has no tools", "tools" not in plan["eff_only"]),
            ("str sets stream", plan["str"].get("stream") is True),
        )
        for name, condition in checks:
            check("chat " + name, condition)
    except Exception as error:
        check("chat body shape checks", False, repr(error))

    try:
        native = dict(variants("messages", "e"))
        check("native sys is top-level", native["sys"].get("system") == SYSTEM_TEXT)
        check("native ptc disables parallel",
              native["ptc"]["tool_choice"].get("disable_parallel_tool_use") is True)
        check("native round uses tool_result",
              native["round"]["messages"][-1]["content"][0]["type"] == "tool_result")
        check("native eff uses output_config", "output_config" in native["eff"])
    except Exception as error:
        check("native body shape checks", False, repr(error))

    try:
        resp = dict(variants("responses", "e"))
        check("responses uses input", "input" in resp["min"])
        check("responses round uses function_call_output",
              any(isinstance(i, dict) and i.get("type") == "function_call_output" for i in resp["round"]["input"]))
    except Exception as error:
        check("responses body shape checks", False, repr(error))

    check("paths unique letters", len({p[0] for p in PATHS}) == len(PATHS))
    check("paths use known kinds", all(p[2] in KINDS for p in PATHS))
    check("cell label count is 20", len(CELL_LABELS) == 20)
    check("cell labels unique", len(set(CELL_LABELS)) == len(CELL_LABELS))

    print("")
    if failures:
        print("SELFTEST FAILED: {} check(s): {}".format(len(failures), ", ".join(failures)))
        return 1
    print("SELFTEST PASSED: probe logic is sound; safe to run against the workspace.")
    return 0


# ------------------------------------------------------------------------ main


def run(args):
    host = args.host.strip() or input("workspace host (https://...): ").strip()
    if not host:
        raise SystemExit("no host supplied")
    if not host.startswith("http"):
        host = "https://" + host
    token = args.token.strip() or getpass.getpass("token (hidden): ").strip()
    if not token:
        raise SystemExit("no token supplied")

    only = {p.strip().upper() for p in args.only.split(",") if p.strip()} if args.only else None
    opener = make_opener(args.insecure)
    dump = [] if args.dump else None

    codes = []
    for label, endpoint in (("P", args.endpoint), ("Q", args.control)):
        if not endpoint:
            continue
        if args.verbose:
            sys.stderr.write("probing {} ...\n".format(label))
        codes.append(label + "|" + probe_endpoint(opener, host, token, endpoint, dump, args.verbose, only))

    if dump is not None:
        with open(args.dump, "w", encoding="utf-8") as handle:
            json.dump(dump, handle, indent=2)
        sys.stderr.write("local transcript: {} (keep it on this machine)\n".format(args.dump))

    sys.stderr.write(LEGEND)
    return VERSION + "#" + "#".join(codes)


def main():
    parser = argparse.ArgumentParser(description="Databricks tools-rejection probe (v4)")
    parser.add_argument("--endpoint", default="system.ai.databricks-claude-sonnet-5")
    parser.add_argument("--control", default="", help="endpoint known to work, e.g. system.ai.gpt-oss-120b")
    parser.add_argument("--host", default=os.environ.get("DATABRICKS_HOST", ""))
    parser.add_argument("--token", default=os.environ.get("DATABRICKS_TOKEN", ""))
    parser.add_argument("--dump", default="", help="write a full LOCAL transcript to this file")
    parser.add_argument("--insecure", action="store_true", help="skip TLS verification")
    parser.add_argument("--verbose", action="store_true", help="per-cell progress on stderr")
    parser.add_argument("--only", default="", help="restrict to paths, e.g. A,C,E")
    parser.add_argument("--selftest", action="store_true", help="offline logic check, no network")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    try:
        print(run(args))
        return 0
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("FAIL#probe-error")
        return 3


if __name__ == "__main__":
    sys.exit(main())
