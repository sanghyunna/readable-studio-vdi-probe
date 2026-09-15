#!/usr/bin/env python3
"""Read-only endpoint discovery plus tiny Claude/GPT tool-envelope A/B requests.

Python standard library only. Credentials stay in memory and are never printed.
No tools are executed. Only the final fixed-position code needs to leave the VDI.
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

MODELS = ("system.ai.databricks-claude-sonnet-5", "system.ai.gpt-oss-120b")
VARIANTS = ("B", "O", "A", "OC", "OP", "OS", "AC", "AP", "AS", "OT", "AT")
FIELDS = {
    "tools": "T", "function": "F", "name": "N", "parameters": "J",
    "input_schema": "I", "tool_choice": "C", "parallel_tool_calls": "P",
    "strict": "S", "max_tokens": "M", "messages": "G", "model": "D",
}


class ProbeError(Exception):
    pass


def cli_json(args, profile):
    command = ["databricks"] + args
    if profile:
        command += ["--profile", profile]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise ProbeError("databricks CLI not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise ProbeError("databricks CLI timed out (30 seconds)") from None
    if result.returncode:
        # CLI output may contain credentials: never echo it, even on failure.
        raise ProbeError(f"databricks {' '.join(args[:2])} failed (exit {result.returncode})")
    try:
        return json.loads(result.stdout)
    except ValueError:
        raise ProbeError("databricks CLI returned invalid JSON") from None


def credentials(profile):
    auth = cli_json(["auth", "describe", "--output", "json"], profile)
    host = auth.get("details", {}).get("host") or auth.get("host")
    if not isinstance(host, str) or urllib.parse.urlsplit(host).scheme != "https":
        raise ProbeError("CLI auth describe did not return an HTTPS workspace host")
    data = cli_json(["auth", "token"], profile)
    token = data.get("access_token") or data.get("token")
    if not isinstance(token, str) or not token:
        raise ProbeError("CLI auth token returned no access token; run databricks auth login")
    return host.rstrip("/"), token


def rejection(text, data):
    """Prefer an explicitly rejected path, never a mere mention of 'tools'."""
    match = re.search(r"rejected parameter\s*:\s*[\"'`]?([\w.\[\]/-]+)", text, re.I)
    if match:
        return match.group(1).rstrip(".")
    error = data.get("error", {}) if isinstance(data, dict) else {}
    if isinstance(error, dict) and isinstance(error.get("param"), str):
        return error["param"]
    # Common validation forms. Preserve the path for the local table.
    patterns = (
        r"(?:unknown|unrecognized|unsupported|invalid|unexpected) (?:parameter|field|argument)\s*:?\s*[\"'`]?([\w.\[\]/-]+)",
        r"[\"'`]([\w.\[\]/-]+)[\"'`]\s*(?:is|are)\s*(?:not supported|not permitted|not allowed|invalid)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).rstrip(".")
    return None


@dataclass
class Result:
    status: int
    param: str = None
    detail: str = ""
    data: object = None

    def code(self):
        if self.status == -2:
            return "--"  # no endpoint; never substitute a different surface
        if self.status == -1:
            return "0N"
        if 200 <= self.status < 300:
            return "2."
        # Distinguish missing route from all other 4xx failures.
        bucket = "N" if self.status == 404 else str(self.status // 100)
        if not self.param:
            return bucket + "."
        parts = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", self.param)
        field = FIELDS.get(parts[-1].lower(), "X") if parts else "X"
        # Lowercase = explicitly nested; uppercase = bare field (scope unknown).
        if len(parts) > 1 or "[" in self.param or "/" in self.param:
            field = field.lower()
        return bucket + field


def request_json(url, token, body=None):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=payload, headers=headers,
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            status, raw = response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        # Do not print raw transport errors; they can contain request information.
        return Result(-1, detail=f"transport failure ({type(error).__name__})")
    # Redact before parsing so even a credential echoed in error.param is safe.
    raw = raw.replace(token, "[REDACTED]")
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if 200 <= status < 300:
        return Result(status, detail="accepted (does not prove tool execution)", data=data)
    # Show the gateway's actual message locally, without exposing the credential.
    error_data = data.get("error", data) if isinstance(data, dict) else None
    message = error_data.get("message", raw) if isinstance(error_data, dict) else raw
    message = str(message).replace(token, "[REDACTED]")
    return Result(status, rejection(message, data), " ".join(message.split()), data)


def select_endpoints(data):
    if not isinstance(data, dict) or not isinstance(data.get("endpoints"), list):
        raise ProbeError("endpoint listing returned an unexpected JSON structure")
    selected = []
    for model in MODELS:
        matches = []
        for endpoint in data["endpoints"]:
            config = endpoint.get("config", {})
            entities = config.get("served_entities", []) + config.get("served_models", [])
            if any((entity.get("entity_name") or entity.get("model_name")) == model
                   for entity in entities):
                matches.append(endpoint)
        # Prefer the canonical endpoint when duplicates serve the same entity.
        canonical = model.removeprefix("system.ai.")
        if not canonical.startswith("databricks-"):
            canonical = "databricks-" + canonical
        matches.sort(key=lambda endpoint: (endpoint["name"] != canonical, endpoint["name"]))
        selected.append(matches[0]["name"] if matches else None)
    return selected


def bodies():
    base = {"messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 32}
    schema = {"type": "object", "properties": {"x": {"type": "string"}},
              "required": ["x"], "additionalProperties": False}
    function = {"name": "echo", "description": "Return x.", "parameters": schema}
    openai = {**base, "tools": [{"type": "function", "function": function}]}
    anthropic = {**base, "tools": [{"name": "echo", "description": "Return x.",
                                  "input_schema": schema}]}
    requests = {"B": base, "O": openai, "A": anthropic}
    for prefix, tool_body, choice in (("O", openai, "auto"), ("A", anthropic, {"type": "auto"})):
        requests[prefix + "C"] = {**tool_body, "tool_choice": choice}
        requests[prefix + "P"] = {**tool_body, "parallel_tool_calls": False}
        strict = copy.deepcopy(tool_body)
        target = strict["tools"][0]["function"] if prefix == "O" else strict["tools"][0]
        target["strict"] = True
        requests[prefix + "S"] = strict
        requests[prefix + "T"] = {**tool_body, "strict": True}
    return requests


def probe_endpoint(host, token, endpoint):
    if endpoint is None:
        return {key: Result(-2, detail="no matching served entity") for key in VARIANTS}
    url = host + "/serving-endpoints/" + urllib.parse.quote(endpoint, safe="") + "/invocations"
    requests = bodies()
    return {key: request_json(url, token, requests[key]) for key in VARIANTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.environ.get("DATABRICKS_PROFILE", ""))
    args = parser.parse_args()
    host, token = credentials(args.profile)
    discovery = request_json(host + "/api/2.0/serving-endpoints", token)
    if 200 <= discovery.status < 300:
        endpoints = select_endpoints(discovery.data)
    else:
        endpoints = [None, None]
    print("Claude / GPT-OSS-120B on /serving-endpoints/{name}/invocations", flush=True)
    print(f"Discovery: HTTP {discovery.status}, code {discovery.code()}", flush=True)
    if not 200 <= discovery.status < 300:
        print(f"  {discovery.detail}", flush=True)
    for label, endpoint in zip(("Claude", "GPT"), endpoints):
        print(f"{label}: {endpoint or 'NO MATCH (skipped; no UC/ai-gateway fallback)'}", flush=True)
    print("Running 11 tiny requests per matched endpoint; no tool is executed.", flush=True)
    # Independent model tracks, sequential variants per endpoint, at most two workers.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(probe_endpoint, host, token, name) for name in endpoints]
        claude, gpt = [future.result() for future in futures]
    print("\nProbe  Claude HTTP/code  GPT HTTP/code  A/B")
    for key in VARIANTS:
        left, right = claude[key], gpt[key]
        comparable = left.status != -2 and right.status != -2
        comparison = ("DIFF" if left.code() != right.code() else "same code") if comparable else "unpaired"
        print(f"{key:5s}  {left.status:4d}/{left.code():2s}          {right.status:4d}/{right.code():2s}       {comparison}")
        for label, result in (("Claude", left), ("GPT", right)):
            if result.status != -2 and not 200 <= result.status < 300:
                print(f"  {label}: field={result.param or '(not named)'}; {result.detail}")
    coded = "V2:" + discovery.code() + ":" + "/".join(
        claude[key].code() + gpt[key].code() for key in VARIANTS)
    print("\nCODED RESULT (only this line needs to leave the VDI):")
    print(coded)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        sys.exit(1)
    except ProbeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        # A final boundary keeps unexpected CLI/network data from producing a traceback
        # or leaking credentials. Nonzero exit makes the failure explicit.
        print(f"ERROR: probe failed ({type(error).__name__}); no complete result", file=sys.stderr)
        sys.exit(1)
