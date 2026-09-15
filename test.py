#!/usr/bin/env python3
"""Self-contained Databricks AI Gateway probe.

Runs from the corporate VDI using only the Databricks CLI and the Python
standard library. It learns which surface accepts the model's request shape
and which parameter the gateway rejects, then emits a short hand-retypable
code.
"""

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

PROFILE = os.environ.get("DATABRICKS_PROFILE", "")
MODEL = "system.ai.databricks-claude-sonnet-5"


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def run(args):
    cmd = ["databricks"] + args
    if PROFILE:
        cmd += ["--profile", PROFILE]
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=30)
    except subprocess.CalledProcessError as e:
        fail(f"databricks CLI failed ({e.returncode}): {e.output.strip()}")
    except FileNotFoundError:
        fail("databricks CLI not found; ensure it is on PATH")


def get_token():
    out = run(["auth", "token"])
    try:
        data = json.loads(out)
        token = data.get("access_token") or data.get("token")
        if not token:
            raise ValueError("no access_token in response")
        return token
    except json.JSONDecodeError as e:
        fail(f"could not parse databricks auth token output: {e}")


def get_host():
    # Parse the CLI config file the same way the CLI would.
    cfg_path = os.path.join(os.path.expanduser("~"), ".databrickscfg")
    if not os.path.exists(cfg_path):
        fail("~/.databrickscfg not found")

    target = PROFILE
    if not target:
        with open(cfg_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^default_profile\s*=\s*(\S+)", line)
                if m:
                    target = m.group(1)
                    break
    if not target:
        target = "DEFAULT"

    in_section = False
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            sec = re.match(r"^\[([^\]]+)\]", line)
            if sec:
                in_section = (sec.group(1) == target)
                continue
            if in_section:
                m = re.match(r"^host\s*=\s*(\S+)", line)
                if m:
                    return m.group(1).rstrip("/")

    fail(f"host not found for profile '{target}' in ~/.databrickscfg")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": f"func_{i}",
            "description": f"Function {i}",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        },
    }
    for i in range(4)
]


def request_json(url, token, body=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, None
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        param = None
        for candidate in ["tools", "max_tokens", "reasoning_effort"]:
            if candidate in text or candidate.replace("_", " ") in text:
                param = candidate
                break
        if not param:
            m = re.search(r"rejected parameter:\s*([^\s\"']+)", text, re.I)
            if m:
                param = m.group(1)
        return e.code, param
    except urllib.error.URLError as e:
        return -1, str(e.reason)
    except Exception as e:
        return -1, str(e)


def probe(url, token, body):
    return request_json(url, token, body)


def code(status, param):
    if status == -1:
        return "0N"
    if 200 <= status < 300:
        return "2."
    if param == "tools":
        return f"{status // 100}T"
    if param == "max_tokens":
        return f"{status // 100}M"
    if param == "reasoning_effort":
        return f"{status // 100}R"
    if param:
        return f"{status // 100}X"
    return f"{status // 100}."


def main():
    token = get_token()
    host = get_host()

    surfaces = {
        "A": f"{host}/ai-gateway/anthropic/v1/messages",
        "C": f"{host}/ai-gateway/openai/v1/chat/completions",
        "R": f"{host}/ai-gateway/openai/v1/responses",
    }

    results = {}
    for key, url in surfaces.items():
        base_body = {
            "model": MODEL,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        }
        if key == "R":
            base_body = {
                "model": MODEL,
                "input": "hi",
                "max_tokens": 10,
            }

        variants = [
            ("base", base_body),
            ("tools", {**base_body, "tools": TOOLS}),
            ("large", {**base_body, "max_tokens": 8192}),
            ("reason", {**base_body, "reasoning_effort": "low"}),
        ]

        group = []
        for name, body in variants:
            status, param = probe(url, token, body)
            group.append((name, status, param))
        results[key] = group

    # Build compact code: one 2-char token per probe, groups separated by '/'.
    coded = "/".join(
        f"{key}:" + "".join(code(s, p) for _, s, p in group)
        for key, group in results.items()
    )

    print()
    print("=" * 60)
    print("DATABRICKS AI GATEWAY PROBE RESULTS")
    print("=" * 60)
    print()
    for key, group in results.items():
        print(f"Surface {key} ({surfaces[key]})")
        for name, status, param in group:
            detail = f"param={param}" if param else "no param named"
            print(f"  {name:6s}: HTTP {status:4d}  {detail}")
        print()

    print("-" * 60)
    print("CODED RESULT (copy this short string back exactly):")
    print(coded)
    print("-" * 60)

    # Sanity-check against a known-good read-only API on this workspace.
    ref_url = f"{host}/api/2.0/serving-endpoints"
    ref_status, ref_param = request_json(ref_url, token)
    print()
    print(f"SANITY CHECK (known-good API): {ref_url}")
    print(f"  status={ref_status}, code={code(ref_status, ref_param)}")
    print()


if __name__ == "__main__":
    main()
