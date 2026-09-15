"""Deterministic unit and real-local-HTTP integration tests; no credentials needed."""
import contextlib
import io
import json
import subprocess
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import test as probe


class ProbeTests(unittest.TestCase):
    def test_discovery_matches_entities_not_endpoint_names(self):
        data = {"endpoints": [
            {"name": "custom-claude", "config": {"served_entities": [{"entity_name": probe.MODELS[0]}]}},
            {"name": "z-gpt", "config": {"served_models": [{"model_name": probe.MODELS[1]}]}},
            {"name": "databricks-gpt-oss-120b", "config": {"served_entities": [{"entity_name": probe.MODELS[1]}]}},
            {"name": "databricks-claude-sonnet-5", "config": {"served_entities": [{"entity_name": "other"}]}},
        ]}
        self.assertEqual(probe.select_endpoints(data), ["custom-claude", "databricks-gpt-oss-120b"])
        self.assertEqual(probe.select_endpoints({"endpoints": []}), [None, None])

    def test_one_variable_variants(self):
        bodies = probe.bodies()
        self.assertEqual(set(bodies), set(probe.VARIANTS))
        for prefix in ("O", "A"):
            parent = bodies[prefix]
            self.assertEqual({k: v for k, v in parent.items() if k != "tools"}, bodies["B"])
            self.assertEqual(len(parent["tools"]), 1)
            for suffix, field in (("C", "tool_choice"), ("P", "parallel_tool_calls"), ("T", "strict")):
                child = bodies[prefix + suffix]
                self.assertEqual({k: v for k, v in child.items() if k != field}, parent)
            strict = bodies[prefix + "S"]
            target = strict["tools"][0]["function"] if prefix == "O" else strict["tools"][0]
            self.assertIs(target.pop("strict"), True)
            self.assertEqual(strict, parent)
        self.assertEqual(bodies["O"]["tools"][0]["function"]["parameters"],
                         bodies["A"]["tools"][0]["input_schema"])

    def test_rejection_priority_and_nested_paths(self):
        cases = [
            ('tools supplied; Rejected parameter: tool_choice', {}, 'tool_choice', '4C'),
            ('Rejected parameter: tools[0].function.strict', {}, 'tools[0].function.strict', '4s'),
            ('Bad request: json: unknown field "name"', {}, 'name', '4N'),
            ('tool validation failed', {'error': {'param': 'tools.0.input_schema'}}, 'tools.0.input_schema', '4i'),
            ('tools appear somewhere in this message', {}, None, '4.'),
        ]
        for message, data, expected, coded in cases:
            with self.subTest(message=message):
                param = probe.rejection(message, data)
                self.assertEqual(param, expected)
                self.assertEqual(probe.Result(400, param).code(), coded)
        for status, code in ((200, '2.'), (404, 'N.'), (403, '4.'), (500, '5.'), (-1, '0N'), (-2, '--')):
            self.assertEqual(probe.Result(status).code(), code)

    def test_missing_endpoint_makes_no_request(self):
        with patch.object(probe, 'request_json', side_effect=AssertionError('unexpected request')):
            results = probe.probe_endpoint('https://unused', 'unused', None)
        self.assertEqual([r.code() for r in results.values()], ['--'] * 11)

    def test_cli_failures_do_not_echo_output(self):
        failed = subprocess.CompletedProcess([], 1, stdout='private fixture', stderr='private fixture')
        with patch.object(probe.subprocess, 'run', return_value=failed):
            with self.assertRaises(probe.ProbeError) as error:
                probe.cli_json(['auth', 'token'], 'profile')
        self.assertNotIn('private fixture', str(error.exception))
        with patch.object(probe.subprocess, 'run', side_effect=subprocess.TimeoutExpired([], 30)):
            with self.assertRaises(probe.ProbeError):
                probe.cli_json(['auth', 'token'], 'profile')

    def test_complete_main_through_http_with_asymmetric_discovery(self):
        received = []
        credential = 'local-test-credential'

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # The test server has no console logging.

            def respond(self, status, payload):
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def do_GET(self):
                received.append((self.command, self.path, self.headers.get('Authorization'), None))
                self.respond(200, {'endpoints': [{'name': 'custom/gpt', 'config': {
                    'served_entities': [{'entity_name': probe.MODELS[1]}]}}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append((self.command, self.path, self.headers.get('Authorization'), body))
                tools = body.get('tools', [])
                field = ('name' if tools and 'name' in tools[0] else
                         'parallel_tool_calls' if 'parallel_tool_calls' in body else
                         'strict' if 'strict' in body else None)
                if field:
                    self.respond(400, {'error': {'message': f'Rejected parameter: {field}; {credential}', 'param': field}})
                else:
                    self.respond(200, {'choices': []})

        with HTTPServer(('127.0.0.1', 0), Handler) as server:
            # Bind/listen has already completed. Each request is the exact progress
            # signal; there are no sleeps or timing-based assertions.
            def serve():
                for _ in range(12):
                    server.handle_request()
            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            output = io.StringIO()
            host = f'http://127.0.0.1:{server.server_port}'
            with patch.object(probe, 'credentials', return_value=(host, credential)), \
                    patch('sys.argv', ['test.py']), contextlib.redirect_stdout(output):
                probe.main()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertNotIn(credential, output.getvalue())
        code = next(line for line in output.getvalue().splitlines() if line.startswith('V2:'))
        self.assertEqual(code, 'V2:2.:--2./--2./--4N/--2./--4P/--2./--4N/--4N/--4N/--4S/--4N')
        self.assertEqual(received[0][:2], ('GET', '/api/2.0/serving-endpoints'))
        self.assertEqual(len(received), 12)
        for index, request in enumerate(received[1:]):
            self.assertEqual(request[:3], ('POST', '/serving-endpoints/custom%2Fgpt/invocations', 'Bearer ' + credential))
            self.assertEqual(request[3], probe.bodies()[probe.VARIANTS[index]])


if __name__ == '__main__':
    unittest.main()
