"""Unknown-tool feedback: real parser and HTTP service, scripted engine, no GPU."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.request
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import frontend, server

TOOLS = [{"name": "shell", "parameters": {"type": "object", "properties": {
    "command": {"type": "string"}}, "required": ["command"]}}]
API_TOOLS = [{"type": "function", "function": tool} for tool in TOOLS]
ROOT = Path(__file__).resolve().parents[1]


def body(name="bash", command="pwd"):
    return (f"<tool_call>\n<function={name}>\n<parameter=command>\n{command}\n"
            "</parameter>\n</function>\n</tool_call>")


def parse(raw, policy="text", width=0, recover=False, stream=True,
          tools=TOOLS, thinking=False, declared_only=True, reason="stop"):
    parser = frontend.OutputParser(thinking=thinking, tools=tools, stream_tools=stream,
                                   recover=recover, declared_only=declared_only,
                                   undeclared_tool_policy=policy)
    events = []
    with mock.patch.object(frontend.LOGGER, "warning") as warning:
        chunks = [raw] if not width else [raw[i:i + width] for i in range(0, len(raw), width)]
        for chunk in chunks:
            events.extend(parser.feed(chunk))
        events.extend(parser.finish(reason))
    return events, parser, warning


def calls(events):
    return [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"]


def text(events):
    return "".join(e.text or "" for e in events if e.kind == "content")


class ParserPolicy(unittest.TestCase):
    def test_default_keeps_unknown_call_as_text(self):
        for width in (0, 1, 3, 9):
            for recover in (False, True):
                with self.subTest(width=width, recover=recover):
                    events, p, warning = parse(body(), width=width, recover=recover)
                    self.assertEqual(calls(events), [])
                    self.assertEqual(text(events), body())
                    self.assertFalse([e for e in events if e.kind.startswith("tool_")])
                    self.assertEqual((p.undeclared_as_text, p.undeclared_passthrough), (1, 0))
                    self.assertEqual((p.unread, p.unread_names), (1, ["bash"]))
                    self.assertEqual(warning.call_count, 1)

    def test_passthrough_streaming_recovery_matrix(self):
        for width in (0, 1, 2, 7, 31):
            for recover in (False, True):
                for stream in (False, True):
                    with self.subTest(width=width, recover=recover, stream=stream):
                        events, p, warning = parse(body(), "passthrough", width, recover, stream)
                        self.assertEqual(calls(events), [("bash", {"command": "pwd"})])
                        self.assertEqual(text(events), "")
                        self.assertEqual((p.undeclared_as_text, p.undeclared_passthrough), (0, 1))
                        self.assertEqual((p.unread, p.unread_names), (0, []))
                        self.assertEqual(warning.call_count, 1)
                        self.assertNotIn("bash", p.schemas)
                        self.assertFalse(p._callable("bash"))
                        if stream:
                            starts = [e for e in events if e.kind == "tool_start"]
                            self.assertEqual(len(starts), 1)
                            final = next(e.call for e in events if e.kind == "tool_call")
                            self.assertEqual(starts[0].call.id, final.id)
                            args = "".join(e.text for e in events if e.kind == "tool_args")
                            self.assertEqual(json.loads(args), {"command": "pwd"})

    def test_no_tool_events_before_outer_closer(self):
        p = frontend.OutputParser(thinking=False, tools=TOOLS, stream_tools=True,
                                  declared_only=True, undeclared_tool_policy="passthrough")
        raw = body()
        events = []
        with mock.patch.object(frontend.LOGGER, "warning"):
            for char in raw[:-1]:
                events.extend(p.feed(char))
            self.assertFalse([e for e in events if e.kind.startswith("tool_")])
            events.extend(p.feed(raw[-1:]))
            events.extend(p.finish("stop"))
        self.assertEqual(calls(events), [("bash", {"command": "pwd"})])

    def test_declared_call_is_unchanged(self):
        for policy in ("text", "passthrough"):
            events, p, warning = parse(body("shell"), policy, 1)
            self.assertEqual(calls(events), [("shell", {"command": "pwd"})])
            self.assertEqual((p.undeclared_as_text, p.undeclared_passthrough), (0, 0))
            warning.assert_not_called()

    def test_empty_catalog_does_not_enable_passthrough(self):
        for tools in (None, []):
            events, _, _ = parse(body(), "passthrough", 1, tools=tools)
            self.assertEqual(calls(events), [])
            self.assertEqual(text(events), body())

    def test_reasoning_is_not_promoted(self):
        for width in (0, 1):
            events, p, _ = parse(body() + "</think>\nDone.", "passthrough", width, thinking=True)
            self.assertEqual(calls(events), [])
            self.assertEqual(p.undeclared_passthrough, 0)
            self.assertIn("<function=bash>", "".join(e.text for e in events if e.kind == "reasoning"))

    def test_code_examples_remain_text(self):
        for raw in ("```xml\n" + body() + "\n```", "`" + body() + "`"):
            for width in (0, 1):
                events, p, _ = parse(raw, "passthrough", width)
                self.assertEqual(calls(events), [])
                self.assertEqual(p.undeclared_passthrough, 0)

    def test_recovery_does_not_expand_unknown_syntax(self):
        variants = [body().replace("<function=bash>", "<=bash>"),
                    body().replace("<tool_call>\n", "").replace("\n</tool_call>", ""),
                    '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>']
        for raw in variants:
            for width in (0, 1):
                events, _, _ = parse(raw, "passthrough", width, recover=True)
                self.assertEqual(calls(events), [])
                self.assertEqual(text(events), raw)

    def test_invalid_names_stay_text(self):
        for name in ("", "bad name", "bad/name", " bash", "bash ", "x" * 129):
            events, p, _ = parse(body(name), "passthrough", 1)
            self.assertEqual(calls(events), [])
            self.assertEqual(p.undeclared_passthrough, 0)

    def test_following_calls_are_not_lost_or_duplicated(self):
        raw = body() + "\n" + body("shell") + "\n" + body("missing")
        for width in (0, 1, 7):
            events, p, _ = parse(raw, "passthrough", width)
            self.assertEqual([name for name, _ in calls(events)], ["bash", "shell", "missing"])
            ids = [e.call.id for e in events if e.kind == "tool_call"]
            self.assertEqual(len(set(ids)), 3)
            self.assertEqual(p.undeclared_passthrough, 2)

    def test_whole_and_streamed_prose_agree(self):
        raw = "Before.\n" + body() + "\nAfter."
        events, _, _ = parse(raw, "passthrough")
        for width in (1, 3, 7):
            other, _, _ = parse(raw, "passthrough", width)
            self.assertEqual((calls(other), text(other)), (calls(events), text(events)))

    def test_truncation_never_announces_a_tool(self):
        for raw in (body()[:-1], body().split("</parameter>")[0],
                    body().replace("\n</tool_call>", "")):
            for reason in ("stop", "length", "cancel", "error"):
                events, p, _ = parse(raw, "passthrough", 1, reason=reason)
                self.assertFalse([e for e in events if e.kind.startswith("tool_")])
                self.assertEqual(text(events), raw)
                self.assertEqual((p.undeclared_as_text, p.undeclared_passthrough), (1, 0))

    def test_malformed_and_multiple_functions_are_not_promoted(self):
        variants = [body().replace("</function>", ""),
                    body().replace("</function>", "</function>junk"),
                    body().replace("</tool_call>", "<function=shell></function></tool_call>")]
        for raw in variants:
            for width in (0, 1):
                events, _, _ = parse(raw, "passthrough", width, recover=True)
                self.assertFalse([e for e in events if e.kind.startswith("tool_")])
                self.assertEqual(text(events), raw)

    def test_unknown_buffer_is_bounded(self):
        with mock.patch.object(frontend, "UNKNOWN_TOOL_CALL_MAX", 256):
            for raw in (body(command="x" * 512), body(command="x" * 512).split("</parameter>")[0]):
                for width in (0, 1, 7):
                    events, p, _ = parse(raw, "passthrough", width)
                    self.assertFalse([e for e in events if e.kind.startswith("tool_")])
                    self.assertEqual(text(events), raw)
                    self.assertEqual(p.undeclared_passthrough, 0)

    def test_exact_buffer_limit_is_chunk_independent(self):
        raw = body(command="x" * 100)
        size = len(raw) - len(frontend.CALL_START) - len(frontend.CALL_END)
        with mock.patch.object(frontend, "UNKNOWN_TOOL_CALL_MAX", size):
            for width in (0, 1, 7):
                events, _, _ = parse(raw, "passthrough", width)
                self.assertEqual(calls(events), [("bash", {"command": "x" * 100})])

    def test_diagnostics_do_not_log_arguments(self):
        for policy in ("text", "passthrough"):
            _, _, warning = parse(body(command="private_argument_marker"), policy, 1)
            self.assertEqual(warning.call_count, 1)
            self.assertNotIn("private_argument_marker", str(warning.call_args_list))

    def test_invalid_policy_is_rejected(self):
        for value in (True, False, None, "", "pass", [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                frontend.OutputParser(undeclared_tool_policy=value)


class ServicePolicy(unittest.TestCase):
    def make_service(self, script, policy="passthrough"):
        tok = server.ByteTokenizer()
        engine = server.MockEngine(tok, script)
        svc = server.Service(engine, tok, frontend.ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.undeclared_tool_policy = policy
        return svc

    def effective_policy(self, request=None, declared=True, force=None, config="passthrough"):
        svc = self.make_service(body("shell"), config)
        with mock.patch.object(server, "OutputParser", wraps=frontend.OutputParser) as factory:
            list(svc.run(svc.tok.encode("u"), False, TOOLS, 2048, request or {}, threading.Event(),
                         force=force, declared_only=declared))
        return factory.call_args.kwargs["undeclared_tool_policy"]

    def test_default_service_is_strict(self):
        self.assertEqual(self.effective_policy(config="text"), "text")

    def test_auto_request_uses_opt_in(self):
        self.assertEqual(self.effective_policy(), "passthrough")
        self.assertEqual(self.effective_policy({"tool_choice": "auto"}), "passthrough")

    def test_forced_none_required_and_unknown_choices_stay_strict(self):
        for choice in ("none", "required", "other", {"type": "function", "function": {"name": "shell"}}):
            self.assertEqual(self.effective_policy({"tool_choice": choice}), "text")
        self.assertEqual(self.effective_policy(force="<tool_call>\n<function=shell>\n"), "text")

    def test_mcp_and_responses_paths_stay_strict(self):
        self.assertEqual(self.effective_policy({"strata_mcp": True}), "text")
        self.assertEqual(self.effective_policy(declared=False), "text")

    def test_request_cannot_enable_server_policy(self):
        self.assertEqual(self.effective_policy({"undeclared_tool_policy": "passthrough"}, config="text"), "text")

    def test_invalid_config_fails_before_engine_creation(self):
        for value in (True, None, "pass", [], {}):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                config = Path(folder) / "policy.json"
                config.write_text(json.dumps({"undeclared_tool_policy": value}))
                error = io.StringIO()
                with mock.patch.object(server, "MockEngine") as engine, contextlib.redirect_stderr(error):
                    with self.assertRaises(SystemExit) as raised:
                        server.main(["--engine", "mock", "--config", str(config), "--port", "0"])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("undeclared_tool_policy", error.getvalue())
                engine.assert_not_called()

    @contextlib.contextmanager
    def endpoint(self, script, policy="passthrough"):
        svc = self.make_service(script, policy)
        httpd = server.serve(svc, host="127.0.0.1", port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            yield base, svc
        finally:
            httpd.shutdown()
            httpd.server_close()

    def post(self, base, request):
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(request).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read().decode()

    def request(self, **overrides):
        return {"model": "m", "messages": [{"role": "user", "content": "Run pwd."}],
                "tools": API_TOOLS, "max_tokens": 2048, **overrides}

    def test_http_default_keeps_text_and_stop(self):
        with self.endpoint("</think>\n" + body(), "text") as (base, _):
            result = json.loads(self.post(base, self.request()))["choices"][0]
        self.assertEqual(result["finish_reason"], "stop")
        self.assertNotIn("tool_calls", result["message"])
        self.assertEqual(result["message"]["content"], body())

    def test_http_passthrough_delivers_original_name(self):
        with self.endpoint("</think>\n" + body()) as (base, _):
            result = json.loads(self.post(base, self.request()))["choices"][0]
        self.assertEqual(result["finish_reason"], "tool_calls")
        call = result["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "bash")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"command": "pwd"})
        self.assertFalse(result["message"].get("content"))

    def test_http_stream_has_one_complete_call(self):
        with self.endpoint("</think>\n" + body()) as (base, _):
            response = self.post(base, self.request(stream=True))
        chunks = [json.loads(line[6:]) for line in response.splitlines()
                  if line.startswith("data: ") and line[6:] != "[DONE]"]
        result = server.openai_collect(iter(chunks))["choices"][0]
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual(len(result["message"]["tool_calls"]), 1)
        call = result["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "bash")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"command": "pwd"})

    def test_http_truncated_call_has_no_tool_delta(self):
        with self.endpoint("</think>\n" + body()[:-1]) as (base, _):
            response = self.post(base, self.request(stream=True))
        self.assertNotIn('"tool_calls"', response)

    def test_http_none_and_empty_tools_do_not_enable_passthrough(self):
        for overrides in ({"tool_choice": "none"}, {"tools": []}):
            with self.endpoint("</think>\n" + body()) as (base, _):
                result = json.loads(self.post(base, self.request(**overrides)))["choices"][0]
            self.assertNotIn("tool_calls", result["message"])

    def test_client_error_can_be_followed_by_a_valid_call(self):
        # Scripted protocol round-trip, not a claim about a real model or OpenCode.
        scripts = ["</think>\n" + body(), "</think>\n" + body("shell")]
        with self.endpoint(scripts) as (base, svc):
            request = self.request()
            first = json.loads(self.post(base, request))["choices"][0]["message"]
            call = first["tool_calls"][0]
            self.assertEqual(call["function"]["name"], "bash")
            error = "No tool named bash is currently available. Use shell from the available tool list."
            request["messages"] += [first, {"role": "tool", "tool_call_id": call["id"], "content": error}]
            second = json.loads(self.post(base, request))["choices"][0]["message"]
            self.assertEqual(second["tool_calls"][0]["function"]["name"], "shell")
            self.assertIn(error, svc.tok.decode(svc.engine.last_prompt))


if __name__ == "__main__":
    unittest.main()
