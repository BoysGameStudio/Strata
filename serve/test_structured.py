"""Exercise the JSON contract through HTTP, including H3's strict Story shape."""
import json
import os
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve import structured
from serve.server import ByteTokenizer, MockEngine, Service, serve

try:
    import jsonschema  # noqa: F401
    HAVE_JSONSCHEMA = True
except ImportError:                  # optional: json_schema answers are then only checked to be one object
    HAVE_JSONSCHEMA = False


STORY = {"title": "Photobooth", "sheet": "A single continuous shot.", "cast": [],
         "boxes": [{"box": 1, "duration_ms": 15000, "new_shot": True, "beat": "Hailee changes poses.",
                    "on_screen": ["Hailee"], "references": [], "ends": "She smiles."}], "suggestions": []}


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TEXT = {"type": "string"}
STORY_SCHEMA = object_schema({
    "title": TEXT, "sheet": TEXT, "cast": {"type": "array", "items": object_schema({"name": TEXT,
        "references": {"type": "array", "items": TEXT, "maxItems": 0}})},
    "boxes": {"type": "array", "minItems": 1, "maxItems": 1, "items": object_schema({
        "box": {"type": "integer", "minimum": 1, "maximum": 1},
        "duration_ms": {"type": "integer", "minimum": 1, "maximum": 15000}, "new_shot": {"type": "boolean"},
        "beat": TEXT, "on_screen": {"type": "array", "items": TEXT},
        "references": {"type": "array", "items": TEXT, "maxItems": 0}, "ends": TEXT})},
    "suggestions": {"type": "array", "items": object_schema({"what": TEXT, "why": TEXT,
        "boxes": {"type": "array", "items": {"type": "integer"}}, "description": TEXT})}})
FORMAT = {"type": "json_schema", "json_schema": {"name": "h3_story", "strict": True, "schema": STORY_SCHEMA}}


class Structured(unittest.TestCase):
    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, json.dumps(STORY), max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, script=None, **kwargs):
        if script is not None:
            self.engine = self.svc.engine = MockEngine(self.tok, script, max_context=16384)
        req = {"messages": [{"role": "system", "content": "Write a film plan."},
                            {"role": "user", "content": "Photobooth. 15 seconds."}],
               "response_format": FORMAT, "reasoning_effort": "none", **kwargs}
        return self.request("/v1/chat/completions", req)

    def test_h3_schema_success_and_stream_with_reasoning(self):
        code, reply = self.chat()
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(reply["choices"][0]["message"]["content"]), STORY)
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertIn("OUTPUT FORMAT REQUIREMENT", prompt)
        self.assertIn('"duration_ms"', prompt)
        code, raw = self.chat("Consider the shot.</think>\n" + json.dumps(STORY), stream=True, reasoning_effort="low")
        self.assertEqual(code, 200)
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertEqual(json.loads(content), STORY)
        self.assertIn("reasoning_content", raw)
        self.assertIn("data: [DONE]", raw)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_invalid_generation_is_an_error_and_never_streams_content(self):
        for script in ("## Film Plan - Photobooth", '{"title":"Only a title"}',
                       json.dumps({**STORY, "extra": True}),
                       json.dumps({**STORY, "boxes": [{**STORY["boxes"][0], "duration_ms": "15000"}]})):
            with self.subTest(script=script):
                code, reply = self.chat(script)
                self.assertEqual(code, 502)
                self.assertEqual(reply["error"]["code"], "structured_output_failed")
                _, raw = self.chat(script, stream=True)
                self.assertIn('"structured_output_failed"', raw)
                self.assertNotIn('"delta"', raw)
                self.assertIn("data: [DONE]", raw)

    def test_object_mode_strict_json_and_truncation(self):
        fmt = {"type": "json_object"}
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt)[0], 200)
        for script in ('[]', '{"x":NaN}', '{"x":1,"x":2}', '{"x":1e999}', 'no json at all', '{"x":'):
            self.assertEqual(self.chat(script, response_format=fmt)[0], 502)
        # #762: the JSON object is taken out of a code fence or the prose around it (the object itself stays strict)
        for script in ('```json\n{}\n```', 'Here it is: {"answer":4} - done'):
            self.assertEqual(self.chat(script, response_format=fmt)[0], 200, script)
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt, max_tokens=5)[0], 502)
        # Plain text callers retain normal streaming, without a JSON validation gate.
        self.assertEqual(self.chat("Prose.", response_format={"type": "text"})[0], 200)

    def test_escape_repair_is_opt_in_and_works_before_streaming(self):
        script = r'{"description":"条件为 $\alpha_0 > 1$，且 $\sigma > 0$"}'
        fmt = {"type": "json_object"}
        for setting in (None, "0", "true", "1"):
            with self.subTest(setting=setting), mock.patch.dict(os.environ):
                os.environ.pop("STRATA_JSON_ESCAPE_REPAIR", None)
                if setting is not None:
                    os.environ["STRATA_JSON_ESCAPE_REPAIR"] = setting
                code, reply = self.chat(script, response_format=fmt)
                if setting != "1":
                    self.assertEqual(code, 502)
                    self.assertEqual(reply["error"]["code"], "structured_output_failed")
                    continue
                self.assertEqual(code, 200, reply)
                expected = {"description": r"条件为 $\alpha_0 > 1$，且 $\sigma > 0$"}
                self.assertEqual(json.loads(reply["choices"][0]["message"]["content"]), expected)
                _, raw = self.chat(script, response_format=fmt, stream=True)
                chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
                self.assertEqual(json.loads(content), expected)
                self.assertIn("data: [DONE]", raw)

    def test_rejected_schema_does_not_load_the_engine_or_retry(self):
        with mock.patch.object(self.svc, "load") as load:
            code, _ = self.chat(response_format={"type": "json_schema"})
            self.assertEqual(code, 400)
            load.assert_not_called()
        self.engine = self.svc.engine = MockEngine(self.tok, "Markdown is not JSON", max_context=16384)
        with mock.patch.object(self.engine, "generate", wraps=self.engine.generate) as generate:
            self.assertEqual(self.chat()[0], 502)
            self.assertEqual(generate.call_count, 1)

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_invalid_schema_is_400_and_local_refs_work(self):
        for schema in ({"type": "object", "properties": {"x": {"type": "invalid"}}},
                       {"type": "object", "$ref": "https://other.example/schema"}):
            fmt = {"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": schema}}
            self.assertEqual(self.chat(response_format=fmt)[0], 400)
        schema = object_schema({"answer": {"$ref": "#/$defs/n"}})
        schema["$defs"] = {"n": {"type": "integer"}}
        fmt = {"type": "json_schema", "json_schema": {"name": "local", "schema": schema}}
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt)[0], 200)

    def test_malformed_formats_and_tools_are_rejected_before_loading(self):
        formats = ([], {"type": "unsupported"},
                   {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
                   {"type": "json_schema", "json_schema": {"name": "x", "strict": "true", "schema": {"type": "object"}}})
        with mock.patch.object(self.svc, "load") as load:
            for fmt in formats:
                with self.subTest(fmt=fmt):
                    self.assertEqual(self.chat(response_format=fmt)[0], 400)
            self.assertEqual(self.chat(tools=[{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}])[0], 400)
            self.assertEqual(self.chat(strata_mcp=True)[0], 400)
            self.assertEqual(self.request("/v1/chat/completions", [])[0], 400)
            load.assert_not_called()

    def test_without_jsonschema_an_object_is_still_required(self):
        # jsonschema is optional: without it json_schema answers are checked to be one JSON object, json_object
        # never needs it, and nothing is imported at server start
        with mock.patch.object(structured, "_jsonschema", False):
            self.assertEqual(self.chat()[0], 200)
            for script in ("## Film Plan - Photobooth", "[1, 2]"):
                code, reply = self.chat(script)
                self.assertEqual(code, 502, script)
                self.assertEqual(reply["error"]["code"], "structured_output_failed")
            self.assertEqual(self.chat('{"answer":4}', response_format={"type": "json_object"})[0], 200)

    def test_root_unions_of_object_shapes_are_accepted(self):
        # A root anyOf of object shapes (sent by apps written against llama.cpp's grammar path) still means
        # "one JSON object", so it is accepted; a non-object answer is still refused.
        claim = {"type": "object", "additionalProperties": False, "required": ["claim_text"],
                 "properties": {"claim_text": {"type": "string", "minLength": 1}}}
        empty = {"type": "object", "additionalProperties": False, "required": ["outcome"],
                 "properties": {"outcome": {"type": "string", "enum": ["no_substantive_content"]}}}
        schemas = [
            {"anyOf": [claim, empty]},
            {"oneOf": [claim, empty]},
            {"allOf": [claim, {"required": ["claim_text"]}]},
            {"$ref": "#/$defs/claim", "$defs": {"claim": claim}},
            {"anyOf": [{"$ref": "#/$defs/claim"}, {"anyOf": [empty]}], "$defs": {"claim": claim}},
        ]
        for schema in schemas:
            fmt = {"type": "json_schema", "json_schema": {"name": "union", "strict": True, "schema": schema}}
            with self.subTest(schema=schema):
                self.assertEqual(self.chat('{"claim_text":"The term is 12 months."}', response_format=fmt)[0], 200)
                self.assertEqual(self.chat("[1, 2]", response_format=fmt)[0], 502)

    def test_schemas_that_allow_non_objects_are_still_rejected(self):
        obj = {"type": "object"}
        schemas = [
            {"type": "array", "items": obj},
            {"type": "string"},
            {"type": ["object", "null"]},
            {"anyOf": [obj, {"type": "string"}]},
            {"anyOf": []},
            {"oneOf": [obj, {"type": "array"}]},
            {"allOf": [{"type": "string"}]},
            {"properties": {"x": {"type": "string"}}},           # no type: any JSON value validates
            {"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/a"}}},   # reference cycle
            {"$ref": "#/$defs/missing", "$defs": {}},
            {"$ref": "#/$defs/s", "$defs": {"s": {"type": "string"}}},
        ]
        with mock.patch.object(self.svc, "load") as load:
            for schema in schemas:
                fmt = {"type": "json_schema", "json_schema": {"name": "bad", "schema": schema}}
                with self.subTest(schema=schema):
                    code, reply = self.chat(response_format=fmt)
                    self.assertEqual(code, 400)
                    self.assertIn("only JSON objects", reply["error"]["message"])
            load.assert_not_called()

    def test_the_server_does_not_import_jsonschema_at_start(self):
        import subprocess
        import sys
        code = ("import sys; import serve.server; "
                "sys.exit(1 if any(m == 'jsonschema' or m.startswith('jsonschema.') for m in sys.modules) else 0)")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).resolve().parents[1]))
        self.assertEqual(r.returncode, 0)


class EscapeRepair(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"STRATA_JSON_ESCAPE_REPAIR": "1"})
        env.start()
        self.addCleanup(env.stop)
        _, self.validator = structured.prepare_format({"type": "json_object"}, [])

    def test_repairs_nested_values_without_changing_valid_escapes(self):
        script = r'{"values":["\alpha",{"s":"\sum"}],"ok":"\" \\ \/ \b \f \n \r \t \u4e2d","mixed":"\\alpha + \sigma","odd":"\\\alpha"}'
        expected = {"values": [r"\alpha", {"s": r"\sum"}],
                    "ok": '" \\ / \b \f \n \r \t 中', "mixed": r"\alpha + \sigma", "odd": r"\\alpha"}
        result = structured.validated_json(script, self.validator, "stop")
        self.assertEqual(json.loads(result), expected)
        self.assertEqual(structured.validated_json(result, self.validator, "stop"), result)

    def test_valid_json_stays_byte_identical_to_default(self):
        script = r'{"text":"\n\t\r\b\f\u4e2d\\alpha","nested":[true,null,3]}'
        enabled = structured.validated_json(script, self.validator, "stop")
        with mock.patch.dict(os.environ, {"STRATA_JSON_ESCAPE_REPAIR": "0"}):
            self.assertEqual(structured.validated_json(script, self.validator, "stop"), enabled)

    def test_other_invalid_json_is_not_repaired(self):
        for script in (r'{"\alpha":"x"}', r'{"x":"\alpha","y":"\uZZZZ"}',
                       r'{"x":"\alpha","x":2}', r'{"x":"\alpha","y":NaN}',
                       r'{"x":"\alpha","y":1e999}', r'{"x":"\alpha" "y":2}',
                       r'{"x":"\alpha', r'{"x":"\alpha' + '\\', r'{"x":"\alpha","y":"line' + '\n' + 'break"}'):
            with self.subTest(script=script), self.assertRaises(structured.StructuredOutputError):
                structured.validated_json(script, self.validator, "stop")

    def test_repair_does_not_bypass_finish_or_validation(self):
        script = r'{"x":"\alpha"}'
        with self.assertRaisesRegex(structured.StructuredOutputError, "incomplete"):
            structured.validated_json(script, self.validator, "length")
        validator = mock.Mock()
        validator.iter_errors.return_value = iter([mock.Mock(absolute_path=("x",), message="wrong type")])
        with self.assertRaisesRegex(structured.StructuredOutputError, "JSON Schema"):
            structured.validated_json(script, validator, "stop")
        validator.iter_errors.assert_called_once_with({"x": r"\alpha"})

    def test_unterminated_escaped_string_does_not_rescan_each_quote(self):
        import subprocess
        import sys
        code = r'''
from serve import structured
_, validator = structured.prepare_format({"type": "json_object"}, [])
text = r'{"x":"\alpha' + r'\"' * 65536
try:
    structured.validated_json(text, validator, "stop")
except structured.StructuredOutputError:
    pass
else:
    raise AssertionError("unterminated string was accepted")
'''
        # Isolate the timeout so a quadratic regression cannot hang the test process.
        subprocess.run([sys.executable, "-B", "-c", code], timeout=5, check=True,
                       cwd=str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    unittest.main()
