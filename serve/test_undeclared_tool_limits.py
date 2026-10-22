"""An overflowed unknown envelope must never expose its contents as tool actions."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import frontend

TOOLS = [{"name": "shell", "parameters": {"properties": {"command": {"type": "string"}}}}]


def body(name="bash", command="pwd"):
    return (f"<tool_call>\n<function={name}>\n<parameter=command>\n{command}\n"
            "</parameter>\n</function>\n</tool_call>")


class OverflowIsolation(unittest.TestCase):
    def test_nested_and_subsequent_calls_stay_text_after_overflow(self):
        overflow = body(command="x" * 512 + "\n" + body("shell")) + "\n" + body("shell")
        for prefix in ("", body("shell")):
            for width in (0, 1, 7):
                for recover in (False, True):
                    with self.subTest(prefix=bool(prefix), width=width, recover=recover):
                        parser = frontend.OutputParser(thinking=False, tools=TOOLS, stream_tools=True,
                            recover=recover, declared_only=True, undeclared_tool_policy="passthrough")
                        raw = prefix + overflow
                        chunks = [raw] if not width else [raw[i:i + width] for i in range(0, len(raw), width)]
                        events = []
                        with mock.patch.object(frontend, "UNKNOWN_TOOL_CALL_MAX", 256), \
                                mock.patch.object(frontend.LOGGER, "warning"):
                            for chunk in chunks:
                                events.extend(parser.feed(chunk))
                            events.extend(parser.finish("stop"))
                        calls = [e.call.name for e in events if e.kind == "tool_call"]
                        self.assertEqual(calls, ["shell"] if prefix else [])
                        text = "".join(e.text or "" for e in events if e.kind == "content")
                        self.assertEqual(text, overflow)
                        self.assertEqual(parser.undeclared_passthrough, 0)
                        self.assertEqual(parser.undeclared_as_text, 1)

    def test_overflow_drains_text_without_retaining_a_buffer(self):
        parser = frontend.OutputParser(thinking=False, tools=TOOLS, stream_tools=True,
                                      declared_only=True, undeclared_tool_policy="passthrough")
        with mock.patch.object(frontend, "UNKNOWN_TOOL_CALL_MAX", 256), \
                mock.patch.object(frontend.LOGGER, "warning"):
            events = parser.feed(body(command="x" * 512).split("</parameter>")[0])
            self.assertEqual(parser.state, "utext")
            for chunk in ("x" * 4096, body("shell"), "tail"):
                events.extend(parser.feed(chunk))
                self.assertEqual(parser.buf, "")
            events.extend(parser.finish("length"))
        self.assertFalse([e for e in events if e.kind.startswith("tool_")])
        self.assertEqual(parser.undeclared_as_text, 1)


if __name__ == "__main__":
    unittest.main()
