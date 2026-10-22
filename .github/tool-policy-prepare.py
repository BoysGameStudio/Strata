"""One-shot preparation of native source changes, guarded by exact base blobs."""
from pathlib import Path
import hashlib

ROOT = Path.cwd()
BASE = {
    'serve/frontend.py': '4cf0b4fabb6f90f6346bc3bfa1e2417af879f143',
    'serve/server.py': '632d3c042ebbe6364410fa046038a588f07acaf1',
}

def replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Expected one patch anchor: {old[:100]!r}')
    return source.replace(old, new, 1)

sources = {}
for name, sha in BASE.items():
    data = (ROOT / name).read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if actual != sha:
        raise RuntimeError(f'{name}: unexpected base {actual}; refusing to overwrite')
    sources[name] = data.decode()

s = sources['serve/frontend.py']
s = replace(s, 'class OutputParser:\n',
    '# Bound the extra buffering for opt-in unknown calls (characters, not tokens).\n'
    'UNKNOWN_TOOL_CALL_MAX = 1024 * 1024\n\n\nclass OutputParser:\n')
s = replace(s, '                 recover: bool = False, declared_only: bool = False):',
    '                 recover: bool = False, declared_only: bool = False,\n'
    '                 undeclared_tool_policy: str = "text"):')
s = replace(s, '        self.declared_only = declared_only\n', '''        self.declared_only = declared_only
        # Opt-in error feedback for clients that reject unavailable tools. This
        # never registers, aliases, authorizes, or executes an unknown tool.
        if undeclared_tool_policy not in ("text", "passthrough"):
            raise ValueError('undeclared_tool_policy must be "text" or "passthrough"')
        self.undeclared_tool_policy = undeclared_tool_policy
        self.undeclared_as_text = 0
        self.undeclared_passthrough = 0
        self._undeclared_name = ""
''')
s = replace(s, '    def _loose_open(self, after: str):\n', '''    def _note_undeclared(self, name: str, passthrough: bool) -> None:
        # Log names only, never arguments or the surrounding conversation.
        if passthrough:
            self.undeclared_passthrough += 1
            action = "passed to client for validation; not executed by this policy"
        else:
            self.undeclared_as_text += 1
            action = "returned as text"
        LOGGER.warning("[strata] undeclared tool %r: %s", name[:128], action)

    def _loose_open(self, after: str):
''')
s = replace(s, '            if self.state == "rcall":\n', '''            if self.state == "ucall":
                # An opt-in unknown call waits for its complete canonical envelope.
                # Unlike a declared call, it must never leak partial tool events.
                end = self._wrapped_call_end()
                if end < 0 and len(self.buf) <= UNKNOWN_TOOL_CALL_MAX + len(CALL_END):
                    return out
                if end >= 0:
                    body = self.buf[:end]
                    raw = CALL_START + body + CALL_END
                    self.buf = self.buf[end + len(CALL_END):]
                else:                               # bounded fallback for a missing closer
                    body, raw = "", CALL_START + self.buf
                    self.buf = ""
                extent = function_end(body)
                call = None
                if 0 <= extent and len(body) <= UNKNOWN_TOOL_CALL_MAX and not body[extent:].strip():
                    try:
                        call = parse_tool_call(body)
                    except ValueError:
                        pass
                self._note_undeclared(self._undeclared_name, call is not None)
                self._undeclared_name = ""
                self._reset_scan()
                self.state, self.lead = "content", call is not None
                if call is None:
                    out.append(Event("content", self._track(raw)))
                else:
                    if self.stream_tools:
                        out.append(Event("tool_start", call=call))
                        out.append(Event("tool_args", json.dumps(call.arguments, ensure_ascii=False), call=call))
                    out.append(Event("tool_call", call=call))
                continue
            if self.state == "rcall":
''')
s = replace(s, '''                    if gt >= 0 and (nl < 0 or gt < nl) and not self._callable(after[len(FUNC_START):gt].strip()):
                        out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                        self.buf = self.buf[i + len(CALL_START):]
                        continue
''', '''                    if gt >= 0 and (nl < 0 or gt < nl) and not self._callable(after[len(FUNC_START):gt].strip()):
                        name = after[len(FUNC_START):gt]
                        # Do not relax _callable(): drifted, bare, JSON and reasoning
                        # recovery retain their existing declared-name checks.
                        passthrough = (self.undeclared_tool_policy == "passthrough" and bool(self.schemas)
                                       and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name) is not None)
                        if passthrough:
                            if i:
                                out.append(Event("content", self._track(self.buf[:i])))
                            self.buf = self.buf[i + len(CALL_START):]
                            self._undeclared_name = name
                            self._reset_scan()
                            self.state = "ucall"
                            continue
                        self._note_undeclared(name, False)
                        out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                        self.buf = self.buf[i + len(CALL_START):]
                        continue
''')
s = replace(s, '''        out = []
        if self.pending:
            out += self._release(reason in (None, "stop")''', '''        out = []
        if self.state == "ucall":
            # A cut or malformed unknown call is text, including a missing outer
            # closer. No tool_start/tool_args have been sent for this buffer.
            raw = CALL_START + self.buf
            self._note_undeclared(self._undeclared_name, False)
            self._undeclared_name = ""
            self.buf, self.state = "", "content"
            self._reset_scan()
            return [Event("content", self._track(raw))]
        if self.pending:
            out += self._release(reason in (None, "stop")''')
sources['serve/frontend.py'] = s

s = sources['serve/server.py']
s = replace(s, "        self.tool_call_recovery = False                   # opt-in: tool calls in forms next to the template's\n",
    "        self.tool_call_recovery = False                   # opt-in: tool calls in forms next to the template's\n"
    '        self.undeclared_tool_policy = "text"              # opt-in: client-side unknown-tool feedback\n')
s = replace(s, '''        parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True, recover=self.tool_call_recovery,
                              declared_only=declared_only)
''', '''        parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True, recover=self.tool_call_recovery,
                              declared_only=declared_only,
                              undeclared_tool_policy=(self.undeclared_tool_policy
                                  if declared_only and not force and not (sampling or {}).get("strata_mcp")
                                  and tool_choice_of((sampling or {}).get("tool_choice"))[0] == "auto"
                                  else "text"))
''')
s = replace(s, '    if cfg.get("tokenizer"):\n', '''    undeclared_policy = cfg.get("undeclared_tool_policy", "text")
    if undeclared_policy not in ("text", "passthrough"):
        ap.error('config undeclared_tool_policy must be "text" or "passthrough"')
    if cfg.get("tokenizer"):
''')
s = replace(s, '    svc.tool_call_recovery = recovery\n', '''    svc.tool_call_recovery = recovery
    svc.undeclared_tool_policy = undeclared_policy
    if undeclared_policy == "passthrough":
        print('[strata] undeclared_tool_policy=passthrough: eligible external clients must reject '
              'unavailable tools; no aliasing or automatic execution', flush=True)
''')
sources['serve/server.py'] = s

# Validate every result before writing either source file.
for name, source in sources.items():
    compile(source, name, 'exec')
for name, source in sources.items():
    (ROOT / name).write_text(source, encoding='utf-8')
    print('Prepared', name)
