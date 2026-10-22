"""One-shot guarded upgrade; removed after publishing the tested native blob."""
from pathlib import Path
import hashlib
p = Path('serve/frontend.py')
b = p.read_bytes()
assert hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == '7cc9dbf522fc29a8ce63dbee2aa66d43ead66414'
s = b.decode()
old = '''        while True:
            if self.state == "ucall":
'''
new = '''        while True:
            if self.state == "utext":
                # Overflow forfeits parsing for the rest of this turn. Resuming
                # at a later tag could turn an argument/example into an action.
                if self.buf:
                    out.append(Event("content", self.buf))
                    self.buf = ""
                return out
            if self.state == "ucall":
'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = '''                if end >= 0:
                    body = self.buf[:end]
                    raw = CALL_START + body + CALL_END
                    self.buf = self.buf[end + len(CALL_END):]
                else:                               # bounded fallback for a missing closer
                    body, raw = "", CALL_START + self.buf
                    self.buf = ""
                extent = function_end(body)
'''
new = '''                if end < 0 or end > UNKNOWN_TOOL_CALL_MAX:
                    # Whole and streamed input fail closed in the same way:
                    # do not interpret nested or subsequent calls after overflow.
                    self._note_undeclared(self._undeclared_name, False)
                    self._undeclared_name = ""
                    out.append(Event("content", CALL_START + self.buf))
                    self.buf, self.state = "", "utext"
                    self._reset_scan()
                    return out
                body = self.buf[:end]
                raw = CALL_START + body + CALL_END
                self.buf = self.buf[end + len(CALL_END):]
                extent = function_end(body)
'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
compile(s, str(p), 'exec')
p.write_text(s)
print('Hardened', p)
