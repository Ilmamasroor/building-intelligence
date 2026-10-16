"""Story 3.2: incremental extraction of the JSON "answer" string value from streamed pieces.

Only decides what is shown early; the full reply is parsed and validated after the stream ends.
"""

from __future__ import annotations

_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "/": "/", "\\": "\\",
                   '"': '"'}


class AnswerExtractor:
    def __init__(self) -> None:
        self._depth = 0
        self._in_string = False
        self._mode = "other"  # other | key | answer  (what the open string is)
        self._buf: list[str] = []  # contents of a non-answer string (to read keys)
        self._pending_key: str | None = None  # string seen at depth 1, maybe a key
        self._current_key: str | None = None  # key whose value we are in
        self._esc = False
        self._unicode: str | None = None  # hex digits collected after \u
        self._high: str | None = None  # a high surrogate waiting for its pair
        self.done = False  # the answer value has been fully emitted

    def feed(self, piece: str) -> str:
        out: list[str] = []
        for ch in piece:
            if self._in_string:
                self._string_char(ch, out)
            else:
                self._structure_char(ch)
        return "".join(out)

    # -- outside strings --
    def _structure_char(self, ch: str) -> None:
        if ch == '"':
            self._in_string = True
            if self._depth == 1 and self._current_key == "answer" and not self.done:
                self._mode = "answer"
            else:
                self._mode = "key" if self._depth == 1 and self._current_key is None else "other"
                self._buf = []
        elif ch in "{[":
            self._depth += 1
            self._pending_key = None
        elif ch in "}]":
            self._depth = max(0, self._depth - 1)
            if self._depth == 1:
                self._current_key = None
        elif self._depth == 1:
            if ch == ":" and self._pending_key is not None:
                self._current_key, self._pending_key = self._pending_key, None
            elif ch == ",":
                self._current_key = self._pending_key = None

    # -- inside strings --
    def _string_char(self, ch: str, out: list[str]) -> None:
        decoded: str | None
        if self._unicode is not None:
            self._unicode += ch
            if len(self._unicode) < 4:
                return
            try:
                decoded = chr(int(self._unicode, 16))
            except ValueError:
                decoded = "�"
            self._unicode = None
            decoded = self._pair(decoded)
        elif self._esc:
            self._esc = False
            if ch == "u":
                self._unicode = ""
                return
            decoded = self._pair(_SIMPLE_ESCAPES.get(ch, ch))
        elif ch == "\\":
            self._esc = True
            return
        elif ch == '"':
            self._close_string()
            return
        else:
            decoded = self._pair(ch)
        if decoded:
            if self._mode == "answer":
                out.append(decoded)
            elif self._mode == "key":
                self._buf.append(decoded)

    def _pair(self, ch: str) -> str:
        """Join a \\ud83d\\ude00 style surrogate pair; pass everything else through."""
        if self._high is not None:
            high, self._high = self._high, None
            if "\udc00" <= ch <= "\udfff":
                return chr(0x10000 + ((ord(high) - 0xD800) << 10) + (ord(ch) - 0xDC00))
            return "�" + self._pair(ch)
        if "\ud800" <= ch <= "\udbff":
            self._high = ch
            return ""
        return ch

    def _close_string(self) -> None:
        self._in_string = False
        self._high = None
        if self._mode == "answer":
            self.done = True
            self._current_key = None
        elif self._mode == "key":
            self._pending_key = "".join(self._buf)
        elif self._depth == 1:
            self._current_key = None
