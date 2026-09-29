import json
from pathlib import Path


class JSONSyntaxTokenIDs:
    def __init__(self, vocab_path: str) -> None:
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)

        self.curly_open = self._get_id("{")
        self.curly_close = self._get_id("}")
        self.colon = self._get_id(":")
        self.comma = self._get_id(",")
        self.double_quotes = self._get_id('"')
        self.single_quotes = self._get_id("'")
        self.square_open = self._get_id("[")
        self.square_close = self._get_id("]")

    def _get_id(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        raise Exception("Intended token not available in LLM vocab file")

    def get_all(self) -> dict[str, int]:
        return {
            "{": self.curly_open,
            "}": self.curly_close,
            ":": self.colon,
            ",": self.comma,
            '"': self.double_quotes,
            "'": self.single_quotes,
            "[": self.square_open,
            "]": self.square_close
        }
