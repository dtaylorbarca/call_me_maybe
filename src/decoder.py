def _bytes_to_unicode() -> dict[int, str]:
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class TokenDecoder:
    def __init__(self, vocab: dict[str, int]):
        self.byte_decoder = {c: b for b, c in _bytes_to_unicode().items()}
        size = max(vocab.values()) + 1
        self.texts: list[str] = [""] * size
        for token_str, token_id in vocab.items():
            self.texts[token_id] = self._decode(token_str)

    def _decode(self, raw_token: str) -> str:
        try:
            data = bytes(self.byte_decoder[c] for c in raw_token)
        except KeyError:
            return raw_token
        return data.decode("utf-8", errors="replace")

    def decode_token(self, token_id: int) -> str:
        if 0 <= token_id < len(self.texts):
            return self.texts[token_id]
        return ""
