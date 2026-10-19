def _bytes_to_unicode() -> dict[int, str]:
    """Return a list of unicode string tokens corresponding to byte values.

    Maps byte values to readable unicode characters to ensure all bytes can
    be represented as strings in the vocabulary (similar to GPT-2 byte-level
    tokenizer encoding).

    Returns
    -------
    dict[int, str]
        A mapping from integer byte values to their corresponding unicode
        characters.
    """
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
    """Decoder for mapping model token IDs back to readable text strings.

    Handles byte-level unicode transformations and token lookup using the
    provided vocabulary mapping.
    """
    def __init__(self, vocab: dict[str, int]):
        """Initialize the TokenDecoder with a vocabulary mapping.

        Parameters
        ----------
        vocab : dict[str, int]
            A dictionary mapping token strings to their respective integer IDs.
        """
        self.byte_decoder = {c: b for b, c in _bytes_to_unicode().items()}
        size = max(vocab.values()) + 1
        self.texts: list[str] = [""] * size
        for token_str, token_id in vocab.items():
            self.texts[token_id] = self._decode(token_str)

    def _decode(self, raw_token: str) -> str:
        """Decode a raw byte-mapped token string into its utf-8 text
        representation.

        Parameters
        ----------
        raw_token : str
            The raw token string from the vocabulary.

        Returns
        -------
        str
            The decoded utf-8 string, or the raw token if decoding fails.
        """
        try:
            data = bytes(self.byte_decoder[c] for c in raw_token)
        except KeyError:
            return raw_token
        return data.decode("utf-8", errors="replace")

    def decode_token(self, token_id: int) -> str:
        """Retrieve the decoded text string for a given token ID.

        Parameters
        ----------
        token_id : int
            The integer ID of the token.

        Returns
        -------
        str
            The corresponding decoded text string, or an empty string if out
            of bounds.
        """
        if 0 <= token_id < len(self.texts):
            return self.texts[token_id]
        return ""
