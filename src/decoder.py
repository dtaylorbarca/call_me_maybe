class TokenDecoder:
    def __init__(self, vocab: dict[str, int]):
        self.id_to_token: dict[int, str] = {
            token_id: token_str for token_str, token_id in vocab.items()
        }

    def decode_token(self, token_id: int) -> str:
        """Converts a token ID back to its raw string representation."""
        if token_id not in self.id_to_token:
            raise KeyError(f"Token ID {token_id} not found in vocabulary.")

        raw_token = self.id_to_token[token_id]
        clean_token = raw_token.replace(" ", " ")
        clean_token = clean_token.replace("Ġ", " ")
        return clean_token
