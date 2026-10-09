class DataTypeValidation:
    def is_valid_number_prefix(self, s: str) -> bool:
        if not s:
            return False
        import re

        return bool(re.fullmatch(r"-?\d*\.?\d*", s))

    def is_valid_number(self, s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _not_escaped_escapes(self, s: str) -> bool:
        backslash_count = 0
        for char in reversed(s):
            if char == '\\':
                backslash_count += 1
            else:
                break

        return backslash_count % 2 == 1

    def _has_raw_control_chars(self, s: str) -> bool:
        return any(ord(c) < 32 for c in s)

    def _has_invalid_escapes(self, s: str) -> bool:
        i = 0
        n = len(s)
        valid_simples_escapes = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't'}

        while i < n:
            if s[i] == '\\':
                if i + 1 >= n:
                    return True
                next_char = s[i + 1]
                if next_char in valid_simples_escapes:
                    i += 2
                elif next_char == 'u':
                    if i + 5 >= n:
                        return True
                    hex_part = s[i + 2:i + 6]
                    if not all(
                            c in "0123456789abcdefABCDEF" for c in hex_part):
                        return True
                    i += 6
                else:
                    return True
                i += 6
            else:
                i += 1

        return False

    def _has_unescaped_internal_quotes(self, s: str) -> bool:
        i = 0
        n = len(s)
        while i < n:
            if s[i] == '\\':
                i += 2
            elif s[i] == '"':
                return True
            else:
                i += 1
        return False

    def is_valid_json_string(self, s: str) -> bool:
        if self._not_escaped_escapes(s):
            return False

        if self._has_raw_control_chars(s):
            return False

        if self._has_invalid_escapes(s):
            return False

        if self._has_unescaped_internal_quotes(s):
            return False

        return True
