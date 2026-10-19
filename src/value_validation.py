import re


class DataTypeValidation:
    """
    Validator for JSON data types, numbers, and strict string escape rules.
    """
    def is_valid_number_prefix(self, s: str) -> bool:
        """Check if a string is a valid prefix for a JSON number."""
        if not s:
            return False
        return bool(re.fullmatch(r"-?(?:(?:0|[1-9]\d*)(?:\.\d*)?)?", s))

    def is_valid_number(self, s: str) -> bool:
        """Check if a string represents a fully valid JSON number."""
        return bool(re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", s))

    def _not_escaped_escapes(self, s: str) -> bool:
        """
        Check if the string ends with an odd number of backslashes
        (trailing escape).
        """
        backslash_count = 0
        for char in reversed(s):
            if char == '\\':
                backslash_count += 1
            else:
                break
        return backslash_count % 2 == 1

    def _has_raw_control_chars(self, s: str) -> bool:
        """Check if the string contains unescaped raw control characters."""
        return any(ord(c) < 32 for c in s)

    def _has_invalid_escapes(self, s: str) -> bool:
        """Check if the string contains invalid JSON escape sequences."""
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
            else:
                i += 1
        return False

    def _has_unescaped_internal_quotes(self, s: str) -> bool:
        """Check if the string contains unescaped internal double quotes."""
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
        """
        Validate if a string strictly conforms to JSON string literal rules.
        """
        if self._not_escaped_escapes(s):
            return False
        if self._has_raw_control_chars(s):
            return False
        if self._has_invalid_escapes(s):
            return False
        if self._has_unescaped_internal_quotes(s):
            return False
        return True

    def json_string_status(self, s: str) -> str:
        """Evaluate the parsing status of a JSON string fragment.

        Returns
        -------
        str
            'open' if valid so far, 'closed' if closed by a terminal quote, or
            'invalid'.
        """
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            if c == "\\":
                if i + 1 >= n:
                    return "open"
                e = s[i + 1]
                if e in '"\\/bfnrt':
                    i += 2
                elif e == "u":
                    h = s[i + 2:i + 6]
                    if any(x not in "0123456789abcdefABCDEF" for x in h):
                        return "invalid"
                    if len(h) < 4:
                        return "open"
                    else:
                        i += 6
                else:
                    i += 1
            elif c == '"':
                return "closed" if i == n - 1 else "invalid"
            elif ord(c) < 32 or c == "\ufffd":
                return "invalid"
            else:
                i += 1
        return "open"
