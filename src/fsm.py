from .value_validation import DataTypeValidation
from .parser import FunctionTool, UserQuery
from .decoder import TokenDecoder
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from enum import Enum, auto
from typing import Any
import numpy as np
import json
import re


class JSONState(Enum):
    """States for the JSON generation Finite State Machine (FSM)."""
    START = auto()
    IN_NAME = auto()
    AFTER_NAME = auto()
    IN_PARAMS = auto()
    IN_PARAM_VALUE = auto()
    END = auto()
    FINISHED = auto()


MAX_STEPS = 600
MAX_VALUE_CHARS = 300
AFTER_PROMPT_PATH = '",\n        "name": "'
AFTER_NAME_PATH = ',\n        "parameters": {'
ESCAPE_OR_BACKSLASH = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')


def double_lone_backslashes(text: str) -> str:
    """Double every backslash that does not start a valid JSON escape sequence.

    Parameters
    ----------
    text : str
        The string text to process.

    Returns
    -------
    str
        The text with lone backslashes safely escaped.
    """
    return ESCAPE_OR_BACKSLASH.sub(
        lambda m: m.group(0) if len(m.group(0)) > 1 else "\\\\", text)


class FSMTokenizerIndex:
    """
    Index vocabulary tokens by their initial characters for fast FSM
    lookups.
    """

    def __init__(self, vocab: dict[str, int], decoder: TokenDecoder):
        """
        Initialize the tokenizer index and categorize tokens by buckets and
        types.

        Parameters
        ----------
        vocab : dict[str, int]
            Vocabulary mapping tokens to their integer IDs.
        decoder : TokenDecoder
            The token decoder instance used to decode token IDs into text.
        """
        self.decoder = decoder
        self.buckets: dict[str, list[int]] = {}
        self.all_token_ids = []
        self.safe_string_tokens = []
        self.exact_ids = {}

        decoder_texts = decoder.texts
        for token_id, text in enumerate(decoder_texts):
            if not text:
                continue
            self.all_token_ids.append(token_id)
            if text in ('"', ',', '}'):
                self.exact_ids[text] = token_id

            if '\n' not in text and '\r' not in text:
                self.safe_string_tokens.append(token_id)

            first_char = text[0]
            if first_char not in self.buckets:
                self.buckets[first_char] = []
            self.buckets[first_char].append(token_id)

    def get_tokens_starting_with(self, chars: str) -> list[int]:
        """Retrieve all token IDs that start with any of the specified
        characters.

        Parameters
        ----------
        chars : str
            A string of characters to check against token starting characters.

        Returns
        -------
        list[int]
            A list of matching token IDs.
        """
        result = []
        for c in chars:
            if c in self.buckets:
                result.extend(self.buckets[c])
        return result


class JSONStateMachine:
    """
    Finite State Machine to enforce strict JSON schema-compliant text
    generation.
    """

    def __init__(self, tools: list[FunctionTool], prompts: list[UserQuery],
                 vocab: dict[str, int], tokenizer_index: FSMTokenizerIndex,
                 decoder: TokenDecoder, is_first: bool, is_last: bool) -> None:
        """
        Initialize the JSON state machine with tools, prompts, vocabulary, and
        flags.

        Parameters
        ----------
        tools : list[FunctionTool]
            Available tool definitions.
        prompts : list[UserQuery]
            List of user input queries.
        vocab : dict[str, int]
            Token vocabulary mapping.
        tokenizer_index : FSMTokenizerIndex
            Indexed tokenizer helper.
        decoder : TokenDecoder
            Decoder for mapping tokens to text.
        is_first : bool
            Whether this prompt is the first in the batch output array.
        is_last : bool
            Whether this prompt is the last in the batch output array.
        """
        self.vocab = vocab
        self.tools: list[FunctionTool] = tools
        self.prompts: list[UserQuery] = prompts
        self.state: JSONState = JSONState.START
        self.selected_tool: FunctionTool | None = None
        self.buffer = ""
        self.user_query: str = prompts[0].prompt
        self.decoder = decoder
        self.used_params: list[str] = []
        self.finished_prompts: list[str] = []
        self.data_type_validator: DataTypeValidation = DataTypeValidation()
        self.token_index: FSMTokenizerIndex = tokenizer_index
        self.current_param = None
        self.is_first = is_first
        self.is_last = is_last
        self._looping = False

        if is_first:
            self.start_path = '[\n    {\n        "prompt": "'
        else:
            self.start_path = ',\n    {\n        "prompt": "'

        if is_last:
            self.end_path = '\n    }\n]'
        else:
            self.end_path = '\n    }'

    def _get_allowed_name_tokens(self) -> set[int]:
        """Determine allowed token IDs for generating the tool name."""
        allowed = set()
        current_len = len(self.buffer)

        valid_chars = set()
        targets = []
        for tool in self.tools:
            target = tool.name + '"'
            if not target.startswith(self.buffer):
                continue
            remaining_target = target[current_len:]
            if remaining_target:
                valid_chars.add(remaining_target[0])
                targets.append(remaining_target)
        token_ids_to_check = self.token_index.get_tokens_starting_with(
            "".join(valid_chars))

        for token_id in token_ids_to_check:
            token_text = self.decoder.decode_token(token_id)
            if not token_text:
                continue

            for remaining_target in targets:
                if remaining_target.startswith(token_text):
                    allowed.add(token_id)
                    break

        return allowed

    def _key_target(self, param_name: str) -> str:
        """Construct the target prefix string for a given parameter name."""
        target = f'"{param_name}": '
        if self.selected_tool:
            if self.selected_tool.parameters[param_name].get(
                    "type") == "string":
                target += '"'
        return target

    def _get_allowed_param_tokens(self) -> set[int]:
        """
        Determine allowed token IDs for generating parameter names and keys.
        """
        allowed: set[int] = set()
        if not self.selected_tool or not self.selected_tool.parameters:
            return allowed

        current_len = len(self.buffer)

        valid_chars = set()
        targets = []
        for param_name in self.selected_tool.parameters.keys():
            if param_name in self.used_params:
                continue

            target = self._key_target(param_name)
            if not target.startswith(self.buffer):
                continue
            remaining_target = target[current_len:]
            if remaining_target:
                valid_chars.add(remaining_target[0])
                targets.append(remaining_target)

        if not valid_chars:
            return allowed

        token_ids_to_check = self.token_index.get_tokens_starting_with(
            "".join(valid_chars))

        for token_id in token_ids_to_check:
            token_text = self.decoder.decode_token(token_id)
            if not token_text:
                continue

            for remaining_target in targets:
                if remaining_target.startswith(token_text):
                    allowed.add(token_id)
                    break

        return allowed

    def _value_token_kind(self, token_text: str) -> None | str:
        """
        Classify how a candidate token interacts with the current parameter
        value.

        Returns
        -------
        str | None
            'extend' if part of the value, 'stop' if it ends the value, or
            None if disallowed.
        """
        if not self.selected_tool or not self.current_param:
            return None
        params = self.selected_tool.parameters
        param_type = params[self.current_param].get("type")
        validator = self.data_type_validator
        candidate = self.buffer + token_text

        if param_type == "string":
            if validator.json_string_status(self.buffer) == "closed":
                return "stop" if token_text[0] in ",}" else None
            if (token_text[0] == '"' and
                    validator.json_string_status(
                        self.buffer + '"') == "closed"):
                return "stop"
            if validator.json_string_status(candidate) != "invalid":
                must_close = (len(self.buffer) >= MAX_VALUE_CHARS
                              or self._looping)
                if (must_close and validator.json_string_status(
                        self.buffer + '"') == "closed"):
                    return None
                return "extend"
            return None

        if param_type in ("number", "integer"):
            if param_type == "integer" and "." in candidate:
                return None
            is_done = validator.is_valid_number(self.buffer)
            if (validator.is_valid_number_prefix(candidate) and
                    not (is_done and len(self.buffer) >= 40)):
                return "extend"
            if is_done and token_text[0] in ",}":
                return "stop"
            return None

        if param_type == "boolean":
            if self.buffer in ("true", "false"):
                return "stop" if token_text[0] in ",}" else None
            if "true".startswith(candidate) or "false".startswith(candidate):
                return "extend"
        return None

    def _stop_token_id(self) -> None | int:
        """
        Retrieve the exact token ID required to terminate the current
        parameter value.
        """
        if not self.selected_tool or not self.current_param:
            return None
        params = self.selected_tool.parameters
        param_type = params[self.current_param].get("type")
        is_last = len(self.used_params) == len(params)
        exact = self.token_index.exact_ids
        if (param_type == "string" and
                self.data_type_validator.json_string_status(
                    self.buffer) != "closed"):
            return exact['"']
        return exact["}" if is_last else ","]

    def _ranked_ids(self, logits: list[float]) -> Any:
        """Yield token IDs sorted from best to worst logit score."""
        scores = np.asarray(logits, dtype=np.float32)[:len(self.decoder.texts)]
        k = min(64, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        yield from top.tolist()
        if k < len(scores):
            seen = set(top.tolist())
            for token_id in np.argsort(-scores).tolist():
                if token_id not in seen:
                    yield token_id

    def _is_looping(self) -> bool:
        """
        Check if the current value buffer is exhibiting a repetitive
        generation loop.
        """
        text = self.buffer
        for size in range(8, len(text) // 2 + 1):
            if text[-size:] == text[-2 * size:-size]:
                return True
        return False

    def _best_value_token(self, logits: list[float]) -> tuple[int, str] | None:
        """
        Find the best valid token ID and action kind for the current value
        generation step.
        """
        self._looping = self._is_looping()
        texts = self.decoder.texts
        for token_id in self._ranked_ids(logits):
            token_text = texts[token_id]
            if token_text:
                kind = self._value_token_kind(token_text)
                if kind:
                    return token_id, kind
        return None

    def _get_allowed_token_ids(self) -> set[int]:
        """Retrieve allowed token IDs based on the current FSM state."""
        match self.state:
            case JSONState.IN_NAME:
                return self._get_allowed_name_tokens()

            case JSONState.IN_PARAMS:
                return self._get_allowed_param_tokens()

            case _:
                return set()

    def _build_next_path(self, next_prompt_text: str) -> str:
        """
        Build the transition text between consecutive prompts in the batch.
        """
        return (
            f'}},\n<|im_end|>\n<|im_start|>user\n{next_prompt_text}<|im_end|>'
            '\n  {\n\t\t"prompt": "'
        )

    def _value_is_complete(self, text: str, param_type: str) -> bool:
        """
        Verify if a generated value string satisfies its expected data type
        completeness.
        """
        if param_type == "string":
            return bool(
                self.data_type_validator.json_string_status(text) == "closed"
            )
        if param_type in ("number", "integer"):
            return bool(self.data_type_validator.is_valid_number(text))
        return bool(text in ("true", "false"))

    def _update_state(self, token_str: str) -> None:
        """Advance the FSM state based on newly decoded token text."""
        self.buffer += token_str
        while True:
            prev_state = self.state
            match self.state:
                case JSONState.IN_NAME:
                    if '"' in self.buffer:
                        quote_index = self.buffer.index('"')
                        extracted_name = self.buffer[:quote_index]

                        matched = False
                        for tool in self.tools:
                            if extracted_name == tool.name:
                                self.selected_tool = tool
                                matched = True
                                break

                        if matched:
                            self.buffer = self.buffer[quote_index + 1:]
                            self.state = JSONState.AFTER_NAME
                        else:
                            raise ValueError(
                                "Generated unknown tool name: "
                                f"'{extracted_name}'")

                case JSONState.IN_PARAMS:
                    if self.selected_tool:
                        for param_name in (
                                self.selected_tool.parameters.keys()):
                            if param_name not in self.used_params:
                                expected_prefix = self._key_target(param_name)
                                if self.buffer.startswith(expected_prefix):
                                    self.current_param = param_name
                                    self.used_params.append(param_name)
                                    self.buffer = self.buffer[len(
                                        expected_prefix):]
                                    self.state = JSONState.IN_PARAM_VALUE
                                    break

                case JSONState.IN_PARAM_VALUE:
                    if not self.selected_tool or not self.current_param:
                        break

                    params = self.selected_tool.parameters
                    param_type = params[self.current_param].get("type")
                    is_last = len(self.used_params) == len(params)
                    terminator = "}" if is_last else ","
                    before = self.buffer[:len(self.buffer) - len(token_str)]

                    if (token_str == terminator and
                            param_type and
                            self._value_is_complete(before, param_type)):
                        self.buffer = ""
                        self.current_param = None
                        self.state = (JSONState.END if is_last
                                      else JSONState.IN_PARAMS)

                case JSONState.FINISHED:
                    break

            if self.state == prev_state:
                break

    def _forced_text(self) -> str | None:
        """
        Determine fixed structural text that must be forced into the
        generation stream.
        """
        match self.state:
            case JSONState.START:
                escaped_query = json.dumps(
                    self.user_query, ensure_ascii=False)[1:-1]
                text = self.start_path + escaped_query + AFTER_PROMPT_PATH
                self.state = JSONState.IN_NAME
            case JSONState.AFTER_NAME:
                text = AFTER_NAME_PATH
                has_params = bool(self.selected_tool
                                  and self.selected_tool.parameters)
                self.state = (JSONState.IN_PARAMS if has_params
                              else JSONState.END)
            case JSONState.END:
                has_params = bool(self.selected_tool
                                  and self.selected_tool.parameters)
                text = self.end_path if has_params else "}" + self.end_path
                self.state = JSONState.FINISHED
            case _:
                return None
        self.buffer = ""
        return text

    def _is_value_token_ok(self, token_text: str) -> bool:
        """
        Validate whether a single token text is acceptable for the current
        value.
        """
        if not self.selected_tool or not self.current_param:
            return False
        params = self.selected_tool.parameters
        param_type = params[self.current_param].get("type")
        is_last = len(self.used_params) == len(params)
        terminator = "}" if is_last else ","
        validator = self.data_type_validator
        candidate = self.buffer + token_text

        if param_type == "string":
            if validator.json_string_status(self.buffer) == "closed":
                return token_text == terminator
            return validator.json_string_status(candidate) != "invalid"

        if param_type in ("number", "integer"):
            if param_type == "integer" and "." in candidate:
                return False
            if validator.is_valid_number_prefix(candidate):
                return True
            return (validator.is_valid_number(self.buffer)
                    and token_text == terminator)

        if param_type == "boolean":
            if self.buffer in ("true", "false"):
                return token_text == terminator
            return ("true".startswith(candidate)
                    or "false".startswith(candidate))
        return False

    @staticmethod
    def _flatten_ids(encoded: Any) -> list[int]:
        """
        Flatten nested tensor or list structures into a clean 1-D list of
        integer token IDs.
        """
        data = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        if data and isinstance(data[0], list):
            data = data[0]
        return [int(v) for v in data]

    def constrained_decoding(self, model: Small_LLM_Model,
                             initial_input_ids: Any) -> str:
        """
        Execute the constrained decoding loop to generate valid JSON via the
        FSM.

        Parameters
        ----------
        model : Small_LLM_Model
            The language model instance used for inference.
        initial_input_ids : Any
            The starting token IDs for the generation prompt.

        Returns
        -------
        str
            The final generated and formatted JSON output string.
        """
        input_ids = self._flatten_ids(initial_input_ids)
        prompt_len = len(input_ids)
        steps = 0
        best: tuple[int, str] | None = None
        next_id: int | None = 0

        while self.state != JSONState.FINISHED:
            forced = self._forced_text()
            if forced is not None:
                input_ids.extend(self._flatten_ids(model.encode(forced)))
                continue

            steps += 1
            if steps > MAX_STEPS:
                raise RuntimeError("Generation did not finish in time")

            logits = model.get_logits_from_input_ids(input_ids)
            if self.state == JSONState.IN_PARAM_VALUE:
                best = self._best_value_token(logits)
                allowed = {best[0]} if best else set()
            else:
                allowed = self._get_allowed_token_ids()

            if not allowed:
                generated = "".join(
                    self.decoder.decode_token(i)
                    for i in input_ids[prompt_len:]
                )
                raise RuntimeError(
                    f"FSM dead end: state={self.state}, "
                    f"buffer={self.buffer!r}, used={self.used_params}, "
                    f"generated={generated!r}"
                )

            if self.state == JSONState.IN_PARAM_VALUE and best:
                next_id, kind = best
                if kind == "stop":
                    next_id = self._stop_token_id()
            else:
                next_id = int(max(allowed, key=lambda tid: logits[tid]))
            if next_id:
                self._update_state(self.decoder.decode_token(next_id))
                input_ids.append(next_id)

        generated_text = "".join(
            self.decoder.decode_token(i) for i in input_ids[prompt_len:])
        return self._get_final_output(generated_text)

    def _parse_generated_text(self, text: str) -> dict[str, Any]:
        """
        Parse and validate the generated text chunk into a Python dictionary.

        Raises
        ------
        RuntimeError
            If the generated text fails JSON parsing or lacks required
            structure.
        """
        body = text.strip()
        if body.startswith(("[", ",")):
            body = body[1:]
        if body.endswith("]"):
            body = body[:-1]
        try:
            record = json.loads(double_lone_backslashes(body))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Generated invalid JSON ({exc}): {text!r}") from exc
        if (not isinstance(record, dict)
                or not isinstance(record.get("parameters"), dict)):
            raise RuntimeError(
                f"Generated JSON has no parameters object: {text!r}")
        return record

    def _coerce_parameters(self, record: dict[str, Any]) -> None:
        """
        Coerce generated parameter values to match their declared schema types.
        """
        if not self.selected_tool:
            return
        params = record["parameters"]
        for name, schema in self.selected_tool.parameters.items():
            if name not in params:
                continue
            kind = schema.get("type")
            try:
                if kind == "number":
                    params[name] = float(params[name])
                elif kind == "integer":
                    params[name] = int(params[name])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Parameter '{name}' is not a valid {kind}: "
                    f"{params[name]!r}") from exc

    def _get_final_output(self, text: str) -> str:
        """
        Format the final validated prompt record into the batch JSON output
        array.
        """
        record = self._parse_generated_text(text)
        self._coerce_parameters(record)
        body = json.dumps(record, indent=4, ensure_ascii=False)
        body = "\n".join("    " + line for line in body.splitlines())
        return ("[\n" if self.is_first else ",\n") + body + (
            "\n]" if self.is_last else "")
