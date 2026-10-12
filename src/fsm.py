from .value_validation import DataTypeValidation
from .parser import FunctionTool, UserQuery
from .decoder import TokenDecoder
from llm_sdk import Small_LLM_Model
from enum import Enum, auto
from pathlib import Path
import re
import json


class JSONState(Enum):
    START = auto()
    PROMPT = auto()
    IN_NAME = auto()
    AFTER_NAME = auto()
    IN_PARAMS = auto()
    IN_PARAM_VALUE = auto()
    NEXT = auto()
    END = auto()
    FINISHED = auto()


START_PATH = '[\n    {\n        "prompt": "'
AFTER_PROMPT_PATH = '",\n        "name": "'
AFTER_NAME_PATH = '",\n        "parameters": {'
END_PATH = '}\n    }\n]'


class FSMTokenizerIndex:
    def __init__(self, vocab: dict[str, int], decoder: TokenDecoder):
        self.decoder = decoder
        self.buckets: dict[str, list[int]] = {}
        self.all_token_ids = []
        self.safe_string_tokens = []

        for token_str, token_id in vocab.items():
            text = decoder.decode_token(token_id)
            if not text:
                continue
            self.all_token_ids.append(token_id)

            if '"' not in text and '\\' not in text:
                self.safe_string_tokens.append(token_id)

            first_char = text[0]
            if first_char not in self.buckets:
                self.buckets[first_char] = []
            self.buckets[first_char].append(token_id)

    def get_tokens_starting_with(self, chars: str) -> list[int]:
        result = []
        for c in chars:
            if c in self.buckets:
                result.extend(self.buckets[c])
        return result


class JSONStateMachine:
    def __init__(self, tools: list[FunctionTool], prompts: list[UserQuery],
                 vocab_path: str) -> None:
        with open(Path(vocab_path), encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)
        self.tools: list[FunctionTool] = tools
        self.prompts: list[UserQuery] = prompts
        self.state: JSONState = JSONState.START
        self.selected_tool: FunctionTool | None = None
        self.buffer = ""
        self.user_query: str = prompts[0].prompt
        self.decoder = TokenDecoder(self.vocab)
        self.used_params: list[str] = []
        self.finished_prompts: list[str] = []
        self.data_type_validator: DataTypeValidation = DataTypeValidation()
        self.token_index: FSMTokenizerIndex = FSMTokenizerIndex(
            self.vocab, self.decoder)

    def _get_allowed_determ_tokens(self, fixed_path: str) -> set[int]:
        allowed = set()
        current_len = len(self.buffer)
        remaining = fixed_path[current_len:]

        if not remaining:
            return allowed

        token_ids_to_check = self.token_index.get_tokens_starting_with(
            remaining[0])

        for token_id in token_ids_to_check:
            token_text = self.decoder.decode_token(token_id)
            if not token_text:
                continue

            current_text = self.buffer + token_text
            if remaining.startswith(token_text) or (
                    token_text.startswith(remaining) and
                    fixed_path.startswith(current_text)):
                allowed.add(token_id)

        return allowed

    def _get_allowed_name_tokens(self) -> set[int]:
        allowed = set()
        current_len = len(self.buffer)

        valid_chars = set()
        targets = []
        for tool in self.tools:
            target = tool.name + '"'
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
                if (remaining_target.startswith(token_text) or
                        token_text.startswith(remaining_target)):
                    allowed.add(token_id)
                    break

        return allowed

    def _get_allowed_param_tokens(self) -> set[int]:
        allowed = set()
        if not self.selected_tool or not self.selected_tool.parameters:
            return allowed

        is_all_used = len(self.used_params) == len(
            self.selected_tool.parameters)
        if is_all_used:
            token_ids_to_check = self.token_index.get_tokens_starting_with("}")
            for token_id in token_ids_to_check:
                token_text = self.decoder.decode_token(token_id)
                if token_text.startswith("}"):
                    allowed.add(token_id)

        current_len = len(self.buffer)

        valid_chars = set()
        targets = []
        for param_name in self.selected_tool.parameters.keys():
            if param_name in self.used_params:
                continue

            target = f'"{param_name}": '
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
                if (remaining_target.startswith(token_text) or
                        token_text.startswith(remaining_target)):
                    allowed.add(token_id)
                    break

        return allowed

    def _get_allowed_value_tokens(self) -> set[int]:
        allowed = set()
        if not self.selected_tool or not self.current_param:
            return allowed

        param_info = self.selected_tool.parameters.get(self.current_param, {})
        param_type = param_info.get("type")
        is_last = len(self.used_params) == len(self.selected_tool.parameters)

        if param_type == "number":
            candidate_chars = "-.0123456789"
            token_ids_to_check = self.token_index.get_tokens_starting_with(
                candidate_chars)

            if self.data_type_validator.is_valid_number(self.buffer):
                token_ids_to_check += self.token_index.get_tokens_starting_with(
                    ",}\n ")

        elif param_type == "string":
            if self.buffer.endswith('"'):
                token_ids_to_check = self.token_index.safe_string_tokens
            else:
                token_ids_to_check = self.token_index.get_tokens_starting_with(",}\n ")

        else:
            token_ids_to_check = self.token_index.all_token_ids

        for token_id in token_ids_to_check:
            token_text = self.decoder.decode_token(token_id)
            if not token_text:
                continue

            candidate = self.buffer + token_text

            if param_type == "string":
                if (self.data_type_validator.is_valid_json_string(self.buffer)
                        and self.buffer.endswith('"')):
                    if (is_last and (token_text.startswith("}") or
                                     token_text.strip() == "")):
                        allowed.add(token_id)
                    elif (not is_last and (token_text.startswith(",") or
                                           token_text.startswith(" ") or
                                           token_text.startswith("\n"))):
                        allowed.add(token_id)
                else:
                    is_prefix_func = getattr(
                        self.data_type_validator,
                        "is_valid_json_string_prefix",
                        lambda x: True
                    )
                    if (self.data_type_validator.is_valid_json_string(
                        candidate) or
                            is_prefix_func(candidate)):
                        allowed.add(token_id)

            elif param_type == "number":
                if self.data_type_validator.is_valid_number_prefix(candidate):
                    allowed.add(token_id)
                elif self.data_type_validator.is_valid_number(self.buffer):
                    if (is_last and (token_text.startswith("}") or
                                     token_text.strip() == "")):
                        allowed.add(token_id)
                    elif (not is_last and (token_text.startswith(",") or
                                           token_text.startswith(" ") or
                                           token_text.startswith("\n"))):
                        allowed.add(token_id)

            elif param_type == "boolean":
                target_vals = ["true", "false"]
                matched_bool = False

                for t in target_vals:
                    full_target = f"{t}{'}' if is_last else ', '}"
                    if (
                        full_target.startswith(candidate) or
                            candidate.startswith(full_target[:len(candidate)])
                    ):
                        allowed.add(token_id)
                        matched_bool = True
                        break

                if not matched_bool:
                    for t in target_vals:
                        if self.buffer == t:
                            if (is_last and (token_text.startswith("}") or
                                             token_text.strip() == "")):
                                allowed.add(token_id)
                            elif (not is_last and (
                                    token_text.startswith(",") or
                                    token_text.startswith(" ") or
                                    token_text.startswith("\n"))):
                                allowed.add(token_id)

        return allowed

    def _get_allowed_token_ids(self) -> set[int]:
        match self.state:
            case JSONState.START:
                return self._get_allowed_determ_tokens(
                    START_PATH + self.user_query + AFTER_PROMPT_PATH
                )

            case JSONState.PROMPT:
                return self._get_allowed_determ_tokens(
                    self.user_query + AFTER_PROMPT_PATH
                )

            case JSONState.IN_NAME:
                return self._get_allowed_name_tokens()

            case JSONState.AFTER_NAME:
                return self._get_allowed_determ_tokens(AFTER_NAME_PATH)

            case JSONState.IN_PARAMS:
                return self._get_allowed_param_tokens()

            case JSONState.IN_PARAM_VALUE:
                return self._get_allowed_value_tokens()

            case JSONState.END:
                return self._get_allowed_determ_tokens(END_PATH)

            case _:
                return set()

    def _build_next_path(self, next_prompt_text: str) -> str:
        return (
            f'}},\n<|im_end|>\n<|im_start|>user\n{next_prompt_text}<|im_end|>'
            '\n  {\n\t\t"prompt": "'
        )

    def _update_state(self, token_str: str) -> None:
        self.buffer += token_str
        while True:
            prev_state = self.state
            match self.state:
                case JSONState.START:
                    target_path = (
                        START_PATH + self.user_query + AFTER_PROMPT_PATH
                    )
                    if len(self.buffer) >= len(target_path):
                        self.buffer = self.buffer[len(target_path):]
                        self.state = JSONState.IN_NAME

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
                                f"Generated unknown tool name: '{extracted_name}'")

                case JSONState.AFTER_NAME:
                    if len(self.buffer) >= len(AFTER_NAME_PATH):
                        self.buffer = self.buffer[len(AFTER_NAME_PATH):]
                        self.state = JSONState.IN_PARAMS

                case JSONState.IN_PARAMS:
                    if self.selected_tool and not self.selected_tool.parameters:
                        self.state = JSONState.END
                    elif self.selected_tool and self.selected_tool.parameters:
                        if len(self.used_params) == len(self.selected_tool.parameters):
                            if "}" in self.buffer:
                                delim_indx = self.buffer.index("}")
                                self.buffer = self.buffer[delim_indx + 1:]
                                self.state = JSONState.END
                        else:
                            for param_name in self.selected_tool.parameters.keys():
                                if param_name not in self.used_params:
                                    expected_prefix = f'"{param_name}": '
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

                    has_comma = "," in self.buffer
                    has_bracket = "}" in self.buffer

                    if has_comma or has_bracket:
                        if has_comma:
                            delim_indx = self.buffer.index(",")
                            self.buffer = self.buffer[delim_indx + 2:]
                        else:
                            delim_indx = self.buffer.index("}")
                            self.buffer = self.buffer[delim_indx + 1:]

                        self.current_param = None
                        self.state = JSONState.IN_PARAMS

                case JSONState.END:
                    if len(self.buffer) >= len(END_PATH):
                        self.state = JSONState.FINISHED

                case JSONState.FINISHED:
                    break

            if self.state == prev_state:
                break

    def constrained_decoding(self, model: Small_LLM_Model, initial_input_ids) -> str:
        input_ids = []
        for x in initial_input_ids:
            if hasattr(x, "tolist"):
                flattened = x.tolist()
                if isinstance(flattened, list):
                    input_ids.extend([int(val) for val in flattened])
                else:
                    input_ids.append(int(flattened))
            elif hasattr(x, "item"):
                input_ids.append(int(x.item()))
            else:
                input_ids.append(int(x))

        self.state = JSONState.START

        while self.state != JSONState.FINISHED:
            allowed_tokens = self._get_allowed_token_ids()
            if not allowed_tokens:
                raise RuntimeError(
                    f"FSM reached dead end in state {self.state}")

            logits = model.get_logits_from_input_ids(input_ids)
            next_token_id = int(
                max(allowed_tokens, key=lambda tid: logits[tid]))

            token_str = self.decoder.decode_token(next_token_id)
            self._update_state(token_str)
            input_ids.append(next_token_id)

        return self._get_final_output(model, input_ids, initial_input_ids)

    def _repair_json(self, text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        t = text.replace('\\"', '"')
        t = re.sub(r'\\[nrt]', ' ', t)
        t = t.replace("\\", "")
        return json.loads(t)

    def _format_calls(self, calls: list) -> str:
        items = []
        for call in calls:
            fields = ",\n".join(
                f"        {json.dumps(key)}: {json.dumps(value)}"
                for key, value in call.items()
            )
            items.append("    {\n" + fields + "\n    }")
        return "[\n" + ",\n".join(items) + "\n]"

    def _get_final_output(self, model, output_ids, initial_input_ids) -> str:
        try:
            generated_ids = output_ids[len(initial_input_ids):]
            raw_text = model.decode(generated_ids)
        except Exception:
            raw_text = model.decode(output_ids)

        for marker in ["<|im_start|>assistant", "assistant\n"]:
            if marker in raw_text:
                raw_text = raw_text.split(marker)[-1]

        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        json_text = match.group(0) if match else raw_text

        try:
            parsed = self._repair_json(json_text)
        except json.JSONDecodeError:
            return json_text.strip()

        if isinstance(parsed, dict):
            parsed = [parsed]

        return self._format_calls(parsed)
