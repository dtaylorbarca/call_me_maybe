from .value_validation import DataTypeValidation
from .parser import FunctionTool, UserQuery
from .decoder import TokenDecoder
from llm_sdk import Small_LLM_Model
from enum import Enum, auto
import re
import json


class JSONState(Enum):
    START = auto()
    NEXT = auto()
    IN_NAME = auto()
    AFTER_NAME = auto()
    IN_PARAMS = auto()
    IN_PARAM_VALUE = auto()
    END = auto()
    FINISHED = auto()


AFTER_PROMPT_PATH = '",\n        "name": "'
AFTER_NAME_PATH = '        "parameters": {'


class FSMTokenizerIndex:
    def __init__(self, vocab: dict[str, int], decoder: TokenDecoder):
        self.decoder = decoder
        self.buckets: dict[str, list[int]] = {}
        self.all_token_ids = []
        self.safe_string_tokens = []

        decoder_texts = decoder.texts
        for token_id, text in enumerate(decoder_texts):
            if not text:
                continue
            self.all_token_ids.append(token_id)

            if '\n' not in text and '\r' not in text:
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
                 vocab: dict[str, int], tokenizer_index: FSMTokenizerIndex,
                 decoder: TokenDecoder, is_first: bool, is_last: bool) -> None:
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

        if is_first:
            self.start_path = '[\n    {\n        "prompt": "'
        else:
            self.start_path = ',\n    {\n        "prompt": "'

        if is_last:
            self.end_path = '\n    }\n]'
        else:
            self.end_path = '\n    }'



    def _get_allowed_name_tokens(self) -> set[int]:
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

        current_len = len(self.buffer)

        valid_chars = set()
        targets = []
        for param_name in self.selected_tool.parameters.keys():
            if param_name in self.used_params:
                continue

            target = f'"{param_name}": '
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

        get_tokens = self.token_index.get_tokens_starting_with
        if param_type == "number":
            candidate_chars = "-.0123456789"
            token_ids_to_check = self.token_index.get_tokens_starting_with(
                candidate_chars)

            if self.data_type_validator.is_valid_number(self.buffer):
                token_ids_to_check += get_tokens(",}\n ")

        elif param_type == "string":
            if self.buffer.endswith('"'):
                token_ids_to_check = self.token_index.safe_string_tokens
            else:
                token_ids_to_check = self.token_index.safe_string_tokens

        else:
            token_ids_to_check = self.token_index.all_token_ids

        decoder_texts = self.decoder.texts
        is_valid_string = self.data_type_validator.is_valid_json_string
        is_valid_number_prefix = (
            self.data_type_validator.is_valid_number_prefix)
        is_valid_number = self.data_type_validator.is_valid_number
        is_prefix_func = getattr(
            self.data_type_validator,
            "is_valid_json_string_prefix",
            lambda x: True
        )

        for token_id in token_ids_to_check:
            token_text = decoder_texts[token_id]
            if not token_text:
                continue

            candidate = self.buffer + token_text

            if param_type == "string":
                if is_valid_string(self.buffer) and self.buffer.endswith('"'):
                    if (is_last and (token_text.startswith("}") or
                                     token_text.strip() == "")):
                        allowed.add(token_id)
                    elif (not is_last and (token_text.startswith(",") or
                                           token_text.startswith(" ") or
                                           token_text.startswith("\n"))):
                        allowed.add(token_id)
                else:
                    if is_valid_string(candidate) or is_prefix_func(candidate):
                        allowed.add(token_id)

            elif param_type == "number":
                if is_valid_number_prefix(candidate):
                    allowed.add(token_id)
                elif is_valid_number(self.buffer):
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
            case JSONState.IN_NAME:
                return self._get_allowed_name_tokens()

            case JSONState.IN_PARAMS:
                return self._get_allowed_param_tokens()

            case JSONState.IN_PARAM_VALUE:
                return self._get_allowed_value_tokens()

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
                    if (self.selected_tool and not
                            self.selected_tool.parameters):
                        self.state = JSONState.END
                    elif self.selected_tool and self.selected_tool.parameters:
                        if len(self.used_params) == len(
                                self.selected_tool.parameters):
                            if "}" in self.buffer:
                                delim_indx = self.buffer.index("}")
                                self.buffer = self.buffer[delim_indx + 1:]
                                self.state = JSONState.END
                        else:
                            for param_name in (
                                    self.selected_tool.parameters.keys()):
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

                    param_info = self.selected_tool.parameters.get(
                        self.current_param, {})
                    if param_info.get("type") == "number":
                        if self.buffer.endswith('"'):
                            self.buffer = self.buffer.rstrip('"')

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

                case JSONState.FINISHED:
                    break

            if self.state == prev_state:
                break

    @staticmethod
    def _flatten_ids(encoded) -> list[int]:
        data = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        if data and isinstance(data[0], list):      # shape (1, n)
            data = data[0]
        return [int(v) for v in data]

    def _forced_text(self) -> str | None:
        """
        Return text with exactly one valid continuation and advanced state.
        """
        match self.state:
            case JSONState.START:
                text = self.start_path + self.user_query + AFTER_PROMPT_PATH
                self.state = JSONState.IN_NAME
            case JSONState.NEXT:
                text = self.start_path + self.user_query + AFTER_PROMPT_PATH
                self.state = JSONState.IN_NAME
            case JSONState.AFTER_NAME:
                text = AFTER_NAME_PATH
                self.state = JSONState.IN_PARAMS
            case JSONState.END:
                has_params = bool(self.selected_tool
                                  and self.selected_tool.parameters)
                text = self.end_path if has_params else "}" + self.end_path
                self.state = JSONState.FINISHED
            case _:
                return None
        self.buffer = ""
        return text

    def constrained_decoding(self, model: Small_LLM_Model,
                             initial_input_ids) -> str:
        input_ids = self._flatten_ids(initial_input_ids)
        prompt_len = len(input_ids)

        while self.state != JSONState.FINISHED:
            forced = self._forced_text()
            if forced is not None:
                input_ids.extend(self._flatten_ids(model.encode(forced)))
                continue

            allowed = self._get_allowed_token_ids()
            if not allowed:
                raise RuntimeError(
                    f"FSM reached dead end in state {self.state}"
                )

            logits = model.get_logits_from_input_ids(input_ids)
            next_id = int(max(allowed, key=lambda tid: logits[tid]))
            self._update_state(self.decoder.decode_token(next_id))
            input_ids.append(next_id)

        generated_text = "".join(
            self.decoder.decode_token(i) for i in input_ids[prompt_len:])
        return self._get_final_output(model, generated_text, initial_input_ids)

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
            if not generated_ids:
                generated_ids = output_ids

            clean_ids = [int(x.item() if hasattr(x, "item") else x)
                         for x in generated_ids]
            raw_text = model.decode(clean_ids)
        except Exception:
            try:
                clean_all = [int(x.item() if hasattr(x, "item") else x)
                             for x in output_ids]
                raw_text = model.decode(clean_all)
            except Exception:
                return str(output_ids)

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
