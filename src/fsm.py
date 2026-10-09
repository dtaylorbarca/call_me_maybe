from ..llm_sdk.llm_sdk import Small_LLM_Model
from .parser import FunctionTool, UserQuery
from .decoder import TokenDecoder
from enum import Enum, auto
import json
from value_validation import DataTypeValidation


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


START_PATH = '[\n\t\t{\n\t\t\t"prompt": "'
AFTER_PROMPT_PATH = ',\n\t\t\t"name": "'
AFTER_NAME_PATH = '",\n\t\t\t"parameters": {'
NEXT_PATH = '}\n\t\t},\n\t\t{\n\t\t\t"prompt": "'
END_PATH = '}\n\t\t}\n]'


class JSONStateMachine:
    def __init__(self, tools: list[FunctionTool], prompts: list[UserQuery],
                 vocab_path: str) -> None:
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)
        self.tools: list[FunctionTool] = tools
        self.prompts: list[str] = [prompt.prompt for prompt in prompts]
        self.state: JSONState = JSONState.START
        self.selected_tool: FunctionTool | None = None
        self.buffer = ""
        self.user_query = ""
        self.decoder = TokenDecoder(self.vocab)
        self.used_params: list[str] = []
        self.finished_prompts: list[str] = []
        self.data_type_validator: DataTypeValidation = DataTypeValidation()

    def _get_allowed_determ_tokens(self, fixed_path: str) -> set[int]:
        allowed = set()
        current_len = len(self.buffer)
        remaining = fixed_path[current_len:]

        for token_text, token_id in self.vocab.items():
            if not token_text:
                continue

            if (remaining.startswith(token_text) or
                    token_text.startswith(remaining)):
                allowed.add(token_id)

        return allowed

    def _get_allowed_name_tokens(self) -> set[int]:
        allowed = set()
        current_len = len(self.buffer)

        for token_text, token_id in self.vocab.items():
            if not token_text:
                continue

            for tool in self.tools:
                target = tool.name + '"'
                remaining_target = target[current_len:]

                if (remaining_target.startswith(token_text) or
                        token_text.startswith(remaining_target)):
                    allowed.add(token_id)
                    break

        return allowed

    def _get_allowed_param_tokens(self) -> set[int]:
        allowed = set()
        if not self.selected_tool or not self.selected_tool.parameters:
            return allowed

        current_len = len(self.buffer)
        for param_name in self.selected_tool.parameters.keys():
            if param_name in self.used_params:
                continue

            target = f'"{param_name}": '
            remaining_target = target[current_len:]

            for token_text, token_id in self.vocab.items():
                if not token_text:
                    continue

                if (remaining_target.startswith(token_text) or
                        token_text.startswith(remaining_target)):
                    allowed.add(token_id)

        return allowed

    def _get_allowed_value_tokens(self) -> set[int]:
        allowed = set()
        if not self.selected_tool or not self.current_param:
            return allowed
        param_type = self.selected_tool.parameters[self.current_param].get(
            "type")
        is_last = len(self.used_params) == len(self.selected_tool.parameters)
        suffix = "}" if is_last else ", "
        string_suffix = '"}' if is_last else '", '
        for token_text, token_id in self.vocab.items():
            if not token_text:
                continue

            if param_type == "string":
                candidate = self.buffer + token_text

                if string_suffix not in candidate:
                    if self.data_type_validator.is_valid_json_string(
                            candidate):
                        allowed.add(token_id)

                elif candidate.endswith(string_suffix):
                    body = candidate[:-len(string_suffix)]
                    if self.data_type_validator.is_valid_json_string(body):
                        allowed.add(token_id)

            elif param_type == "number":
                candidate = self.buffer + token_text

                if self.data_type_validator.is_valid_number_prefix(candidate):
                    allowed.add(token_id)

                elif candidate.endswith(suffix):
                    number_part = candidate[:-len(suffix)]
                    if self.data_type_validator.is_valid_number(number_part):
                        allowed.add(token_id)

            elif param_type == "boolean":
                targets = [f"true{suffix}", f"false{suffix}"]
                current_len = len(self.buffer)

                for target in targets:
                    remaining = target[current_len:]
                    if (remaining.startswith(token_text) or
                            token_text.startswith(remaining)):
                        allowed.add(token_id)
        return allowed

    def _get_allowed_token_ids(self) -> set[int]:
        match self.state:
            case JSONState.START:
                return self._get_allowed_determ_tokens(START_PATH)

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

            case JSONState.NEXT:
                return self._get_allowed_determ_tokens(NEXT_PATH)

            case JSONState.END:
                return self._get_allowed_determ_tokens(END_PATH)

            case _:
                return set()

    def _update_state(self, token_str: str) -> None:
        self.buffer += token_str
        while True:
            prev_state = self.state
            match self.state:
                case JSONState.START:
                    if not START_PATH.startswith(self.buffer):
                        raise ValueError("Invalid starting sequence: "
                                         f"'{self.buffer}'")
                    if len(self.buffer) >= len(START_PATH):
                        self.buffer = self.buffer[len(START_PATH):]
                        self.state = JSONState.IN_NAME
                        self.temp_tools = self.tools

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
                                "Generated unknown tool name:"
                                f"'{extracted_name}'"
                            )

                case JSONState.AFTER_NAME:
                    if len(self.buffer) >= len(AFTER_NAME_PATH):
                        self.buffer = self.buffer[len(AFTER_NAME_PATH):]
                        self.state = JSONState.IN_PARAMS

                case JSONState.IN_PARAMS:
                    if (self.selected_tool and not
                            self.selected_tool.parameters):
                        self.finished_prompts.append(self.user_query)
                        if len(self.finished_prompts) < len(self.prompts):
                            self.state = JSONState.NEXT
                        else:
                            self.state = JSONState.END
                    elif self.selected_tool and self.selected_tool.parameters:
                        for param_name in self.selected_tool.parameters.keys():
                            if param_name not in self.used_params:
                                expected_prefix = f'"{param_name}": '
                                if self.buffer.startswith(expected_prefix):
                                    self.current_param = param_name
                                    self.used_params.append(param_name)
                                    self.buffer = self.buffer[
                                        len(expected_prefix):]
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

                        if (len(self.used_params) <
                                len(self.selected_tool.parameters)):
                            self.state = JSONState.IN_PARAMS
                        else:
                            if len(self.finished_prompts) < len(self.prompts):
                                self.state = JSONState.NEXT
                            else:
                                self.state = JSONState.END

                case JSONState.NEXT:
                    if len(self.buffer) >= len(NEXT_PATH):
                        self.buffer = self.buffer[len(NEXT_PATH):]
                        self.used_params = []
                        self.selected_tool = None
                        self.state = JSONState.PROMPT

                case JSONState.END:
                    if len(self.buffer) >= len(END_PATH):
                        self.state = JSONState.FINISHED

                case JSONState.FINISHED:
                    break

            if self.state == prev_state:
                break

    def constrained_decoding(self, model: Small_LLM_Model,
                             initial_input_ids: list[int], user_query: str
                             ) -> str:
        input_ids = initial_input_ids
        self.user_query = user_query
        self.state = JSONState.START
        while (self.state != JSONState.NEXT or self.state != JSONState.END or
               JSONState.FINISHED):
            allowed_tokens = self._get_allowed_token_ids()
            if not allowed_tokens:
                raise RuntimeError(
                    "FSM reached a dead end with no allowed tokens")
            logits = model.get_logits_from_input_ids(input_ids)
            next_token_id = max(
                allowed_tokens, key=lambda token_id: logits[token_id]
            )
            token_str = self.decoder.decode_token(next_token_id)
            self._update_state(token_str)
            input_ids.append(next_token_id)

        return self.get_final_output(model, input_ids, initial_input_ids)

    def get_final_output(self, model: Small_LLM_Model, output_ids: list[int],
                         initial_input_ids: list[int]) -> str:

        output_ids = output_ids[len(initial_input_ids):]
        return model.decode(output_ids)
