from ..llm_sdk.llm_sdk import Small_LLM_Model
from .parser import FunctionTool, UserQuery
from .decoder import TokenDecoder
from enum import Enum, auto
from typing import Any
import json


class JSONState(Enum):
    START = auto()
    PROMPT = auto()
    IN_NAME = auto()
    AFTER_NAME = auto()
    IN_PARAMS = auto()
    NEXT = auto()
    END = auto()


START_PATH = '[\n\t\t{\n\t\t\t"prompt": "'
AFTER_PROMPT_PATH = ',\n\t\t\t"name": "'
AFTER_NAME_PATH = '",\n\t\t\t"parameters": {'
NEXT_PATH = '}\n\t\t},\n\t\t{\n\t\t\t"prompt": "'
END_PATH = ''


class JSONStateMachine:
    def __init__(self, tools: list[FunctionTool], prompts: list[UserQuery],
                 vocab_path: str) -> None:
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)
        self.tools: dict[str, Any] = {}
        for tool in tools:
            self.tools.update({tool.name: tool.parameters})
        self.prompts: list[str] = [prompt.prompt for prompt in prompts]
        self.state: JSONState = JSONState.START
        self.emitted_ids: set[str] = set()
        self.selected_tool: FunctionTool | None = None
        self.buffer = ""
        self.user_query = ""
        self.decoder = TokenDecoder(self.vocab)

    def _get_allowed_start_tokens(self) -> set[int]:
        fixed_prefix = START_PATH + self.user_query + AFTER_PROMPT_PATH
        current_len = len(self.buffer)
        remaining = fixed_prefix[current_len:]
        allowed = set()

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

            for tool_name in self.tools.keys():
                target = tool_name + '"'
                remaining_target = target[current_len:]

                if remaining_target.startswith(token_text) or token_text.startswith(remaining_target):
                    allowed.add(token_id)
                    break

        return allowed

    def _get_allowed_after_name_tokens(self) -> set[int]:
        allowed = set()
        current_len = len(self.buffer)
        remaining = AFTER_NAME_PATH[current_len:]

        for token_text, token_id in self.vocab.items():
            if not token_text:
                continue

            if (remaining.startswith(token_text) or
                    token_text.startswith(remaining)):
                allowed.add(token_id)

        return allowed

    def _get_allowed_param_tokens(self) -> set[int]:
        allowed = set()

        current_len = len(self.buffer)
        if self.selected_tool:
            fixed_path = ''
            for token_text, token_id in self.vocab.items():
                if not token_text:
                    continue

        return allowed

    def _get_allowed_token_ids(self) -> set[int]:
        match self.state:
            case JSONState.START:
                return self._get_allowed_start_tokens()

            case JSONState.IN_NAME:
                return self._get_allowed_name_tokens()

            case JSONState.AFTER_NAME:
                return self._get_allowed_after_name_tokens()

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
                        extracted_name = self.buffer[]

            if self.state == prev_state:
                break

    def constrained_decoding(self, model: Small_LLM_Model,
                             initial_input_ids: list[int], user_query: str
                             ) -> str:
        input_ids = initial_input_ids
        self.user_query = user_query
        self.state = JSONState.START
        while self.state != JSONState.END:
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
