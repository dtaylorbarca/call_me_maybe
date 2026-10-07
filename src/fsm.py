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


class JSONKeys(str, Enum):
    PROMPT = "prompt"
    NAME = "name"
    PARAMETERS = "parameters"


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
        self.key = JSONKeys.NAME
        self.user_query = ""
        self.decoder = TokenDecoder(self.vocab)

    def _get_allowed_start_tokens(self) -> set[int]:
        fixed_prefix = '[\n\t{\n\t\t"prompt": "' + self.user_query
        fixed_prefix = fixed_prefix + ',\n\t\t\t"name": "'
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

    def _get_allowed_key_tokens(self) -> set[int]:
        allowed = set()
        current_len = len(self.buffer)
        target_key_full_path = self.key + '": "'
        remaining = target_key_full_path[current_len:]

        for token_text, token_id in self.vocab.items():
            if not token_text:
                continue

            if (remaining.startswith(token_text)):
                allowed.add(token_id)

        return allowed

    def _get_allowed_name_tokens(self) -> set[int]:
        allowed = set()
        current_len = len(self.buffer)

        for token_text, token_id in self.vocab.items():
            if not token_text:
                continue

            for tool, _ in self.tools.items():
                if self.user_query == self.prompts[-1]:
                    tool_path = tool + '",\n\t\t}\n]'
                else:
                    tool_path = tool + '",\n\t\t},\n\t\t{\n\t\t\t"prompt": "'

                if (tool[current_len:].startswith(token_text) or
                        token_text.startswith(tool[current_len:])):
                    allowed.add(token_id)

        return allowed

    def _get_allowed_after_name_tokens(self) -> set[int]:
        allowed = set()
        current_len = len(self.buffer)
        if self.selected_tool:
            fixed_path = '",\n\t\t\t"parameters": {"' + self.tool.

        for token_text, token_id in self.vocab.items():
            if not token_text:
                continue


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
        fixed_prefix = '[\n\t{\n\t\t"prompt": "'
        while True:
            prev_state = self.state
            match self.state:
                case JSONState.START:
                    if not fixed_prefix.startswith(self.buffer):
                        raise ValueError("Invalid starting sequence: "
                                         f"'{self.buffer}'")
                    if len(self.buffer) >= len(fixed_prefix):
                        self.buffer = self.buffer[len(fixed_prefix):]
                        self.state = JSONState.IN_NAME

                case JSONState.IN_NAME:
                    return

                case JSONState.

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
