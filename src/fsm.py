from enum import Enum, auto
from .parser import FunctionTool
from .syntax import JSONSyntaxTokenIDs
from ..llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any


class JSONState(Enum):
    START = auto()
    EXPECT_OBJECT_START = auto()
    EXPECT_KEY = auto()
    IN_KEY = auto()
    END_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    IN_VALUE = auto()
    NEXT_OR_CLOSE = auto()
    END = auto()


class JSONKeys(str, Enum):
    PROMPT = "prompt"
    NAME = "name"
    PARAMETERS = "parameters"


class JSONStateMachine:
    def __init__(self, tools: list[FunctionTool],
                 syntax: JSONSyntaxTokenIDs) -> None:
        self.tools: dict[str, Any] = {}
        for tool in tools:
            self.tools.update({tool.name: tool.parameters})
        self.syntax = syntax
        self.state: JSONState = JSONState.START
        self.emitted_ids: set[str] = set()
        self.selected_tool: FunctionTool | None = None
        self.buffer: str = ""
        self.key = JSONKeys.NAME
        self.user_query: str = ""

    def _get_allowed_start_tokens(self) -> set[int]:
        fixed_prefix = '[\n\t{\n\t\t"prompt": "'
        current_len = len(self.buffer)
        remaining = fixed_prefix[current_len:]
        allowed = set()

        for token_text, token_id in self.syntax.vocab.items():
            if not token_text:
                continue
            if (remaining.startswith(token_text) or
                    token_text.startswith(remaining)):
                allowed.add(token_id)
        return allowed

    def _get_allowed_key_tokens(self) -> set[int]:
        allowed = set()
        current_len = len(self.buffer)
        target_key = self.key

        remaining = target_key[current_len:]

        for token_text, token_id in self.syntax.vocab.items():
            if not token_text:
                continue
            if (remaining.startswith(token_text) or
                    token_text.startswith(remaining)):
                allowed.add(token_id)

        return allowed

    def _get_allowed_name_tokens(self) -> set[int]:
        allowed = set()

        current_len = len(self.buffer)
        for token_text, token_id in self.syntax.vocab.items():
            if not token_text:
                continue
            for tool in self.tools:
                if (tool[current_len:].startswith(token_text) or
                        token_text.startswith(tool[current_len:])):
                    allowed.add(token_id)
        return allowed

    def _get_allowed_value_tokens(self) -> set[int]:
        current_len = len(self.buffer)
        allowed = set()

        match self.key:
            case JSONKeys.PROMPT:
                remaining = self.user_query[current_len:]
                for token_text, token_id in self.syntax.vocab.items():
                    if not token_text:
                        continue
                    if (remaining.startswith(token_text) or
                            token_text.startswith(remaining)):
                        allowed.add(token_id)
            case JSONKeys.NAME:
                allowed = self._get_allowed_name_tokens()

            case JSONKeys.PARAMETERS:
                if self.selected_tool:


        return allowed

    def get_allowed_token_ids(self) -> set[int]:
        match self.state:
            case JSONState.START:
                return self._get_allowed_start_tokens()

            case JSONState.EXPECT_OBJECT_START:
                return {self.syntax.curly_open}

            case (JSONState.EXPECT_KEY | JSONState.END_KEY |
                  JSONState.EXPECT_VALUE):
                return {self.syntax.double_quotes}

            case JSONState.IN_KEY:
                return self._get_allowed_key_tokens()

            case JSONState.EXPECT_COLON:
                return {self.syntax.colon}

            case JSONState.IN_VALUE:
                return self._get_allowed_value_tokens()

            case JSONState.NEXT_OR_CLOSE:
                if self.key == JSONKeys.PARAMETERS:
                    return {self.syntax.curly_close}
                else:
                    return {self.syntax.comma}

            case JSONState.END:
                return {self.syntax.square_close}

            case _:
                return set()

    def update_state(self, token_str: str) -> None:
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
                        self.state = JSONState.IN_VALUE

                case JSONState.EXPECT_OBJECT_START:
                    if "{" in self.buffer:
                        self.buffer = self.buffer[self.buffer.index("{") + 1:]
                        self.state = JSONState.EXPECT_KEY

                case JSONState.EXPECT_KEY:
                    if '"' in self.buffer:
                        self.buffer = self.buffer[self.buffer.index('"') + 1:]
                        self.state = JSONState.IN_KEY

                case JSONState.IN_KEY:
                    target_key = self.key
                    if len(self.buffer) >= len(target_key):
                        self.buffer = self.buffer[len(target_key):]
                        self.state = JSONState.END_KEY

                case JSONState.END_KEY:
                    if '"' in self.buffer:
                        self.buffer = self.buffer[self.buffer.index('"') + 1:]
                        self.state = JSONState.EXPECT_COLON

                case JSONState.EXPECT_COLON:
                    if ":" in self.buffer:
                        self.buffer = self.buffer[self.buffer.index(": ") + 1:]
                        self.state = JSONState.EXPECT_VALUE

                case JSONState.EXPECT_VALUE:
                    if '"' in self.buffer:
                        self.buffer = self.buffer[self.buffer.index('"') + 1:]
                        self.state = JSONState.NEXT_OR_CLOSE

                case JSONState.IN_KEY:
                    pass

                case JSONState.NEXT_OR_CLOSE:
                    if self.key == JSONKeys.PARAMETERS:
                        if "}" in self.buffer:
                            self.buffer = self.buffer[
                                self.buffer.index("}") + 1:
                            ]

                    else:
                        if "," in self.buffer:
                            self.buffer = self.buffer[
                                self.buffer.index(",") + 1:
                            ]
                            self.state = JSONState.EXPECT_OBJECT_START

            if self.state == prev_state:
                break

    def get_final_output(self, model: Small_LLM_Model, output_ids: list[int],
                         initial_input_ids: list[int]) -> str:

        output_ids = output_ids[len(initial_input_ids):]
        return model.decode(output_ids)
