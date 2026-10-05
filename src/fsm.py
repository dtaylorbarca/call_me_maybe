from enum import Enum, auto
from .parser import FunctionTool
from .syntax import JSONSyntaxTokenIDs
from ..llm_sdk.llm_sdk import Small_LLM_Model


class JSONState(Enum):
    START = auto()
    EXPECT_OBJECT_START = auto()
    EXPECT_KEY = auto()
    IN_KEY = auto()
    END_KEY = auto()
    EXPECT_COLON = auto()
    IN_VALUE = auto()
    NEXT_OR_CLOSE = auto()
    EXPECT_OBJECT_END = auto()
    END = auto()


class JSONStateMachine:
    def __init__(self, tools: list[FunctionTool],
                 syntax: JSONSyntaxTokenIDs) -> None:
        self.tools = tools
        self.syntax = syntax
        self.state: JSONState = JSONState.START
        self.emitted_ids: set[str] = set()
        self.current_key: str | None = None
        self.selected_tool: FunctionTool | None = None
        self.buffer: str = ""

    def _get_allowed_start_tokens(self) -> set[int]:
        fixed_prefix = '[\n\t{\n\t\t"prompt": "'
        current_len = len(self.buffer)
        remaining = fixed_prefix[current_len:]
        allowed = set()

        for token_text, token_id in self.syntax.vocab.items():
            if (remaining.startswith(token_text) or
                    token_text.startswith(remaining)):
                allowed.add(token_id)
        return allowed

    def get_allowed_token_ids(self) -> set[int]:
        match self.state:
            case JSONState.START:
                return self._get_allowed_start_tokens()

            case JSONState.EXPECT_OBJECT_START:
                return {self.syntax.curly_open}

            case JSONState.EXPECT_KEY | JSONState.END_KEY:
                return {self.syntax.double_quotes}

            case JSONState.IN_KEY:
                return self._get_allowed_key_tokens()

            case JSONState.EXPECT_COLON:
                return {self.syntax.colon}

            case JSONState.EXPECT_OBJECT_END:
                return {self.syntax.curly_close}

            case JSONState.NEXT_OR_CLOSE:
                if (self.selected_tool and
                    self.emitted_ids.issuperset(
                        self.selected_tool.get_params())):
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
                        self.buffer = self.buffer[self.buffer.index("{"):]
                        self.state = JSONState.EXPECT_KEY

                
            if self.state == prev_state:
                break

    def get_final_output(self, model: Small_LLM_Model, output_ids: list[int],
                         initial_input_ids: list[int]) -> str:

        output_ids = output_ids[len(initial_input_ids):]
        return model.decode(output_ids)
