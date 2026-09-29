from enum import Enum, auto
from .parser import FunctionTool
from .syntax import JSONSyntaxTokenIDs


class JSONState(Enum):
    START = auto()
    EXPECT_KEY = auto()
    IN_KEY = auto()
    EXPECT_COLON = auto()
    IN_VALUE = auto()
    NEXT_OR_CLOSE = auto()
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

    def get_allowed_token_ids(self) -> set[int]:
        match self.state:
            case JSONState.START:
                return {self.syntax.square_open}
            case JSONState.EXPECT_KEY:
                return {self.syntax.curly_open}
            case JSONState.EXPECT_COLON:
                return {self.syntax.colon}
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
                return {0}

    def update_state(self, token_str: str) -> None:
        self.buffer += token_str
        while True:
            prev_state = self.state
            match self.state:
                case JSONState.START:
                    if "[\n" in self.buffer:
                        self.buffer = self.buffer[self.buffer.index("[") + 1:]
                        self.state = JSONState.EXPECT_KEY
            if self.state == prev_state:
                break

