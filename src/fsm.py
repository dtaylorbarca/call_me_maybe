from enum import Enum, auto


class JSONState(Enum):
    START = auto()
    EXPECT_KEY = auto()
    IN_KEY = auto()
    EXPECT_COLON = auto()
    IN_VALUE = auto()
    NEXT_OR_CLOSE = auto()
    END = auto()
