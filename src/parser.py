import json
from typing_extensions import Self
from typing import Dict, List
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ConfigDict,
    field_validator
)


class FunctionTool(BaseModel):
    """Schema representing a single callable tool available to the model."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str
    parameters: Dict[str, dict[str, str]] = Field(default_factory=dict)
    returns: Dict[str, str] = Field(default_factory=dict)

    def to_system_prompt_snippet(self) -> str:
        """
        Converts the tool definition into a clean formatted string for the
        system prompt.
        """
        props = self.parameters

        lines = []
        for prop_name, prop_data in props.items():
            prop_type = prop_data.get("type", "any")
            lines.append(f"  - {prop_name} ({prop_type})")

        return_type = self.returns.get("type", "any")
        props_str = "\n".join(lines) if lines else "  None"
        return (f"Tool: {self.name}\nDescription: {self.description}\n"
                f"Parameters:\n{props_str}\nReturns:\n  - {return_type}")


class UserQuery(BaseModel):
    """Schema for evaluating function calling against prompts."""
    prompt: str

    @field_validator("prompt")
    def escape_validity(self) -> Self:
        valid_chr = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
        escaped = False

        for c in self.prompt:
            if escaped:
                if c not in valid_chr:
                    raise ValueError(
                        f"Invalid escape sequence '\\{c}' in prompt "
                        f"'{self.prompt}'. JSON only permits \\\", \\\\, "
                        "\\/, \\b, \\f, \\n, \\r, \\t, or \\uXXXX."
                    )
                escaped = False
            elif c == "\\":
                escaped = True
        if escaped:
            raise ValueError("Prompt ends with a dangling backslash; "
                             f"'{self.prompt}'")
        return self


def load_functions_definition(file_path: str) -> List[FunctionTool]:
    """Loads and validates function definitions from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tools = [FunctionTool(**tool) for tool in data]
    except (
        json.decoder.JSONDecodeError,
        OSError,
        ValidationError,
        UnicodeDecodeError,
    ) as e:
        print(e)
        exit(1)
    return tools


def load_user_queries(file_path: str) -> List[UserQuery]:
    """Loads and validates test cases from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tests = [UserQuery(**test) for test in data]
    except (
        json.decoder.JSONDecodeError,
        OSError,
        ValidationError,
        UnicodeDecodeError,
    ) as e:
        print(e)
        exit(1)
    return tests
