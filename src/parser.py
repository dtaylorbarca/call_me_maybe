import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError


class FunctionTool(BaseModel):
    """Schema representing a single callable tool available to the model."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    returns: Dict[str, Any] = Field(default_factory=dict)

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


class FunctionCallOutput(BaseModel):
    """Expected function call output schema."""
    name: str
    parameters: Dict[str, Any]


class TestCase(BaseModel):
    """Schema for evaluating function calling against prompts."""
    prompt: str
    expected: Optional[FunctionCallOutput] = None


def load_functions_definition(file_path: str) -> List[FunctionTool]:
    """Loads and validates function definitions from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(e)
            exit(1)
    try:
        tools = [FunctionTool(**tool) for tool in data]
    except ValidationError as e:
        print(e)
        exit(1)
    return tools


def load_test_cases(file_path: str) -> List[TestCase]:
    """Loads and validates test cases from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(e)
            exit(1)
    return [TestCase(**test) for test in data]
