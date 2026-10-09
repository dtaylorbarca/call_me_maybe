import json
from typing_extensions import Self
from typing import Dict, List
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
    ConfigDict,
    field_validator
)
from value_validation import DataTypeValidation


class FunctionTool(BaseModel):
    """Schema representing a single callable tool available to the model."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str
    parameters: Dict[str, dict[str, str]] = Field(default_factory=dict)
    returns: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def escape_validity(self) -> Self:
        validator = DataTypeValidation()

        if not validator.is_valid_json_string(self.name):
            raise ValueError(
                f"The function name '{self.name}' is an invalid JSON string"
            )
        if not validator.is_valid_json_string(self.description):
            raise ValueError(
                f"The description of function '{self.name}' is an invalid JSON"
                " string"
            )

        for param_name, param_schema in self.parameters.items():
            if not validator.is_valid_json_string(param_name):
                raise ValueError(
                    f"Parameter name '{param_name}' in function '{self.name}' "
                    "is an invalid JSON string"
                )
            for k, v in param_schema.items():
                if not validator.is_valid_json_string(k):
                    raise ValueError(
                        f"Parameter property key '{k}' in parameter "
                        f"'{param_name}' of function '{self.name}' is an "
                        "invalid JSON string"
                    )
                if not validator.is_valid_json_string(v):
                    raise ValueError(
                        f"Parameter property value '{v}' in parameter "
                        f"'{param_name}' of function '{self.name}' is an "
                        "invalid JSON string"
                    )

        for ret_key, ret_val in self.returns.items():
            if not validator.is_valid_json_string(ret_key):
                raise ValueError(
                    f"Returns key '{ret_key}' in function '{self.name}' is "
                    "an invalid JSON string"
                )
            if not validator.is_valid_json_string(ret_val):
                raise ValueError(
                    f"Returns value '{ret_val}' in function '{self.name}' is "
                    "an invalid JSON string"
                )

        return self

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
        validator = DataTypeValidation()
        if validator.is_valid_json_string(self.prompt):
            raise ValueError(
                f"The prompt '{self.prompt}' is an invalid JSON string"
            )
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
