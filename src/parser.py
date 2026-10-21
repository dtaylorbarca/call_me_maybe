import json
from typing import Any
from typing_extensions import Self
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

SUPPORTED_TYPES = {"string", "number", "integer", "boolean"}


class InputFileError(ValueError):
    """Raised when input JSON content is invalid."""


class DuplicateJSONKeyError(ValueError):
    """Raised when a JSON object contains duplicate keys."""


def reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build a dictionary while rejecting duplicate JSON keys.

    Parameters
    ----------
    pairs : list[tuple[str, Any]]
        A list of key-value pairs decoded from the JSON object.

    Returns
    -------
    dict[str, Any]
        The constructed dictionary without duplicate keys.

    Raises
    ------
    DuplicateJSONKeyError
        If a duplicate key is encountered.
    """
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(
                f"Duplicate JSON key encountered: {key!r}"
            )
        result[key] = value

    return result


def load_json_content(path_str: str) -> Any:
    """Load JSON content from a path already validated by argparse.

    Parameters
    ----------
    path_str : str
        The file path string to load.

    Returns
    -------
    Any
        The decoded JSON data.

    Raises
    ------
    InputFileError
        If an error occurs while reading or decoding the file.
    """
    try:
        with open(path_str, "r", encoding="utf-8") as file:
            return json.load(
                file,
                object_pairs_hook=reject_duplicate_keys,
            )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        DuplicateJSONKeyError,
        RecursionError,
    ) as exc:
        raise InputFileError(
            f"Unable to read JSON content from '{path_str}': {exc}"
        ) from exc


def check_name(kind: str, name: str) -> None:
    """
    Reject names or descriptions that contain quotes, backslashes, or
    control characters.
    """
    if not name.strip():
        raise ValueError(f"{kind} name cannot be blank")
    if any(c in '"\\' or ord(c) < 32 for c in name):
        raise ValueError(
            f"{kind} name {name!r} contains a quote, a backslash "
            "or a control character"
        )


def validate_simple_schema(schema: dict[str, str], location: str) -> None:
    """Validate that a schema contains only a supported type definition."""
    if set(schema) != {"type"}:
        raise ValueError(
            f"{location} must contain only the 'type' field"
        )

    schema_type = schema["type"]

    if schema_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"{location} has an invalid type: {schema_type!r}"
        )


class FunctionTool(BaseModel):
    """Schema representing a single callable tool available to the model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )
    returns: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        """Validate the complete function definition and check string escaping.

        Returns
        -------
        Self
            The validated FunctionTool instance.

        Raises
        ------
        ValueError
            If name, description, parameters, or returns fail validation
            checks.
        """
        check_name("Function", self.name)
        check_name("Function Description", self.description)

        for param_name, param_schema in self.parameters.items():
            check_name("Parameter", param_name)

            validate_simple_schema(
                param_schema,
                f"Parameter {param_name!r}",
            )

        validate_simple_schema(self.returns, "Return schema")

        return self

    def to_system_prompt_snippet(self) -> str:
        """Format the function definition for a system prompt."""
        lines = [
            f"Tool: {self.name}",
            f"Description: {self.description}",
            "Parameters:",
        ]

        if self.parameters:
            for name, schema in self.parameters.items():
                lines.append(f"  - {name} ({schema['type']})")
        else:
            lines.append("  None")

        lines.extend([
            "Returns:",
            f"  - {self.returns['type']}",
        ])

        return "\n".join(lines)


class UserQuery(BaseModel):
    """Schema for evaluating function calling against prompts."""

    model_config = ConfigDict(extra="forbid")

    prompt: str

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        """Ensure the prompt is not blank or whitespace-only.

        Returns
        -------
        Self
            The validated UserQuery instance.

        Raises
        ------
        ValueError
            If the prompt is blank or empty.
        """
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be blank")
        return self


def load_functions_definition(
    file_path: str,
) -> list[FunctionTool]:
    """Load and validate function definitions from a JSON file.

    Parameters
    ----------
    file_path : str
        Path to the JSON file containing function definitions.

    Returns
    -------
    list[FunctionTool]
        A list of validated FunctionTool instances.

    Raises
    ------
    InputFileError
        If the file content is not a valid list, contains duplicates, or fails
        validation.
    """
    data = load_json_content(file_path)

    if not isinstance(data, list) or not data:
        raise InputFileError(
            "Function definitions must be a non-empty JSON list"
        )

    tools: list[FunctionTool] = []
    names: set[str] = set()

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise InputFileError(
                f"Function at index {index} must be a JSON object"
            )
        try:
            tool = FunctionTool.model_validate(item)
        except ValidationError as e:
            raise InputFileError(
                f"Invalid function definition at index {index}: {e}"
            ) from e

        if tool.name in names:
            raise InputFileError(
                f"Duplicate function name: '{tool.name}'"
            )

        names.add(tool.name)
        tools.append(tool)

    return tools


def load_user_queries(file_path: str) -> list[UserQuery]:
    """Load and validate test prompts from a JSON file.

    Parameters
    ----------
    file_path : str
        Path to the JSON file containing user queries.

    Returns
    -------
    list[UserQuery]
        A list of validated UserQuery instances.

    Raises
    ------
    InputFileError
        If the file content is not a valid list or contains invalid queries.
    """
    data = load_json_content(file_path)

    if not isinstance(data, list) or not data:
        raise InputFileError(
            "User queries must be a non-empty JSON list"
        )

    tests: list[UserQuery] = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise InputFileError(
                f"Query at index {index} must be a JSON object"
            )
        try:
            query = UserQuery.model_validate(item)
        except ValidationError as e:
            raise InputFileError(
                f"Invalid user query at index {index}: {e}"
            ) from e

        tests.append(query)

    return tests
