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


def _reject_duplicate_keys(
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


def _load_json_content(path_str: str) -> Any:
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
                object_pairs_hook=_reject_duplicate_keys,
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


def _check_name(kind: str, name: str) -> None:
    """Reject names that would break the decoder's name matching."""
    if not name.strip():
        raise ValueError(f"{kind} name cannot be blank")
    if any(c in '"\\' or ord(c) < 32 for c in name):
        raise ValueError(
            f"{kind} name {name!r} contains a quote, a backslash "
            "or a control character"
        )


def _validate_schema(
    schema: Any,
    location: str,
    *,
    require_type: bool = True,
) -> None:
    """Validate a parameter or return schema recursively.

    Parameters
    ----------
    schema : Any
        The schema object or definition to validate.
    location : str
        The current location path for error reporting.
    require_type : bool, optional
        Whether the schema must explicitly define a 'type' keyword, by default
        True.

    Raises
    ------
    ValueError
        If the schema contains unsupported keywords, invalid types, or broken
        constraints.
    """
    if not isinstance(schema, dict):
        raise ValueError(
            f"{location} must be a JSON object"
        )

    if require_type and "type" not in schema:
        raise ValueError(
            f"{location} must specify a 'type'"
        )

    schema_type = schema.get("type")

    if schema_type is not None and schema_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported type {schema_type!r} at {location}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_TYPES))}"
        )

    if (
        "description" in schema
        and not isinstance(schema["description"], str)
    ):
        raise ValueError(
            f"{location}.description must be a string"
        )

    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(
                f"{location}.enum must be a non-empty list"
            )

    if "const" in schema and "enum" in schema:
        if schema["const"] not in schema["enum"]:
            raise ValueError(
                f"{location}.const must be one of the enum values"
            )

    # Numeric constraints.
    for key in (
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    ):
        if key in schema:
            value = schema[key]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"{location}.{key} must be a number"
                )

            if key == "multipleOf" and value <= 0:
                raise ValueError(
                    f"{location}.multipleOf must be greater than zero"
                )

    if (
        "minimum" in schema
        and "maximum" in schema
        and schema["minimum"] > schema["maximum"]
    ):
        raise ValueError(
            f"{location}.minimum cannot exceed maximum"
        )

    if (
        "exclusiveMinimum" in schema
        and "exclusiveMaximum" in schema
        and schema["exclusiveMinimum"] >= schema["exclusiveMaximum"]
    ):
        raise ValueError(
            f"{location}.exclusiveMinimum must be less than "
            "exclusiveMaximum"
        )

    # String constraints.
    for key in ("minLength", "maxLength"):
        if key in schema:
            value = schema[key]

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{location}.{key} must be an integer"
                )

            if value < 0:
                raise ValueError(
                    f"{location}.{key} cannot be negative"
                )

    if (
        "minLength" in schema
        and "maxLength" in schema
        and schema["minLength"] > schema["maxLength"]
    ):
        raise ValueError(
            f"{location}.minLength cannot exceed maxLength"
        )

    if "pattern" in schema:
        import re

        pattern = schema["pattern"]

        if not isinstance(pattern, str):
            raise ValueError(
                f"{location}.pattern must be a string"
            )

        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"Invalid regular expression at "
                f"{location}.pattern: {exc}"
            ) from exc

    # Array constraints.
    for key in ("minItems", "maxItems"):
        if key in schema:
            value = schema[key]

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{location}.{key} must be an integer"
                )

            if value < 0:
                raise ValueError(
                    f"{location}.{key} cannot be negative"
                )

    if (
        "minItems" in schema
        and "maxItems" in schema
        and schema["minItems"] > schema["maxItems"]
    ):
        raise ValueError(
            f"{location}.minItems cannot exceed maxItems"
        )

    if (
        "uniqueItems" in schema
        and not isinstance(schema["uniqueItems"], bool)
    ):
        raise ValueError(
            f"{location}.uniqueItems must be a boolean"
        )

    # Object constraints.
    for key in ("minProperties", "maxProperties"):
        if key in schema:
            value = schema[key]

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{location}.{key} must be an integer"
                )

            if value < 0:
                raise ValueError(
                    f"{location}.{key} cannot be negative"
                )

    if (
        "minProperties" in schema
        and "maxProperties" in schema
        and schema["minProperties"] > schema["maxProperties"]
    ):
        raise ValueError(
            f"{location}.minProperties cannot exceed maxProperties"
        )

    # Nested object properties.
    properties = schema.get("properties", {})

    if not isinstance(properties, dict):
        raise ValueError(
            f"{location}.properties must be a JSON object"
        )

    for name, child_schema in properties.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{location} contains an invalid property name"
            )

        _validate_schema(
            child_schema,
            f"{location}.properties.{name}",
        )

    required = schema.get("required", [])

    if not isinstance(required, list):
        raise ValueError(
            f"{location}.required must be a list"
        )

    if any(not isinstance(name, str) for name in required):
        raise ValueError(
            f"{location}.required must contain only strings"
        )

    if len(required) != len(set(required)):
        raise ValueError(
            f"{location}.required contains duplicate property names"
        )

    missing = set(required) - set(properties)

    if missing:
        raise ValueError(
            f"{location}.required references undefined properties: "
            f"{', '.join(sorted(missing))}"
        )

    if schema_type != "object" and (
        "properties" in schema or "required" in schema
    ):
        raise ValueError(
            f"{location} uses object keywords but its type is "
            f"{schema_type!r}"
        )

    # Array item schema.
    if "items" in schema:
        if schema_type != "array":
            raise ValueError(
                f"{location}.items requires type 'array'"
            )

        _validate_schema(
            schema["items"],
            f"{location}.items",
        )

    if schema_type == "array" and "items" not in schema:
        raise ValueError(
            f"{location} must define an 'items' schema for arrays"
        )

    # Additional object properties.
    if "additionalProperties" in schema:
        additional = schema["additionalProperties"]

        if isinstance(additional, dict):
            if schema_type != "object":
                raise ValueError(
                    f"{location}.additionalProperties schema "
                    "requires type 'object'"
                )

            _validate_schema(
                additional,
                f"{location}.additionalProperties",
            )

        elif not isinstance(additional, bool):
            raise ValueError(
                f"{location}.additionalProperties must be a "
                "boolean or schema object"
            )


class FunctionTool(BaseModel):
    """Schema representing a single callable tool available to the model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
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
        _check_name("Function", self.name)

        for param_name, param_schema in self.parameters.items():
            _check_name("Parameter", param_name)

            _validate_schema(
                param_schema,
                f"{self.name}.parameters.{param_name}",
            )

        if self.returns:
            _validate_schema(
                self.returns,
                f"{self.name}.returns",
            )

        return self

    def to_system_prompt_snippet(self) -> str:
        """Format the function definition for inclusion in a prompt.

        Returns
        -------
        str
            A clean text snippet detailing the tool name, description,
            parameters, and return type.
        """
        lines: list[str] = []

        def describe_schema(
            name: str,
            schema: dict[str, Any],
            indent: int = 2,
        ) -> None:
            prefix = " " * indent
            schema_type = schema.get("type", "any")
            description = schema.get("description", "")

            line = f"{prefix}- {name} ({schema_type})"

            if description:
                line += f": {description}"

            lines.append(line)

            if schema_type == "object":
                required = set(schema.get("required", []))

                for child_name, child_schema in schema.get(
                    "properties", {}
                ).items():
                    required_label = (
                        "required"
                        if child_name in required
                        else "optional"
                    )

                    lines.append(
                        f"{' ' * (indent + 2)}[{required_label}]"
                    )

                    describe_schema(
                        child_name,
                        child_schema,
                        indent + 4,
                    )

            elif schema_type == "array" and "items" in schema:
                describe_schema(
                    "item",
                    schema["items"],
                    indent + 2,
                )

            if "enum" in schema:
                lines.append(
                    f"{' ' * (indent + 2)}Allowed values: "
                    f"{schema['enum']}"
                )

            for constraint in (
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "minLength",
                "maxLength",
                "minItems",
                "maxItems",
                "pattern",
            ):
                if constraint in schema:
                    lines.append(
                        f"{' ' * (indent + 2)}"
                        f"{constraint}: {schema[constraint]}"
                    )

        for param_name, param_schema in self.parameters.items():
            describe_schema(param_name, param_schema)

        params_str = "\n".join(lines) if lines else "  None"

        return (
            f"Tool: {self.name}\n"
            f"Description: {self.description}\n"
            f"Parameters:\n{params_str}\n"
            f"Returns:\n  - {self.returns.get('type', 'any')}"
        )


class UserQuery(BaseModel):
    """Schema for evaluating function calling against prompts."""

    model_config = ConfigDict(extra="forbid")

    prompt: str


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
    data = _load_json_content(file_path)

    if not isinstance(data, list):
        raise InputFileError(
            f"Function definition file '{file_path}' "
            "must contain a JSON array"
        )

    tools: list[FunctionTool] = []
    names: set[str] = set()

    try:
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise InputFileError(
                    f"Function at index {index} must be a JSON object"
                )

            tool = FunctionTool.model_validate(item)

            if tool.name in names:
                raise InputFileError(
                    f"Duplicate function name: '{tool.name}'"
                )

            names.add(tool.name)
            tools.append(tool)

    except ValidationError as exc:
        raise InputFileError(
            f"Invalid function definition in '{file_path}':\n{exc}"
        ) from exc

    except ValueError as exc:
        if isinstance(exc, InputFileError):
            raise

        raise InputFileError(
            f"Invalid function definition in '{file_path}': {exc}"
        ) from exc

    if not tools:
        raise InputFileError(
            f"Function definition file '{file_path}' contains no functions"
        )

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
    data = _load_json_content(file_path)

    if not isinstance(data, list):
        raise InputFileError(
            f"User query file '{file_path}' must contain a JSON array"
        )

    tests: list[UserQuery] = []

    try:
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise InputFileError(
                    f"Query at index {index} must be a JSON object"
                )

            tests.append(UserQuery.model_validate(item))

    except ValidationError as exc:
        raise InputFileError(
            f"Invalid user query in '{file_path}':\n{exc}"
        ) from exc

    return tests
