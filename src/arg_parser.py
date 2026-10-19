import argparse
from pathlib import Path
import json


def valid_json_input_file(path_str: str) -> Path:
    """Validate that an input file exists, is a JSON file, and is non-empty.

    Parameters
    ----------
    path_str : str
        The file path string to validate.

    Returns
    -------
    Path
        The validated file path as a Path object.

    Raises
    ------
    argparse.ArgumentTypeError
        If the file extension is not .json, the file does not exist,
        is empty, or contains invalid JSON syntax.
    """
    path = Path(path_str)

    if path.suffix.lower() != ".json":
        raise argparse.ArgumentTypeError(
            f"File '{path_str}' must have a .json extension"
        )

    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"File '{path_str}' does not exist"
        )

    try:
        with open(path_str, encoding="utf-8") as f:
            data = json.load(f)
            if not data:
                raise argparse.ArgumentTypeError(
                    f"'{path_str}' is an empty file"
                )

    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(
            f"File '{path_str}' is not a valid JSON. ERROR: {e}"
        )

    return path


def valid_json_output_file(path_str: str) -> Path:
    """Validate an output file path and ensure its parent directories exist.

    Parameters
    ----------
    path_str : str
        The output file path string to validate.

    Returns
    -------
    Path
        The validated output file path as a Path object.

    Raises
    ------
    argparse.ArgumentTypeError
        If the file extension is not .json.
    """
    path = Path(path_str)

    if path.suffix.lower() != ".json":
        raise argparse.ArgumentTypeError(
            f"File '{path_str}' must have a .json extension"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments for the decoding engine.

    Configures the argument parser for function definitions, input prompts,
    and output paths, and ensures that all provided file paths are unique.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments.

    Raises
    ------
    argparse.ArgumentTypeError
        If any of the input/output file paths overlap or are identical.
    """
    parser = argparse.ArgumentParser(
        description="Run Call Me Maybe constrained decoding engine"
    )

    parser.add_argument(
        "--functions_definition",
        type=valid_json_input_file,
        default=Path("data/input/functions_definition.json"),
        help="Path to valid JSON functions definition file",
    )

    parser.add_argument(
        "--input",
        type=valid_json_input_file,
        default=Path("data/input/function_calling_tests.json"),
        help="path to valid JSON user input prompts file",
    )

    parser.add_argument(
        "--output",
        type=valid_json_output_file,
        default=Path("data/output/function-calls.json"),
        help="Path where output predictions will be written (.json)",
    )

    parsed = parser.parse_args()

    args = [parsed.functions_definition, parsed.input, parsed.output]
    if len(args) != len(set(args)):
        raise argparse.ArgumentTypeError("All files must be unique")
    return parsed
