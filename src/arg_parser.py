import argparse
from pathlib import Path
import json


def valid_json_input_file(path_str: str) -> Path:
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
            json.load(f)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(
            f"File '{path_str}' is not a valid JSON. ERROR: {e}"
        )

    return path


def valid_json_output_file(path_str: str) -> Path:
    path = Path(path_str)

    if path.suffix.lower() != ".json":
        raise argparse.ArgumentTypeError(
            f"File '{path_str}' must have a .json extension"
        )

    if not path.parent.exists():
        raise argparse.ArgumentTypeError(
            f"Output directory '{path.parent}' does not exist"
        )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cal Me Maybe constrained decoding engine"
    )

    parser.add_argument(
        "--functions_definition",
        type=valid_json_input_file,
        default=Path("data/input/function_definitions.json"),
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
