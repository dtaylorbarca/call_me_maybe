from parser import load_functions_definition, load_test_cases


def main() -> None:
    # 1. Parse tool definitions
    tools = load_functions_definition("functions_definition.json")
    print(f"Successfully parsed {len(tools)} tools.\n")

    # Inspect formatted tool snippet for the LLM context
    for tool in tools:
        print(tool.to_system_prompt_snippet())
        print("-" * 40)

    # 2. Parse evaluation test cases
    tests = load_test_cases("function_calling_tests.json")
    print(f"\nSuccessfully parsed {len(tests)} test cases.")
    for test in tests:
        print("Sample prompt:", test.prompt)
        print("-" * 40)


if __name__ == "__main__":
    main()
