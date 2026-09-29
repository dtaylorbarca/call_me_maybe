import json
from .parser import (
    load_functions_definition,
    load_test_cases,
    FunctionTool,
    TestCase
)
from llm_sdk import Small_LLM_Model
from .arg_parser import parse_args
from syntax import JSONSyntaxTokenIDs
from fsm import JSONStateMachine


def format_qwen_system_prompt(tools: list[FunctionTool],
                              user_query: TestCase) -> str:
    tool_snippets = "\n\n".join(
        [tool.to_system_prompt_snippet() for tool in tools])

    user_query_str = user_query.prompt
    prompt = f"""<|im_start|>system
You are a function calling assistant with access to the following tools:

{tool_snippets}

When a tool call is required, you MUST output ONLY a valid JSON array
containing the tool call object. It must match this exact format:
[
  {
    "name": "function_name",
    "parameters": {{
      "key": "value"
    }}
  }
]

CRITICAL RULES:
1. Do NOT include any conversational text, explanations, or introductory fluff
before or after the JSON array.
2. Do NOT wrap the JSON array in Markdown code blocks.
3. The response must start strictly with "[" and end strictly with "]".
<|im_end|>
<|im_start|>user
{user_query_str}<|im_end|>
<|im_start|>assistant
"""
    return prompt


def main() -> None:
    model = Small_LLM_Model()

    args = parse_args()
    tools = load_functions_definition(args.functions_definition)
    prompts = load_test_cases(args.input)
    print(type(prompts[0]))
    syntax_ids = JSONSyntaxTokenIDs(str(model.get_path_to_vocab_file))
    fsm = JSONStateMachine(tools, syntax_ids)
    for prompt in prompts:
        model.encode(format_qwen_system_prompt(tools, prompt))
        run_constrained_decoding(model, fsm)


if __name__ == "__main__":
    main()
