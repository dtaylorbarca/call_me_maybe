from .parser import (
    load_functions_definition,
    load_user_queries,
    FunctionTool,
    UserQuery,
)
from ..llm_sdk.llm_sdk import Small_LLM_Model
from .arg_parser import parse_args
from fsm import JSONStateMachine


def format_qwen_system_prompt(tools: list[FunctionTool],
                              user_query: UserQuery) -> str:
    tool_snippets = "\n\n".join(
        [tool.to_system_prompt_snippet() for tool in tools]
    )

    user_query_str = user_query.prompt
    prompt = f"""<|im_start|>system
You are a function calling assistant with access to the following tools:

{tool_snippets}

When a tool call is required, you MUST output ONLY a valid JSON array
containing the tool call object. It must match this exact format:
[
  {{
    "name": "function_name",
    "parameters": {{
      "key": "value"
    }}
  }}
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
    args = parse_args()
    model = Small_LLM_Model()

    tools = load_functions_definition(args.functions_definition)
    prompts = load_user_queries(args.input)

    fsm = JSONStateMachine(tools, prompts, str(model.get_path_to_vocab_file))

    initial_text = format_qwen_system_prompt(tools, prompts[0])
    initial_input_ids = model.encode(initial_text)
    final_output = fsm.constrained_decoding(
        model, initial_input_ids)

    with open(args.output, "w") as f:
        f.write(final_output)


if __name__ == "__main__":
    main()
