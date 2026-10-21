from .parser import (
    load_functions_definition,
    load_user_queries,
    FunctionTool,
    UserQuery,
)
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .arg_parser import parse_args
from .fsm import JSONStateMachine, FSMTokenizerIndex
from .decoder import TokenDecoder
import time
import json
from pathlib import Path


def format_qwen_system_prompt(tools: list[FunctionTool],
                              user_query: UserQuery) -> str:
    """Format the system and user prompt for the Qwen model.

    Constructs a structured prompt containing tool definitions, formatting
    rules, and strict constraints for JSON generation, wrapped in Qwen
    chat template tags.

    Parameters
    ----------
    tools : list[FunctionTool]
        A list of available function tools that the model can call.
    user_query : UserQuery
        The user query object containing the prompt text to be processed.

    Returns
    -------
    str
        The fully formatted prompt string ready for tokenization and inference.
    """
    tool_snippets = "\n\n".join(
        [tool.to_system_prompt_snippet() for tool in tools]
    )

    user_query_str = user_query.prompt
    prompt = f"""<|im_start|>system
You are a function calling assistant with access to the following tools:

{tool_snippets}

When a tool call is required, you MUST output ONLY a valid JSON array
containing the tool call object. Match this exact format (strings have quotes,
numbers and booleans do not):
[
  {{
    "name": "function_name",
    "parameters": {{
      "string_param": "text",
      "number_param": 123,
      "bool_param": true
    }}
  }}
]

CRITICAL RULES:
1. Do NOT include any conversational text, explanations, or introductory fluff.
2. Do NOT wrap the JSON array in Markdown code blocks.
3. The response must start strictly with "[" and end strictly with "]".
4. Any special characters that require escaping (such as double quotes `"` or
backslashes `\\`) inside string values MUST be properly escaped with a
backslash (e.g., `\\"`).
5. For regular expression parameters write the simplest pattern that works.
\\d for matches a digit, \\w a word character, \\s whitespace, + means "one or
more". Write each part once; never repeat a group
6. IN JSON, write each backslash twice, e.g. the regex \\d+ is written \\\\d+.

<|im_end|>
<|im_start|>user
{user_query_str}<|im_end|>
<|im_start|>assistant
"""
    return prompt


def main() -> None:
    """Execute the main constrained generation pipeline.

    Parses command-line arguments, initializes the language model and
    vocabulary,loads function definitions and user queries, and runs
    batch-constrained decoding via the Finite State Machine (FSM) to produce
    structured, valid JSON array outputs.
    """
    args = parse_args()
    model = Small_LLM_Model()

    try:
        tools = load_functions_definition(args.functions_definition)
        prompts = load_user_queries(args.input)
    except ValueError as e:
        print(f"Input error: {e}")
        raise SystemExit(1)
    vocab_path = model.get_path_to_vocab_file()
    start_time = time.time()
    all_outputs = []
    with open(Path(vocab_path), encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)
    decoder = TokenDecoder(vocab)
    tokenizer_index = FSMTokenizerIndex(vocab, decoder)
    total_prompts = len(prompts)

    for idx, prompt_obj in enumerate(prompts):
        prompt_start = time.time()
        is_first = (idx == 0)
        is_last = (idx == total_prompts - 1)
        fsm = JSONStateMachine(tools, prompts, vocab, tokenizer_index, decoder,
                               is_first, is_last)
        fsm.user_query = prompt_obj.prompt
        fsm.prompts = prompts
        initial_text = format_qwen_system_prompt(tools, prompt_obj)
        initial_input_ids = model.encode(initial_text)
        final_output = fsm.constrained_decoding(model, initial_input_ids)
        all_outputs.append(final_output)
        elapsed = time.time() - prompt_start
        print(
            f"Finished prompt {idx+1} ('{prompt_obj.prompt}') in {elapsed:.2f}"
            " seconds")

    output = "".join(all_outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"All prompts finished in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nHey that's mean, let me finish running", end="")
