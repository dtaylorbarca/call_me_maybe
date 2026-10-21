*This project has been created as part of the 42 curriculum by dtaylor-.*

# call me maybe — Introduction to function calling in LLMs

## Description

Large language models understand natural language, but they do not reliably
produce machine-readable output. A 0.6B-parameter model asked to "answer in
JSON" fails a large share of the time.

**Goal.** Given a natural-language request such as
`What is the sum of 40 and 2?`, the program does **not** answer `42`. It
returns the function call that would solve the request:

```json
{
    "name": "fn_add_numbers",
    "parameters": {
        "a": 40.0,
        "b": 2.0
    }
}
```

**Overview.**

- The model is `Qwen/Qwen3-0.6B`, used only through the provided `llm_sdk`
  package (`encode`, `get_logits_from_input_ids`, `get_path_to_vocab_file`).
- The model **chooses** the function and **extracts** the arguments.
  Nothing is decided by keywords, regular expressions or other heuristics.
- The output is produced with **constrained decoding**: at every step the
  tokens that would break the JSON structure or the schema of the chosen
  function are removed from consideration, so the result is always valid JSON
  whose keys and types match `functions_definition.json`. The program does
  not rely on the model "behaving" when prompted.
- No `transformers`, `torch`, `outlines`, `dspy` or similar package is used.
  Only `numpy`, `pydantic` and the standard library (plus `llm_sdk`).

## Instructions

### Requirements

- Python **3.10 or later**
- [`uv`](https://docs.astral.sh/uv/)
- The `llm_sdk/` folder next to `src/` (provided with the subject). The first
  run loads `Qwen/Qwen3-0.6B` through it, which downloads the weights if they
  are not cached yet.

### Installation

```bash
make install        # runs: uv sync
```

`uv sync` creates the virtual environment from `pyproject.toml` / `uv.lock`
and installs `numpy`, `pydantic` and the development tools (`flake8`, `mypy`).

### Running

```bash
make run
# same as:
uv run python -m src
```

By default the program reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json` and writes
`data/output/function_calling_results.json` (the `data/output/` folder is
created when needed). Other paths can be given:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json

make run ARGS="--input my_prompts.json --output /tmp/out.json"
```

### Makefile rules

| Rule | What it does |
|---|---|
| `make install` | Installs the dependencies with `uv sync`. |
| `make run` | Runs the program (`uv run python -m src $(ARGS)`). |
| `make debug` | Runs the program under Python's debugger, `pdb` (type `c` to continue). |
| `make clean` | Removes `__pycache__`, `.mypy_cache`, `.pytest_cache` and `*.pyc` (the virtual environment is kept). |
| `make lint` | `flake8 .` and `mypy .` with `--warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`. |
| `make lint-strict` | `flake8 .` and `mypy . --strict`. |

`.flake8` excludes `.venv` and cache folders so `flake8 .` only checks the
project's own files.

## Example usage

`data/input/functions_definition.json`

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": { "a": { "type": "number" }, "b": { "type": "number" } },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all matches of a regex in a string.",
    "parameters": {
      "source_string": { "type": "string" },
      "regex": { "type": "string" },
      "replacement": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```

`data/input/function_calling_tests.json`

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS" }
]
```

Command:

```bash
uv run python -m src
```

`data/output/function_calling_results.json`

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    },
    {
        "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "Hello 34 I'm 233 years old",
            "regex": "(\\d+)",
            "replacement": "NUMBERS"
        }
    }
]
```

(In JSON a backslash is written `\\`, so `"(\\d+)"` is the regex `(\d+)`.)

Input problems are reported with a clear message instead of a traceback:

```text
Error: data/input/function_calling_tests.json is not valid JSON: Expecting value: line 1 column 1 (char 0)
Error: file not found: data/input/functions_definition.json
```

## Project structure

```text
.
├── Makefile
├── README.md
├── pyproject.toml / uv.lock      dependency management (uv)
├── .flake8 / .gitignore
├── llm_sdk/                      provided wrapper around Qwen3-0.6B
├── data/input/                   example input files
└── src/
    ├── __main__.py               entry point, one prompt after the other
    ├── arg_parser.py             command-line options
    ├── parser.py                 pydantic models + input-file validation
    ├── decoder.py                token id -> text table built from the vocab file
    ├── fsm.py                    the constrained-decoding state machine
    └── value_validation.py       number / JSON-string validity rules
```

## Algorithm explanation

### Overview

For every prompt the program

1. builds a chat prompt (system message with the available tools, user
   message with the request),
2. encodes it with `Small_LLM_Model.encode`,
3. runs the constrained generation loop of `JSONStateMachine`, which returns
   the function name and its arguments,
4. rebuilds the JSON from what was generated, validates it and adds it to the
   output array.

### From token ids to text

Constrained decoding needs to know which **text** each token id stands for.
The vocabulary file (`get_path_to_vocab_file`) maps byte-level BPE token
strings to ids. `TokenDecoder` inverts the GPT-2 byte/unicode table (so `Ġ`
becomes a space, `Ċ` a newline, ...) and stores one decoded string per id in a
list, built once at start-up. Ids that are not in the file (special tokens
such as `<|im_start|>`) decode to an empty string and are never allowed.

### The state machine

The text of one answer is `{ "prompt": ..., "name": ..., "parameters": ... }`.
Each part is produced in its own state:

| State | Who decides | Rule |
|---|---|---|
| `START` | forced | `[`, the (JSON-escaped) prompt and `"name": "` are appended without asking the model. |
| `IN_NAME` | model | A token is allowed only if, appended to what was generated so far, the text is still a prefix of `<tool name>"` for at least one tool. A token may never run past the end of the name. |
| `AFTER_NAME` | forced | `, "parameters": {`. A function without parameters goes straight to `END`. |
| `IN_PARAMS` | model | The model picks which unused parameter comes next, token by token, by choosing tokens that keep the text a prefix of `"<param>": ` (string parameters also include the opening `"`). |
| `IN_PARAM_VALUE` | model | Tokens must be valid for the declared type (below). |
| `END` | forced | Closing braces and brackets. |

Text with exactly one valid continuation is **forced**: it is encoded and
appended to the input ids directly, with no forward pass of the model.

### Constrained decoding: masking the logits

At each model-driven step:

1. `logits = model.get_logits_from_input_ids(input_ids)`
2. Every token that is not valid in the current state is treated as having
   a logit of `-inf`.
3. The best remaining token is chosen (greedy).
4. Its text is appended to the buffer, the state is updated and its id is
   appended to `input_ids`.

Instead of building a masked array of ~150 000 logits, the tokens are visited
**from the best logit to the worst** (`numpy.argpartition` for the top 64,
then a full `argsort` only if none of them is valid) and the first valid one
is taken. The first valid token in that order is exactly the `argmax` over the
valid set, so the result is identical to "mask with `-inf`, then `argmax`" —
but only a handful of tokens are examined. For names and parameter keys, a
bucket index (tokens grouped by their first character) limits the candidates.

### Value rules

| Type | Allowed tokens |
|---|---|
| `string` | Content of a valid JSON string: escapes `\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\uXXXX` are checked, raw control characters and partial UTF-8 sequences are rejected. A backslash followed by another character (the model often writes `\d` in a regex) is accepted and repaired afterwards. |
| `number` | The JSON number grammar `-?(0\|[1-9][0-9]*)(\.[0-9]+)?`, checked on every prefix. |
| `integer` | Same, without a fraction. |
| `boolean` | Prefixes of `true` / `false`. |

**Ending a value.** A value can end only when it is complete (a closed string,
a finished number, `true`/`false`). At that point, if the best valid token
*starts with* `"`, `,` or `}` (for example `"`, `",`, `"}`, `}}`), the model is
telling us it wants the value to end. The value is ended and the **exact**
one-character token required by the structure (`"`, then `,` or `}`) is used
instead of the model's token. This guarantees that no stray characters are
left in the output, whatever combined tokens the tokenizer offers.

**Safety limits.** A global step limit (`MAX_STEPS`) and a per-value length
limit stop a runaway generation. A string that repeats the same 8+ characters
twice in a row is closed (small models sometimes loop on regexes).

### Producing the output

The generated text is parsed with `json.loads` (after doubling any backslash
that is not part of a valid JSON escape), `number` values are converted to
`float` and `integer` values to `int`, and the record is re-serialised with
`json.dumps`. The per-prompt pieces are joined into one JSON array. If the
generated text is not valid JSON, an error naming the prompt is reported and
the program continues with the next prompt instead of crashing.

## Design decisions

- **Constrain, don't prompt.** Validity comes from the decoder, not from the
  model. The prompt only tells the model what is expected; the decoder
  guarantees the structure.
- **Let the model choose, let the code force.** The function name, the
  parameter order and the values come from the model. Everything with a single
  valid continuation (separators, braces, the echoed prompt) is forced, which
  saves forward passes (the dominant cost) and removes whole classes of errors.
- **The prompt is never generated.** It is copied into the output with
  `json.dumps`, so quotes, backslashes and newlines in a prompt cannot break
  the JSON.
- **No overshooting tokens.** A token that would run past the text being
  matched (for example `",` after a function name) is rejected, because the
  leftover characters cannot be explained by the grammar. Ending a value is
  handled explicitly (see *Ending a value*).
- **Best-first search instead of a full mask.** Same result as masking, much
  cheaper than checking every token of the vocabulary at each step.
- **Own token table.** Decoding ids with the vocabulary file (rather than a
  tokenizer library) keeps the project within the allowed packages and makes
  the id/text correspondence explicit.
- **pydantic for all structured data.** Function definitions, prompts and
  results are validated models.
- **Input validation is strict where the decoder depends on it.** Function and
  parameter names must be non-empty and free of quotes, backslashes and control
  characters (the decoder uses `"` to find the end of a name), and every
  parameter must declare a supported type. Descriptions are free text.
- **Errors never crash the program.** Missing or invalid input files, bad
  structures and per-prompt failures produce a clear message.

## Performance analysis

Measured on `<describe the machine: CPU/GPU, RAM>` with
`<number>` prompts (`time make run`):

| Metric | Result |
|---|---|
| Valid JSON | `<x>` / `<n>` outputs (by construction 100 %) |
| Correct function selected | `<x>` / `<n>` (`<x> %`) |
| Correct arguments | `<x>` / `<n>` (`<x> %`) |
| Total time | `<x>` s (limit: 5 minutes) |
| Average time per prompt | `<x>` s |

**Reliability.** Structure and types are enforced token by token, and the
final text is re-parsed with `json.loads` before it is written, so every entry
in the file is valid JSON with the keys `prompt`, `name`, `parameters` and
argument types that match the definition.

**Accuracy.** Validity is guaranteed; *correctness* still depends on the
model. Function selection and simple arguments (numbers, short strings) are
reliable. The weak spot is free-form content such as regular expressions: the
constraint keeps the output valid but cannot make a 0.6B model write a good
regex (see *Challenges faced*).

**Speed.** One forward pass is needed per model-chosen token. Forced text costs
none. Choosing the token costs a few milliseconds thanks to the best-first
search, so the runtime is dominated by the model itself and grows with the
number of generated characters (long strings are the slowest part). The
vocabulary table is built once per run.

## Challenges faced

- **Dead ends in the state machine.** After the last parameter, and for
  functions without parameters, no token was allowed. *Fix:* explicit
  transitions to `END`, and the closing text is forced.
- **Stray characters in the output.** Tokens such as `}}`, `",` or `": "`
  carried more text than the grammar expected, and the leftovers corrupted the
  parsing (an extra `}`, a quote inside a number). *Fix:* tokens may never run
  past the text being matched; values end on an exact, forced terminator.
- **Strings that never ended.** The closing quote and the terminator after it
  were not among the candidate tokens. *Fix:* a proper JSON-string checker with
  an explicit `closed` state, and the "token starting with `"` means close"
  rule.
- **Escapes.** Quotes inside a prompt, `\"` inside values and backslashes in
  regexes. *Fix:* the prompt is escaped with `json.dumps`, escapes are
  validated while generating, and a lone backslash is repaired before parsing.
- **Speed.** Checking every one of the ~150 000 vocabulary tokens at each
  step made string values very slow. *Fix:* best-first search with `numpy`,
  and forcing all text that has a single continuation.
- **Special tokens.** Ids such as `<|im_start|>` are not in the vocabulary
  file, so a plain list lookup raised `IndexError`. *Fix:* a safe lookup that
  returns an empty string, and such tokens are never allowed.
- **Small-model looping.** On regular expressions the model sometimes repeated
  the same group indefinitely. *Fix:* step and length limits, the repetition
  guard, and a short generic regex hint in the system prompt.
- **Typing and linting.** `mypy --warn-return-any` rejects untyped helper
  return values. *Fix:* complete type hints, including the validators.

## Testing strategy

- **Input files.** The sample files from the subject plus my own, covering:
  empty strings, very large and negative numbers, decimals, special
  characters (quotes, backslashes, newlines, accents), ambiguous prompts,
  functions with several parameters, with no parameters and with boolean and
  integer parameters.
- **Invalid inputs.** Missing files, empty files, invalid JSON, a JSON object
  instead of an array, a function without a supported parameter type,
  duplicate function names, a prompt without the `prompt` key. All must give
  a clear message and no traceback.
- **Decoder tests with a fake model.** `pytest` tests drive the state machine
  with a scripted fake `Small_LLM_Model` (it favours a chosen answer) and with
  random logits, and check that the output is always valid, has the right
  types and never raises. These tests are not part of the submission.
- **Output validation.** After a run, this script checks the structure and the
  types of every entry against the definitions:

  ```python
  import json

  defs = {f["name"]: f for f in json.load(
      open("data/input/functions_definition.json"))}
  out = json.load(open("data/output/function_calling_results.json"))
  types = {"string": str, "number": (int, float), "integer": int,
           "boolean": bool}
  for i, item in enumerate(out, start=1):
      assert set(item) == {"prompt", "name", "parameters"}, f"#{i}: bad keys"
      schema = defs[item["name"]]["parameters"]
      assert set(item["parameters"]) == set(schema), f"#{i}: bad parameters"
      for key, value in item["parameters"].items():
          wanted = schema[key]["type"]
          ok = isinstance(value, types[wanted])
          if wanted != "boolean" and isinstance(value, bool):
              ok = False
          assert ok, f"#{i}: '{key}' should be {wanted}"
  print(f"{len(out)} entries OK")
  ```

- **Static checks.** `make lint` and `make lint-strict` (flake8 and mypy).
- **Timing.** `time make run` on the full input to check the 5-minute limit.

## Limitations

- Supported parameter types are `string`, `number`, `integer` and `boolean`.
  Other types (arrays, objects, null) are rejected with a clear message when
  the definitions are loaded.
- Characters that need several byte-level tokens (for example emoji) cannot be
  generated inside a string value, because a partial UTF-8 token is rejected.
  They are copied correctly when they appear in the prompt itself.
- The model is greedy; free-form values such as regexes may be valid but not
  optimal.
- Only the first valid tool call per prompt is produced (no chaining).

## Resources

### References

- Qwen3 model card — <https://huggingface.co/Qwen/Qwen3-0.6B>
- B. T. Willard, R. Louf, *Efficient Guided Generation for Large Language
  Models* (2023) — <https://arxiv.org/abs/2307.09702>
- S. Geng et al., *Grammar-Constrained Decoding for Structured NLP Tasks
  without Finetuning* (2023) — <https://arxiv.org/abs/2305.13971>
- R. Sennrich et al., *Neural Machine Translation of Rare Words with Subword
  Units* (byte-pair encoding) — <https://arxiv.org/abs/1508.07909>
- A. Radford et al., *Language Models are Unsupervised Multitask Learners*
  (GPT-2, byte-level BPE)
- RFC 8259, *The JavaScript Object Notation (JSON) Data Interchange Format* —
  <https://www.rfc-editor.org/rfc/rfc8259>
- Python `json` and `re` documentation — <https://docs.python.org/3/library/>
- pydantic documentation — <https://docs.pydantic.dev>
- NumPy `argpartition` / `argsort` — <https://numpy.org/doc/>
- uv documentation — <https://docs.astral.sh/uv/>
- flake8 — <https://flake8.pycqa.org>, mypy — <https://mypy.readthedocs.io>
- PEP 8, PEP 257 (docstrings), PEP 484 (type hints) — <https://peps.python.org>

### Use of AI

An AI assistant (Claude, by Anthropic) was used as a **reviewer and a
debugging aid**, not as a replacement for understanding the code:

- *Reading the subject:* turning it into a requirements checklist and
  checking my implementation against it (allowed packages, output format,
  Makefile rules, README requirements).
- *Debugging the state machine:* explaining dead ends and stray characters from
  my debug traces, and suggesting the rules now used for token validity,
  value termination and string escapes.
- *Performance:* suggesting the best-first token search instead of scanning the
  whole vocabulary, which I then adapted to my own data structures.
- *Tests:* drafting the fake-model tests used to check the decoder.
- *Tooling and documentation:* reviewing the Makefile, the input-file
  validation and the type hints, and drafting this README.

Every suggestion was read, run and tested before being kept, and I can explain
each part of the code. The parts that depend on the model's behaviour (prompt
wording, regex handling) were tuned by running the real model.
