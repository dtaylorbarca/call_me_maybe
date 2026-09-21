import json
from llm_sdk import Small_LLM_Model
from syntax import JSONSyntaxTokenIDs


def main() -> None:
    model = Small_LLM_Model()

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        try:
            vocab = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(e)
            exit(1)
    syntax_ids = JSONSyntaxTokenIDs(vocab).get_all()


if __name__ == "__main__":
    main()
