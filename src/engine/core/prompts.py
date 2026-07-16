import json
import os

_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts.jsonl")


def load_prompt(name: str) -> dict:
    with open(_PROMPTS_PATH) as f:
        for line in f:
            entry = json.loads(line)
            if entry["name"] == name:
                return entry

    raise KeyError(f"No prompt named '{name}' in {_PROMPTS_PATH}")
