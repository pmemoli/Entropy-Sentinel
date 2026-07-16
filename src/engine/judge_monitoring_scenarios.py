from litellm import completion
from pydantic import BaseModel
import argparse
import torch
import time
import re
import os
from dotenv import load_dotenv

from .core.prompts import load_prompt
from .core.types import MonitoringMessage

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # type: ignore

SINGLE_TURN_PROMPT = load_prompt("single-v1")
MULTI_TURN_PROMPT = load_prompt("single-v1-multi-turn")

MODEL = "gpt-4.1-mini"

MAX_USER_TURNS = 2


class JudgeResponse(BaseModel):
    explanation: str
    score: int


def clean(content: str) -> str:
    return content.replace("<|end|>", "").replace("<|endoftext|>", "").strip()


def parse_score(content: str) -> int:
    match = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", content)
    if match is None:
        raise ValueError(f"No [[rating]] found in judge response: {content}")

    return int(float(match.group(1)))


def evaluate_response(
    messages: list[MonitoringMessage], turn: int
) -> JudgeResponse:
    """Judge the assistant response at `turn` (0-indexed). The first turn is
    judged on its own, the second with the first as context, mirroring MT-bench.
    """
    if turn == 0:
        prompt = SINGLE_TURN_PROMPT
        agent_prompt = prompt["prompt_template"].format(
            question=messages[0]["content"],
            answer=clean(messages[1]["content"]),
        )
    else:
        prompt = MULTI_TURN_PROMPT
        agent_prompt = prompt["prompt_template"].format(
            question_1=messages[0]["content"],
            answer_1=clean(messages[1]["content"]),
            question_2=messages[2]["content"],
            answer_2=clean(messages[3]["content"]),
        )

    judge_messages = [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": agent_prompt},
    ]

    result = completion(
        model=MODEL,
        messages=judge_messages,
        api_key=api_key,  # type: ignore
    )

    content = str(result.choices[0].message.content)  # type: ignore

    return JudgeResponse(explanation=content, score=parse_score(content))


def evaluate_suite(suite: str):
    tensor_path = f"src/data/runs/{suite}"
    tensor_files = os.listdir(tensor_path)

    for file in tensor_files:
        print(f"Processing file: {file}")

        full_path = f"{tensor_path}/{file}"
        tensor = torch.load(full_path)

        for tensor_item in tensor:
            messages = tensor_item["messages"]

            user_turns = sum(
                1 for message in messages if message["role"] == "user"
            )
            if user_turns > MAX_USER_TURNS:
                print(f"Skipping {user_turns}-turn conversation...")
                continue

            for turn in range(user_turns):
                message = messages[2 * turn + 1]

                if message.get("judge_score") is not None:
                    print("Already evaluated, skipping...")
                    continue

                for i in range(12):
                    try:
                        judgement = evaluate_response(messages, turn)

                        message["judge_explanation"] = judgement.explanation
                        message["judge_score"] = judgement.score
                        print(f"Evaluated score: {judgement.score}")

                        break
                    except Exception as e:
                        print(e)
                        time.sleep(2**i)

        torch.save(tensor, full_path)

    print("done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="The name of the evaluation suite to process.",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        evaluate_suite(args.suite)
    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


if __name__ == "__main__":
    main()
