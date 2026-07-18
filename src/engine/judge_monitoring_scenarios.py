from litellm import completion
from openai import OpenAI
from pydantic import BaseModel
import argparse
import torch
import time
import json
import re
import os
from dotenv import load_dotenv

from .core.prompts import load_prompt
from .core.types import MonitoringMessage

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")  # type: ignore

SINGLE_TURN_PROMPT = load_prompt("single-v1")
MULTI_TURN_PROMPT = load_prompt("single-v1-multi-turn")

MODEL = "gpt-5.4-mini"

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


def build_judge_messages(
    messages: list[MonitoringMessage], turn: int
) -> list[dict]:
    """Build the system+user messages sent to the judge for the assistant
    response at `turn` (0-indexed). The first turn is judged on its own, the
    second with the first as context, mirroring MT-bench."""
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

    return [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": agent_prompt},
    ]


def evaluate_response(
    messages: list[MonitoringMessage], turn: int
) -> JudgeResponse:
    """Judge a single assistant response synchronously (one API call)."""
    judge_messages = build_judge_messages(messages, turn)

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


def evaluate_suite_batch(suite: str, poll_seconds: int = 30):
    """Judge every un-scored turn in the suite via the OpenAI Batch API in a
    single offline job (~50% cheaper, async turnaround up to 24h).

    Since each judge call is independent, all requests across all files are
    collected into one in-memory JSONL, submitted, polled, then applied back
    onto the message objects. The input JSONL is never written to disk.
    """
    tensor_path = f"src/data/runs/{suite}"
    tensor_files = os.listdir(tensor_path)

    # Loaded tensors kept in memory so results can be written back and re-saved
    # after the batch returns. `locations[i]` is the message object that the
    # request with custom_id == str(i) should be scored into.
    loaded: dict[str, object] = {}
    requests: list[dict] = []
    locations: list[MonitoringMessage] = []

    for file in tensor_files:
        full_path = f"{tensor_path}/{file}"
        tensor = torch.load(full_path)
        loaded[full_path] = tensor

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

                custom_id = str(len(requests))
                requests.append(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": MODEL,
                            "messages": build_judge_messages(messages, turn),
                        },
                    }
                )
                locations.append(message)

    if not requests:
        print("Nothing to judge — every turn already scored.")
        return

    print(f"Submitting {len(requests)} requests as one batch...")

    client = OpenAI(api_key=api_key)

    jsonl_bytes = "\n".join(json.dumps(req) for req in requests).encode(
        "utf-8"
    )
    input_file = client.files.create(
        file=("judge_requests.jsonl", jsonl_bytes),
        purpose="batch",
    )

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    terminal = {"completed", "failed", "expired", "cancelled"}
    while batch.status not in terminal:
        counts = batch.request_counts
        print(
            f"Batch {batch.id}: {batch.status} "
            f"({counts.completed}/{counts.total} done, {counts.failed} failed)"
        )
        time.sleep(poll_seconds)
        batch = client.batches.retrieve(batch.id)

    if batch.status != "completed":
        raise RuntimeError(
            f"Batch ended in status '{batch.status}': {batch.id}"
        )

    # Apply successful responses; leave failures un-scored so a later run
    # (sync or batch) re-attempts only them.
    scored, failed = 0, 0
    output = client.files.content(batch.output_file_id).text
    for line in output.splitlines():
        if not line.strip():
            continue

        obj = json.loads(line)
        message = locations[int(obj["custom_id"])]

        if obj.get("error") or obj["response"]["status_code"] != 200:
            failed += 1
            print(f"Request {obj['custom_id']} errored: {obj.get('error')}")
            continue

        content = obj["response"]["body"]["choices"][0]["message"]["content"]
        try:
            message["judge_explanation"] = content
            message["judge_score"] = parse_score(content)
            scored += 1
        except ValueError as e:
            failed += 1
            print(e)

    if batch.error_file_id:
        errors = client.files.content(batch.error_file_id).text
        print(f"Batch error file:\n{errors}")

    for full_path, tensor in loaded.items():
        torch.save(tensor, full_path)

    print(f"done! scored {scored}, failed {failed}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="The name of the evaluation suite to process.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Judge via one offline OpenAI Batch job instead of live calls.",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        if args.batch:
            evaluate_suite_batch(args.suite)
        else:
            evaluate_suite(args.suite)
    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


if __name__ == "__main__":
    main()
