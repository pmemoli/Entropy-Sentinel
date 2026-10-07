from pydantic import BaseModel
from litellm import completion
from concurrent.futures import ThreadPoolExecutor
import argparse
import threading
import torch
import time
import os
from dotenv import load_dotenv

from .core.prompts import load_prompt

load_dotenv(override=True)
api_key = os.getenv("OPENROUTER_KEY")  # type: ignore

STEM_JUDGE_PROMPT = load_prompt("stem-judge")

MODEL = "openrouter/qwen/qwen3-235b-a22b-2507"

JUDGE_KEY = "success_qwen"

WORKERS = 16

BATCH_SIZE = 256

cost_lock = threading.Lock()
total_cost = 0.0
total_calls = 0


class Response(BaseModel):
    success: bool


def evaluate_response(
    question: str, response: str, correct_answer: str
) -> bool:
    cleaned_response = (
        response.replace("<|end|>", "").replace("<|endoftext|>", "").strip()
    )

    agent_prompt = STEM_JUDGE_PROMPT["prompt_template"].format(
        question=question, response=cleaned_response, reference=correct_answer
    )

    messages = [
        {"role": "system", "content": STEM_JUDGE_PROMPT["system_prompt"]},
        {"role": "user", "content": agent_prompt},
    ]

    model = MODEL

    result = completion(
        model=model,
        messages=messages,
        api_key=api_key,  # type: ignore
        max_tokens=32,
        timeout=60,
        extra_body={"provider": {"sort": "price"}},
    )

    global total_cost, total_calls
    with cost_lock:
        total_cost += result._hidden_params.get("response_cost") or 0.0  # type: ignore
        total_calls += 1

    json_content = str(result.choices[0].message.content)  # type: ignore

    return "true" in json_content.lower()


def judge_item(tensor_item):
    for i in range(8):
        try:
            verdict = evaluate_response(
                tensor_item["prompt"],
                tensor_item["generation"],
                tensor_item["reference"],
            )
            tensor_item[JUDGE_KEY] = verdict
            if "success" not in tensor_item:
                tensor_item["success"] = verdict
                tensor_item["judge"] = MODEL.removeprefix("openrouter/")
            return
        except Exception as e:
            print(e)
            time.sleep(min(2**i, 60))


def judge_batch(suite, batch):
    pending = [item for _, _, items in batch for item in items]

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        list(executor.map(judge_item, pending))

    for full_path, tensor, _ in batch:
        torch.save(tensor, full_path)

    judged = sum(JUDGE_KEY in item for item in pending)
    print(
        f"{suite} ({len(batch)} files): judged {judged}/{len(pending)} | "
        f"calls {total_calls} | cost ${total_cost:.4f}",
        flush=True,
    )


def evaluate_suite(suite: str, include_judged: bool = False):
    tensor_path = f"src/data/runs/{suite}"
    tensor_files = sorted(os.listdir(tensor_path))

    batch = []
    for file in tensor_files:
        full_path = f"{tensor_path}/{file}"
        tensor = torch.load(full_path)
        pending = [
            item
            for item in tensor
            if JUDGE_KEY not in item
            and (include_judged or "success" not in item)
            and isinstance(item.get("generation"), str)
        ]

        if pending:
            batch.append((full_path, tensor, pending))

        if sum(len(items) for _, _, items in batch) >= BATCH_SIZE:
            judge_batch(suite, batch)
            batch = []

    if batch:
        judge_batch(suite, batch)

    print("done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="The name of the evaluation suite to process.",
    )
    parser.add_argument("--include_judged", action="store_true")
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        evaluate_suite(args.suite, args.include_judged)
    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


if __name__ == "__main__":
    main()
