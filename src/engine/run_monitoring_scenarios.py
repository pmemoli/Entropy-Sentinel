from .core.types import (
    MonitoringGenerationResult,
    MonitoringMessage,
    MonitoringInput,
)
import argparse
import torch
import time
import gc
import os
import traceback

from vllm import LLM, SamplingParams
from transformers import set_seed
from datasets.utils.py_utils import Literal
from dotenv import load_dotenv


from .core.computations import compute_entropy_profile
from .scenarios import MONITORING_REGISTRY

load_dotenv()

epsilon = 1e-10

MAX_TOKENS = 4096
LOGPROBS = 20


def run(
    dataset_name: str,
    model_name: str,
    suite: str,
    result_path: str,
    split: Literal["train", "test"] = "train",
    max_length: int = MAX_TOKENS,
    seed: int = 42,
    temperature: float = 0.5,
    max_samples: int | None = None,
):
    set_seed(42)

    file_path = f"{result_path}/{suite}"
    os.makedirs(file_path, exist_ok=True)

    def store_results(results):
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"{file_path}/{dataset_name}_{model_name.replace('/', '_')}_{time_stamp}.pt"
        torch.save(results, output_file)

    ScenarioClass = MONITORING_REGISTRY[dataset_name]
    scenario = ScenarioClass(split=split, file_path=file_path)

    if max_samples is not None:
        scenario.items = scenario.items[:max_samples]

    if not scenario.has_next():
        print("No samples found in the scenario. Exiting.")
        return

    print(f"Initializing vLLM with model: {model_name}")

    if "ministral" in model_name.lower():
        model_family = "ministral"
    elif "qwen" in model_name.lower():
        model_family = "qwen"
    elif "phi" in model_name.lower():
        model_family = "phi"
    else:
        model_family = "other"

    llm_params = {
        "ministral": {
            "tokenizer_mode": "mistral",
            "config_format": "mistral",
            "load_format": "mistral",
        },
    }

    llm = LLM(
        model=model_name,
        trust_remote_code=False,
        dtype="bfloat16",
        seed=seed,
        gpu_memory_utilization=0.90,
        max_logprobs=LOGPROBS,
        max_model_len=max_length,
        **llm_params.get(model_family, {}),  # type: ignore
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_length,
        logprobs=LOGPROBS,
        seed=seed,
    )

    tokenizer = llm.get_tokenizer()

    print("Starting generation...")

    batch_size = 32

    amount_processed = 0
    while scenario.has_next():
        batch_samples = []

        while len(batch_samples) < batch_size and scenario.has_next():
            sample = scenario.sample()
            if sample:
                batch_samples.append(sample)
            else:
                break

        if not batch_samples:
            break

        print(
            f"Processing batch starting from sample {amount_processed + 1}..."
        )

        try:
            batch_results = generate_conversations(
                llm=llm,
                tokenizer=tokenizer,
                samples=batch_samples,
                sampling_params=sampling_params,
            )

            store_results(batch_results)
            amount_processed += len(batch_samples)

            del batch_results
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            print(traceback.format_exc())
            continue
            # break

    print(f"Benchmark completed. Results saved to {file_path}")
    print(f"Total samples processed: {amount_processed}")


def generate_conversations(
    llm: LLM,
    tokenizer: any,
    samples: list[MonitoringInput],
    sampling_params: SamplingParams,
) -> list[MonitoringGenerationResult]:
    results: list[MonitoringGenerationResult] = [
        {
            "messages": [],
            "primary_category": sample["primary_category"],
            "secondary_categories": sample["secondary_categories"],
        }
        for sample in samples
    ]

    # max_turns = max(len(sample["questions"]) for sample in samples)
    max_turns = 2  # fixing this for the conversation to fit in the GPU

    for turn in range(max_turns):
        active = [
            i
            for i, sample in enumerate(samples)
            if turn < len(sample["questions"])
        ]
        if not active:
            break

        batch_messages = []
        for i in active:
            question = samples[i]["questions"][turn]

            user_message: MonitoringMessage = {
                "role": "user",
                "content": question,
                "token_ids": tokenizer.encode(question),
                "entropy_profile": [],
                "selected_logprobs": [],
            }
            results[i]["messages"].append(user_message)

            batch_messages.append(
                [
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                    for message in results[i]["messages"]
                ]
            )

        print(
            f"Turn {turn + 1}/{max_turns} — {len(active)} active conversations"
        )

        request_outputs = llm.chat(
            messages=batch_messages,  # type: ignore
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        for i, output in zip(active, request_outputs):
            generated_text = output.outputs[0].text
            logprobs_data = output.outputs[0].logprobs
            token_ids = output.outputs[0].token_ids

            selected_logprobs = [
                logprobs_data[t][token_id].logprob
                for t, token_id in enumerate(token_ids)
            ]
            entropy_profile = compute_entropy_profile(logprobs_data)

            assistant_message: MonitoringMessage = {
                "role": "assistant",
                "content": generated_text,
                "token_ids": list(token_ids),
                "entropy_profile": entropy_profile.cpu(),
                "selected_logprobs": torch.tensor(selected_logprobs).cpu(),
            }
            results[i]["messages"].append(assistant_message)

        del request_outputs

    return results


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap the number of items from the scenario (None = all).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    run(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        suite=args.suite,
        result_path=args.result_path,
        split=args.split,
        max_length=args.max_length,
        seed=args.seed,
        temperature=args.temperature,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
