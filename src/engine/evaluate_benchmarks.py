from pydantic import BaseModel
from litellm import completion
import argparse
import torch
import time
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("XAI_API_KEY")  # type: ignore


class Response(BaseModel):
    success: bool


def evaluate_response(
    question: str, response: str, correct_answer: str, suite: str
) -> bool:
    # Clean response
    cleaned_response = (
        response.replace("<|end|>", "").replace("<|endoftext|>", "").strip()
    )

    # Quick normalization check first

    agent_prompt = f"""
# Question
{question}

# Response
{cleaned_response}

# Correct Answer
{correct_answer}
"""

    messages = [
        {
            "role": "system",
            "content": """Task: Determine if the Response corresponds to the Correct Answer for the Question, based on the given Correct Answer text. 

            Answer ONLY with the exact format: {{"success": True}} or {{"success": False}}""",
        },
        {"role": "user", "content": agent_prompt},
    ]

    model = "xai/grok-4-1-fast-reasoning"

    result = completion(
        model=model,
        messages=messages,
        api_key=api_key,  # type: ignore
    )

    json_content = str(result.choices[0].message.content)  # type: ignore

    return "true" in json_content.lower()


def evaluate_suite(suite: str):
    tensor_path = f"src/data/runs/{suite}"
    tensor_files = os.listdir(tensor_path)

    for file in tensor_files:
        print(f"Processing file: {file}")

        full_path = f"{tensor_path}/{file}"
        tensor = torch.load(full_path)
        for tensor_item in tensor:
            if "success" in tensor_item:
                print("Already evaluated, skipping...")
                continue

            for i in range(12):
                try:
                    success = evaluate_response(
                        tensor_item["prompt"],
                        tensor_item["generation"],
                        tensor_item["reference"],
                        suite,
                    )

                    tensor_item["success"] = success
                    print(f"Evaluated success: {success}")

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
