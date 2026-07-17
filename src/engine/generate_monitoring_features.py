# Summarizes each judged assistant response into entropy features

import torch
import argparse
import os

from .core.computations import compute_features


def run(
    suite: str,
):
    file_path = f"src/data/features"

    tensor_path = f"src/data/runs/{suite}"
    tensor_files = os.listdir(tensor_path)

    print(f"Procesando {len(tensor_files)} archivos...")
    feature_list = []
    skipped = 0
    for i, file in enumerate(tensor_files):
        if i % 100 == 0:
            print(f"{i}/{len(tensor_files)}")

        full_path = f"{tensor_path}/{file}"
        tensor = torch.load(full_path)

        for tensor_item in tensor:
            # One row per judged assistant response, not per conversation: the
            # categories are copied onto each so slices stay groupable.
            for message in tensor_item["messages"]:
                if message["role"] != "assistant":
                    continue

                # Conversations the judge skipped carry no score.
                if message.get("judge_score") is None:
                    skipped += 1
                    continue

                entropy_profile = message["entropy_profile"].to(torch.float32)
                selected_logprobs = message["selected_logprobs"]

                features = compute_features(entropy_profile, selected_logprobs)
                feature_list.append(
                    {
                        "features:": features,
                        "judge_score": message["judge_score"],
                        "primary_category": tensor_item["primary_category"],
                        "secondary_categories": tensor_item[
                            "secondary_categories"
                        ],
                    }
                )

    save_path = f"{file_path}/{suite}.pt"
    torch.save(feature_list, save_path)

    print(f"{len(feature_list)} judged responses, {skipped} unjudged skipped")
    print(f"Features saved to {save_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Summarize judged monitoring runs into entropy features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--suite", type=str, required=True, help="Name of the evaluation suite"
    )

    return parser.parse_args()


def main():
    try:
        args = parse_arguments()

        print("\n with the following configuration:\n")
        print(f"  Suite: {args.suite}")
        print("-" * 30)

        run(args.suite)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        raise


if __name__ == "__main__":
    main()
