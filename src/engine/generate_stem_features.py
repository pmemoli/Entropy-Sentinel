# Stores token-level activations and outputs during generation

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
    for i, file in enumerate(tensor_files):
        if i % 100 == 0:
            print(f"{i}/{len(tensor_files)}")

        full_path = f"{tensor_path}/{file}"
        tensor = torch.load(full_path)

        for tensor_item in tensor:
            entropy_profile = tensor_item["entropy_profile"].to(torch.float32)
            selected_logprobs = tensor_item["selected_logprobs"]

            features = compute_features(entropy_profile, selected_logprobs)
            feature_list.append(
                {
                    "features:": features,
                    "success": tensor_item["success"],
                }
            )

    save_path = f"{file_path}/{suite}.pt"
    torch.save(feature_list, save_path)

    print(f"Features saved to {save_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on language models",
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
