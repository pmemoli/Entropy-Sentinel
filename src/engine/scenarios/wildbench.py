from .monitoring_scenario import MonitoringScenario
import datasets
import random
import torch
import os

from typing import Literal


class WILDBENCH(MonitoringScenario):
    def __init__(
        self,
        split: Literal["train", "test"] = "test",
        file_path: str | None = None,
    ):
        super().__init__()
        self.load_dataset(split, file_path=file_path)

    def load_dataset(
        self, split: Literal["train", "test"], file_path: str | None = None
    ):
        print("Loading Wildbench dataset...")

        completed = set()
        if file_path is not None and os.path.isdir(file_path):
            for file in os.listdir(file_path):
                if not file.endswith(".pt"):
                    continue

                print(f"Processing file: {file}")

                for result in torch.load(f"{file_path}/{file}"):
                    completed.add(
                        tuple(
                            message["content"].strip()
                            for message in result["messages"]
                            if message["role"] == "user"
                        )
                    )

            print(f"Skipping {len(completed)} completed conversations.")

        dataset = datasets.load_dataset("allenai/WildBench", "v2")
        for item in list(dataset["test"]):
            questions = [
                turn["content"].strip()
                for turn in item["conversation_input"]
                if turn["role"] == "user"
            ]

            if not questions or tuple(questions) in completed:
                continue

            self.items.append(
                {
                    "questions": questions,
                    "primary_category": str(item["primary_tag"]),
                    "secondary_categories": list(item["secondary_tags"]),
                }
            )

        random.shuffle(self.items)
