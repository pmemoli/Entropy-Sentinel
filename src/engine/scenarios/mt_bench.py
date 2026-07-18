from .monitoring_scenario import MonitoringScenario
import datasets
import random
import torch
import os

from typing import Literal


class MTBENCH(MonitoringScenario):
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
        print("Loading MT-bench dataset...")

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

        dataset = datasets.load_dataset("philschmid/mt-bench")
        for item in list(dataset["train"]):
            questions = [turn.strip() for turn in item["turns"]]

            if not questions or tuple(questions) in completed:
                continue

            self.items.append(
                {
                    "questions": questions,
                    "primary_category": str(item["category"]),
                    "secondary_categories": [],
                }
            )

        random.shuffle(self.items)
