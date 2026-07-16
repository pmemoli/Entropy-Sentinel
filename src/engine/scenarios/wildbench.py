from .monitoring_scenario import MonitoringScenario
import datasets
import random

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

        dataset = datasets.load_dataset("allenai/WildBench", "v2")
        for item in list(dataset["test"]):
            questions = [
                turn["content"].strip()
                for turn in item["conversation_input"]
                if turn["role"] == "user"
            ]

            self.items.append(
                {
                    "questions": questions,
                    "primary_category": str(item["primary_tag"]),
                    "secondary_categories": list(item["secondary_tags"]),
                }
            )

        random.shuffle(self.items)
