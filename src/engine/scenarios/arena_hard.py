from .monitoring_scenario import MonitoringScenario
from huggingface_hub import hf_hub_download
import random
import json
import torch
import os

from typing import Literal


class ARENAHARD(MonitoringScenario):
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
        print("Loading Arena-Hard dataset...")

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

        question_path = hf_hub_download(
            "lmarena-ai/arena-hard-auto",
            "data/arena-hard-v2.0/question.jsonl",
            repo_type="dataset",
        )

        with open(question_path) as file:
            for line in file:
                item = json.loads(line)
                if item["language"] != "English":
                    continue

                questions = [item["prompt"].strip()]

                if not questions[0] or tuple(questions) in completed:
                    continue

                self.items.append(
                    {
                        "questions": questions,
                        "primary_category": str(item["subcategory"]),
                        "secondary_categories": [str(item["category"])],
                    }
                )

        random.shuffle(self.items)
