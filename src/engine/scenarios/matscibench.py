from typing import Literal
from .stem_scenario import STEMScenario
import datasets
import random
import torch
import os


class MATSCIBENCH(STEMScenario):
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
        print("Loading MATERIAL SCI BENCH dataset...")

        tensor_files = os.listdir(file_path)

        questions = set()
        if file_path:
            for file in tensor_files:
                print(f"Detected file: {file}")

                full_path = f"{file_path}/{file}"
                tensor = torch.load(full_path)

                for tensor_item in tensor:
                    question = tensor_item["prompt"].strip()
                    questions.add(question)

        dataset = datasets.load_dataset("MatSciBench/MatSciBench")
        for item in list(dataset["test"]):
            question = item["question"].strip()
            reference = item["answer"]

            if question in questions:
                continue

            if item.get("image") == "" or item.get("image") is not None:
                continue

            questions.add(question)

            self.items.append({"question": question, "reference": reference})

        random.shuffle(self.items)
