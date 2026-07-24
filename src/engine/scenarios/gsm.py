from .stem_scenario import STEMScenario
import datasets
import random
import torch
import os

from typing import Literal


class GSM8K(STEMScenario):
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
        print("Loading GSM8K dataset...")

        tensor_files = os.listdir(file_path)

        questions = set()
        for file in tensor_files:
            print(f"Processing file: {file}")

            full_path = f"{file_path}/{file}"
            tensor = torch.load(full_path)

            for tensor_item in tensor:
                question = tensor_item["prompt"].strip()
                questions.add(question)

        dataset = datasets.load_dataset("gsm8k", "main")
        for item in list(dataset[split]):
            question = item["question"].strip()
            reference = str(item["answer"]).strip()

            if question in questions:
                continue

            questions.add(question)

            self.items.append({"question": question, "reference": reference})

        random.shuffle(self.items)
