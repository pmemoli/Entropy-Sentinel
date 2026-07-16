from .stem_scenario import STEMScenario
import datasets
import random
import torch
import os

from typing import Literal


class THEOREMQA(STEMScenario):
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
        print("Loading TheoremQA dataset...")

        tensor_files = os.listdir(file_path)

        questions = []

        if file_path:
            for file in tensor_files:
                print(f"Processing file: {file}")

                full_path = f"{file_path}/{file}"
                tensor = torch.load(full_path)

                for tensor_item in tensor:
                    question = tensor_item["prompt"].strip()
                    questions.append(question)

        dataset = datasets.load_dataset("TIGER-Lab/TheoremQA")
        for item in list(dataset["test"]):
            question = item["Question"].strip()
            reference = item["Answer"]

            if question in questions or item["Picture"] is not None:
                continue

            self.items.append({"question": question, "reference": reference})

        random.shuffle(self.items)
