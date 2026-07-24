from datasets.utils.py_utils import Literal
from .stem_scenario import STEMScenario
import datasets
import random
import torch
import os


class MATH(STEMScenario):
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
        print("Loading MATH HENDRYCKS dataset...")

        tensor_files = os.listdir(file_path)

        questions = set()
        for file in tensor_files:
            print(f"Detected file: {file}")

            full_path = f"{file_path}/{file}"
            tensor = torch.load(full_path)

            for tensor_item in tensor:
                question = tensor_item["prompt"].strip()
                questions.add(question)

        dataset = datasets.load_dataset("nlile/hendrycks-MATH-benchmark")
        for item in list(dataset[split]):
            question = item["problem"].strip()
            reference = item["solution"].strip()

            if question in questions:
                continue

            questions.add(question)

            self.items.append({"question": question, "reference": reference})

        random.shuffle(self.items)
