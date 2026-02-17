from .scenario import Scenario
import datasets
import random
import torch
import os

from typing import Literal


class SCIBENCH(Scenario):
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
        print("Loading SCIBENCH dataset...")

        tensor_files = os.listdir(file_path)

        questions = []
        for file in tensor_files:
            print(f"Processing file: {file}")

            full_path = f"{file_path}/{file}"
            tensor = torch.load(full_path)

            for tensor_item in tensor:
                question = tensor_item["prompt"].strip()
                questions.append(question)

        dataset = datasets.load_dataset("xw27/scibench")
        for item in list(dataset["train"]):
            question = item["problem_text"].strip()
            reference = f"Answer in latex: {item['answer_latex'].strip()}\nAnswer as a number: {item['answer_number'].strip()}"

            if question in questions:
                continue

            self.items.append({"question": question, "reference": reference})

        random.shuffle(self.items)
