from .scenario import Scenario
import datasets
import random
import torch
import os

from typing import Literal


class OLYMPIADBENCH(Scenario):
    def __init__(
        self,
        split: Literal["train", "test"] = "test",
        file_path: str | None = None,
    ):
        super().__init__()
        self.load_dataset(split, file_path=file_path)

    def load_dataset(
        self,
        split: Literal["train", "test"],
        file_path: str | None = None,
    ):
        print(f"Loading OLYMPIAD BENCH dataset...")

        tensor_files = os.listdir(file_path)

        questions = []
        for file in tensor_files:
            print(f"Processing file: {file}")

            full_path = f"{file_path}/{file}"
            tensor = torch.load(full_path)

            for tensor_item in tensor:
                question = tensor_item["prompt"].strip()
                questions.append(question)

        for subject in ["OE_TO_maths_en_COMP", "OE_TO_physics_en_COMP"]:
            dataset = datasets.load_dataset("Hothan/OlympiadBench", subject)

            for item in list(dataset["train"]):
                question = item["question"].strip()
                reference = item["final_answer"][0]

                if question in questions:
                    continue

                self.items.append(
                    {"question": question, "reference": reference}
                )

        for subject in ["TP_TO_maths_en_COMP", "TP_TO_physics_en_COMP"]:
            dataset = datasets.load_dataset("Hothan/OlympiadBench", subject)

            for item in list(dataset["train"]):
                question = item["question"].strip()
                reference = item["solution"][0]

                if question in questions:
                    continue

                self.items.append(
                    {"question": question, "reference": reference}
                )

        random.shuffle(self.items)
