from .scenario import Scenario
import datasets
import random
import torch
import os

from typing import Literal
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
if token:
    login(token=token)


class GPQA(Scenario):
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
        print("Loading GPQA dataset...")

        tensor_files = os.listdir(file_path)

        questions = []
        for file in tensor_files:
            print(f"Processing file: {file}")

            full_path = f"{file_path}/{file}"
            tensor = torch.load(full_path)

            for tensor_item in tensor:
                question = tensor_item["prompt"].strip()
                questions.append(question)

        dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_main")
        for item in list(dataset["train"]):
            question = item["Question"].strip()
            reference = item["Correct Answer"].strip()

            if question in questions:
                continue

            self.items.append({"question": question, "reference": reference})

        random.shuffle(self.items)
