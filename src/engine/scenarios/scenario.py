from ..core.types import DatasetItem, Task, SampleResult
from ..core.prompts import task_prompts

from typing import Literal


class Scenario:
    def __init__(
        self,
        split: Literal["train", "test"] = "test",
        file_path: str | None = None,
    ):
        self.items: list[DatasetItem] = []

    def load_dataset(
        self, split: Literal["train", "test"], file_path: str | None = None
    ):
        pass  # Implemented in subclasses

    def sample(self, format: Task | None = None) -> SampleResult | None:
        dataset_item = self.items.pop() if self.items else None
        if dataset_item is None:
            return None

        question = dataset_item["question"]
        reference = dataset_item["reference"]

        if format is None:
            prompt = question

        else:
            prompt = task_prompts[format]
            prompt = prompt.format(question=question)

        return {"prompt": prompt, "reference": reference}

    def has_next(self) -> bool:
        return len(self.items) > 0
