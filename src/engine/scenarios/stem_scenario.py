from ..core.types import STEMInput
from typing import Literal


class STEMScenario:
    def __init__(
        self,
        split: Literal["train", "test"] = "test",
        file_path: str | None = None,
    ):
        self.items: list[STEMInput] = []

    def load_dataset(
        self, split: Literal["train", "test"], file_path: str | None = None
    ):
        pass  # Implemented in subclasses

    def sample(self) -> STEMInput | None:
        return self.items.pop() if self.items else None

    def has_next(self) -> bool:
        return len(self.items) > 0
