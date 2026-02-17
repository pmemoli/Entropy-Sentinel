from typing import Literal, TypedDict

Task = Literal["cot"]

class DatasetItem(TypedDict):
    question: str
    reference: str

class SampleResult(TypedDict):
    prompt: str
    reference: str
