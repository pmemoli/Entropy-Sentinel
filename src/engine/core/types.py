from typing import Literal, TypedDict


# For STEM QA benchmarks
class STEMInput(TypedDict):
    question: str
    reference: str


class STEMGenerationResult(TypedDict):
    prompt: str
    reference: str  # correct answer
    generation: str  # assistant response
    sequences: list[int]  # assistant response token ids
    entropy_profile: list[float]  # assistant response entropy profile
    selected_logprobs: list[float]  # assistant response selected logprobs

    # Completed by judge_stem_scenarios.py
    success: bool  # whether the assistant response is correct


# For chat (monitoring) benchmarks
class MonitoringInput(TypedDict):
    questions: list[str]
    primary_category: str
    secondary_categories: list[str]


class MonitoringMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    token_ids: list[int]

    # Only for assistant responses, same size as token_ids
    entropy_profile: list[float]
    selected_logprobs: list[float]

    # Completed by judge_monitoring_scenarios.py
    judge_explanation: None | str  # explanation of the assistant response
    judge_score: None | int  # score of the assistant response (0-10)


class MonitoringGenerationResult(TypedDict):
    messages: list[MonitoringMessage]
    primary_category: str
    secondary_categories: list[str]
