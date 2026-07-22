from dataclasses import dataclass

import pytest
import torch

from src.engine.run_stem_scenarios import generate_responses


# --- vLLM output mocks -----------------------------------------------------
# Mirror only the attributes generate_responses touches:
#   output.outputs[0].text / .token_ids / .logprobs / .finish_reason


@dataclass
class FakeLogprob:
    logprob: float


@dataclass
class FakeCompletion:
    text: str
    token_ids: list[int]
    logprobs: list[dict]
    finish_reason: str = "stop"


@dataclass
class FakeRequestOutput:
    outputs: list


def make_output(
    text: str, steps: list[dict[int, float]], finish_reason: str = "stop"
) -> FakeRequestOutput:
    """steps[t] maps token_id -> logprob for generation step t.

    The generated token at step t is taken to be the first key of steps[t],
    so the chosen token is guaranteed to be present in its own logprob dict
    (the normal vLLM case)."""
    token_ids = [next(iter(step)) for step in steps]
    logprobs = [
        {tid: FakeLogprob(lp) for tid, lp in step.items()} for step in steps
    ]
    return FakeRequestOutput(
        outputs=[
            FakeCompletion(
                text=text,
                token_ids=token_ids,
                logprobs=logprobs,
                finish_reason=finish_reason,
            )
        ]
    )


class FakeLLM:
    """Returns a pre-staged batch of outputs for the single chat() call and
    records the messages it was handed, so tests can assert on routing."""

    def __init__(self, responses: list[FakeRequestOutput]):
        self._responses = responses
        self.chat_calls: list = []

    def chat(self, messages, sampling_params, use_tqdm=False):
        self.chat_calls.append(messages)
        return self._responses


def sample(question: str, reference: str = "ref"):
    return {"question": question, "reference": reference}


class FakeSamplingParams:
    def __init__(self, max_tokens: int = 1024):
        self.max_tokens = max_tokens


SP = FakeSamplingParams()


# --- form / structure ------------------------------------------------------


def test_output_form_and_field_passthrough():
    samples = [sample("p0", "r0"), sample("p1", "r1")]
    llm = FakeLLM([make_output("g0", [{100: -0.1}]), make_output("g1", [{200: -0.2}])])

    results = generate_responses(llm, samples, SP)

    assert len(llm.chat_calls) == 1
    assert len(results) == 2

    for result, src in zip(results, samples):
        assert set(result.keys()) == {
            "prompt",
            "reference",
            "generation",
            "sequences",
            "entropy_profile",
            "selected_logprobs",
        }
        assert result["prompt"] == src["question"]
        assert result["reference"] == src["reference"]
        assert isinstance(result["sequences"], list)
        assert isinstance(result["entropy_profile"], torch.Tensor)
        assert isinstance(result["selected_logprobs"], torch.Tensor)


def test_only_role_and_content_sent_to_chat():
    samples = [sample("p0")]
    llm = FakeLLM([make_output("g0", [{100: -0.1}])])

    generate_responses(llm, samples, SP)

    convo = llm.chat_calls[0][0]
    assert convo == [{"role": "user", "content": "p0"}]


# --- length invariants -----------------------------------------------------


def test_field_lengths_match_token_ids():
    samples = [sample("p0")]
    steps = [{100: -0.1, 1: -2.0}, {101: -0.3, 2: -1.5}, {102: -0.5, 3: -1.0}]
    llm = FakeLLM([make_output("g0", steps)])

    results = generate_responses(llm, samples, SP)

    n = len(results[0]["sequences"])
    assert n == 3
    assert len(results[0]["entropy_profile"]) == n
    assert len(results[0]["selected_logprobs"]) == n


# --- selected_logprobs indexing --------------------------------------------


def test_selected_logprobs_pick_the_generated_token():
    samples = [sample("p0")]
    steps = [
        {100: -0.10, 5: -3.0},
        {101: -0.20, 6: -2.5},
        {102: -0.30, 7: -9.9},
    ]
    llm = FakeLLM([make_output("g0", steps)])

    results = generate_responses(llm, samples, SP)

    selected = results[0]["selected_logprobs"].tolist()
    assert selected == pytest.approx([-0.10, -0.20, -0.30])


def test_raises_when_generated_token_absent_from_logprobs():
    samples = [sample("p0")]
    bad = FakeRequestOutput(
        outputs=[
            FakeCompletion(
                text="g0",
                token_ids=[999],  # not a key in the step below
                logprobs=[{100: FakeLogprob(-0.1)}],
            )
        ]
    )
    llm = FakeLLM([bad])

    with pytest.raises(KeyError):
        generate_responses(llm, samples, SP)


# --- truncation is kept ------------------------------------------------


def test_truncated_sample_is_kept():
    # Truncated responses (finish_reason="length") are stored, not dropped.
    samples = [sample("p0"), sample("p1"), sample("p2")]
    llm = FakeLLM(
        [
            make_output("g0", [{100: -0.1}]),
            make_output("g1", [{200: -0.2}], finish_reason="length"),
            make_output("g2", [{300: -0.3}]),
        ]
    )

    results = generate_responses(llm, samples, FakeSamplingParams(max_tokens=1024))

    assert len(results) == 3
    assert [r["prompt"] for r in results] == ["p0", "p1", "p2"]
    assert [r["generation"] for r in results] == ["g0", "g1", "g2"]


def test_normal_finish_reason_is_kept():
    samples = [sample("p0")]
    llm = FakeLLM([make_output("g0", [{100: -0.1}])])

    results = generate_responses(llm, samples, SP)

    assert len(results) == 1
    assert results[0]["generation"] == "g0"
