from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from src.engine.run_monitoring_scenarios import generate_conversations


# --- vLLM output mocks -----------------------------------------------------
# Mirror only the attributes generate_conversations touches:
#   output.outputs[0].text / .token_ids / .logprobs
# where logprobs is a per-step list of {token_id: obj-with-.logprob}.


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
    """Returns a pre-staged batch of outputs per chat() call and records the
    messages it was handed, so tests can assert on routing and history."""

    def __init__(self, responses_per_call: list[list[FakeRequestOutput]]):
        self._queue = list(responses_per_call)
        self.chat_calls: list = []

    def chat(self, messages, sampling_params, use_tqdm=False):
        self.chat_calls.append(messages)
        return self._queue.pop(0)


class FakeTokenizer:
    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encoded.append(text)
        return [ord(c) for c in text[:3]] or [0]


def sample(questions, primary="cat_p", secondary=None):
    return {
        "questions": questions,
        "primary_category": primary,
        "secondary_categories": secondary if secondary is not None else ["s1"],
    }


SP = object()  # sampling_params sentinel; the mock ignores it


# --- form / structure ------------------------------------------------------


def test_output_form_and_category_passthrough():
    samples = [
        sample(["q0"], primary="p0", secondary=["a"]),
        sample(["q1"], primary="p1", secondary=["b", "c"]),
    ]
    llm = FakeLLM(
        [[make_output("r0", [{100: -0.1}]), make_output("r1", [{200: -0.2}])]]
    )
    tok = FakeTokenizer()

    results = generate_conversations(llm, tok, samples, SP)

    # one chat call: only turn 0 has active samples (both have 1 question)
    assert len(llm.chat_calls) == 1
    assert len(results) == 2

    for res, src in zip(results, samples):
        assert set(res.keys()) == {
            "messages",
            "primary_category",
            "secondary_categories",
        }
        assert res["primary_category"] == src["primary_category"]
        assert res["secondary_categories"] == src["secondary_categories"]

        assert [m["role"] for m in res["messages"]] == ["user", "assistant"]

        user, assistant = res["messages"]
        assert user["content"] == src["questions"][0]
        assert user["token_ids"] == tok.encode(src["questions"][0])
        assert user["entropy_profile"] == []
        assert user["selected_logprobs"] == []

        assert isinstance(assistant["token_ids"], list)
        assert isinstance(assistant["entropy_profile"], torch.Tensor)
        assert isinstance(assistant["selected_logprobs"], torch.Tensor)


def test_only_role_and_content_sent_to_chat():
    # tensor fields must not leak into the chat payload (would break the
    # chat template).
    samples = [sample(["q0"])]
    llm = FakeLLM([[make_output("r0", [{100: -0.1}])]])

    generate_conversations(llm, FakeTokenizer(), samples, SP)

    convo = llm.chat_calls[0][0]
    assert convo == [{"role": "user", "content": "q0"}]


# --- length invariants -----------------------------------------------------


def test_assistant_field_lengths_match_token_ids():
    samples = [sample(["q0"])]
    steps = [{100: -0.1, 1: -2.0}, {101: -0.3, 2: -1.5}, {102: -0.5, 3: -1.0}]
    llm = FakeLLM([[make_output("r0", steps)]])

    results = generate_conversations(llm, FakeTokenizer(), samples, SP)

    assistant = results[0]["messages"][1]
    n = len(assistant["token_ids"])
    assert n == 3
    assert len(assistant["entropy_profile"]) == n
    assert len(assistant["selected_logprobs"]) == n


# --- selected_logprobs indexing --------------------------------------------


def test_selected_logprobs_pick_the_generated_token():
    samples = [sample(["q0"])]
    # chosen token per step is the first key; its logprob is the expected value
    steps = [
        {100: -0.10, 5: -3.0},
        {101: -0.20, 6: -2.5},
        {102: -0.30, 7: -9.9},
    ]
    llm = FakeLLM([[make_output("r0", steps)]])

    results = generate_conversations(llm, FakeTokenizer(), samples, SP)

    selected = results[0]["messages"][1]["selected_logprobs"].tolist()
    assert selected == pytest.approx([-0.10, -0.20, -0.30])


def test_raises_when_generated_token_absent_from_logprobs():
    # vLLM does not guarantee the sampled token is within the returned top-k;
    # pin that this propagates rather than silently mis-scoring.
    samples = [sample(["q0"])]
    bad = FakeRequestOutput(
        outputs=[
            FakeCompletion(
                text="r0",
                token_ids=[999],  # not a key in the step below
                logprobs=[{100: FakeLogprob(-0.1)}],
            )
        ]
    )
    llm = FakeLLM([[bad]])

    with pytest.raises(KeyError):
        generate_conversations(llm, FakeTokenizer(), samples, SP)


def test_truncated_sample_is_kept():
    # Truncated responses (finish_reason="length") are stored, not dropped -
    # a response cut off at max_tokens is retained like any other.
    samples = [sample(["q0"]), sample(["q1"]), sample(["q2"])]
    llm = FakeLLM(
        [
            [
                make_output("r0", [{100: -0.1}]),
                make_output("r1", [{200: -0.2}], finish_reason="length"),
                make_output("r2", [{300: -0.3}]),
            ]
        ]
    )
    sampling_params = SimpleNamespace(max_tokens=1024)

    results = generate_conversations(
        llm, FakeTokenizer(), samples, sampling_params
    )

    assert len(results) == 3
    contents = [r["messages"][0]["content"] for r in results]
    assert contents == ["q0", "q1", "q2"]
    assert [r["messages"][1]["content"] for r in results] == ["r0", "r1", "r2"]


def test_truncation_on_first_turn_still_runs_second_turn():
    # sample 0 truncates on turn 1; it must still participate in turn 2 like
    # any other multi-turn conversation.
    samples = [
        sample(["q0_0", "q0_1"], primary="p0"),
        sample(["q1_0", "q1_1"], primary="p1"),
    ]
    llm = FakeLLM(
        [
            # turn 0: active = [0, 1]
            [
                make_output(
                    "a0_t0", [{100: -0.1}], finish_reason="length"
                ),
                make_output("a1_t0", [{200: -0.2}]),
            ],
            # turn 1: both samples remain active
            [
                make_output("a0_t1", [{300: -0.3}]),
                make_output("a1_t1", [{400: -0.4}]),
            ],
        ]
    )
    sampling_params = SimpleNamespace(max_tokens=1024)

    results = generate_conversations(
        llm, FakeTokenizer(), samples, sampling_params
    )

    # turn-1 chat call saw both conversations, including the truncated one
    assert len(llm.chat_calls) == 2
    assert llm.chat_calls[1] == [
        [
            {"role": "user", "content": "q0_0"},
            {"role": "assistant", "content": "a0_t0"},
            {"role": "user", "content": "q0_1"},
        ],
        [
            {"role": "user", "content": "q1_0"},
            {"role": "assistant", "content": "a1_t0"},
            {"role": "user", "content": "q1_1"},
        ],
    ]

    # both samples completed both turns
    assert len(results) == 2
    assert [m["content"] for m in results[0]["messages"]] == [
        "q0_0",
        "a0_t0",
        "q0_1",
        "a0_t1",
    ]


def test_normal_finish_reason_is_kept():
    samples = [sample(["q0"])]
    llm = FakeLLM([[make_output("r0", [{100: -0.1}])]])

    results = generate_conversations(llm, FakeTokenizer(), samples, SP)

    assert len(results) == 1
    assert results[0]["messages"][1]["content"] == "r0"


# --- routing across the active filter --------------------------------------


def test_outputs_route_to_correct_sample_across_turns():
    # sample 0 has one question (inactive on turn 1); sample 1 has two.
    samples = [
        sample(["q0_0"], primary="p0"),
        sample(["q1_0", "q1_1"], primary="p1"),
    ]
    llm = FakeLLM(
        [
            # turn 0: active = [0, 1]
            [
                make_output("a0", [{100: -0.1}]),
                make_output("a1_t0", [{200: -0.2}]),
            ],
            # turn 1: active = [1] only
            [make_output("a1_t1", [{300: -0.3}])],
        ]
    )

    results = generate_conversations(llm, FakeTokenizer(), samples, SP)

    # sample 0 stopped after turn 0
    assert [m["content"] for m in results[0]["messages"]] == ["q0_0", "a0"]

    # sample 1 accumulated a full two-turn conversation, correctly ordered
    assert [m["content"] for m in results[1]["messages"]] == [
        "q1_0",
        "a1_t0",
        "q1_1",
        "a1_t1",
    ]
    assert [m["role"] for m in results[1]["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    # turn-1 chat saw exactly one conversation (sample 1) with grown history
    assert len(llm.chat_calls) == 2
    turn1 = llm.chat_calls[1]
    assert len(turn1) == 1
    assert turn1[0] == [
        {"role": "user", "content": "q1_0"},
        {"role": "assistant", "content": "a1_t0"},
        {"role": "user", "content": "q1_1"},
    ]
