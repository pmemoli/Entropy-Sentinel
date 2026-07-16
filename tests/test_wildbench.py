from src.engine.scenarios.wildbench import WILDBENCH


def test_loads_expected_number_of_items():
    scenario = WILDBENCH(split="test")

    # allenai/WildBench v2 "test" split has 1024 rows.
    assert len(scenario.items) == 1024
    assert scenario.has_next()


def test_sample_format():
    scenario = WILDBENCH(split="test")

    sample = scenario.sample()

    assert sample is not None
    assert set(sample.keys()) == {
        "questions",
        "primary_category",
        "secondary_categories",
    }

    assert isinstance(sample["questions"], list)
    assert len(sample["questions"]) > 0
    assert all(isinstance(question, str) for question in sample["questions"])

    assert isinstance(sample["primary_category"], str)
    assert len(sample["primary_category"]) > 0

    assert isinstance(sample["secondary_categories"], list)
    assert all(isinstance(tag, str) for tag in sample["secondary_categories"])
