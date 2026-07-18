from src.engine.scenarios.mt_bench import MTBENCH


def test_loads_expected_number_of_items():
    scenario = MTBENCH(split="test")

    assert len(scenario.items) == 80
    assert scenario.has_next()


def test_sample_format():
    scenario = MTBENCH(split="test")

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
    assert all(question for question in sample["questions"])

    assert isinstance(sample["primary_category"], str)
    assert len(sample["primary_category"]) > 0

    assert isinstance(sample["secondary_categories"], list)
    assert all(isinstance(tag, str) for tag in sample["secondary_categories"])


def test_all_items_are_two_turn():
    scenario = MTBENCH(split="test")

    assert all(len(item["questions"]) == 2 for item in scenario.items)
