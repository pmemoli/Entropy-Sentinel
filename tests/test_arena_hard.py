from src.engine.scenarios.arena_hard import ARENAHARD


def test_loads_expected_number_of_items():
    scenario = ARENAHARD(split="test")

    assert len(scenario.items) == 504
    assert scenario.has_next()


def test_sample_format():
    scenario = ARENAHARD(split="test")

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


def test_all_items_are_single_turn():
    scenario = ARENAHARD(split="test")

    assert all(len(item["questions"]) == 1 for item in scenario.items)
