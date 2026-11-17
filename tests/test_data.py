import pandas as pd
import pytest

from ragdecisions.data import render_nq_tokens, sample_questions
from ragdecisions.grid import config_name, configurations


def test_nq_tokens_become_lines_and_lose_html():
    tokens = ["<H1>", "Title", "</H1>", "<P>", "It", "'s", "a", "test", ",", "really", ".", "</P>"]
    is_html = [t.startswith("<") for t in tokens]
    assert render_nq_tokens(tokens, is_html) == "Title\nIt's a test, really."


def test_nq_table_cells_are_separated():
    tokens = ["<Tr>", "<Td>", "Born", "</Td>", "<Td>", "1879", "</Td>", "</Tr>"]
    is_html = [t.startswith("<") for t in tokens]
    assert render_nq_tokens(tokens, is_html) == "Born | 1879"


def test_nq_brackets_and_html_entities_are_cleaned():
    tokens = ["costs", "(", "about", "$", "5", ")", "&amp;", "more"]
    assert render_nq_tokens(tokens, [False] * len(tokens)) == "costs (about $5) & more"


def make_questions():
    return pd.DataFrame({"qid": range(100), "type": ["a"] * 50 + ["b"] * 30 + ["c"] * 20})


def test_sample_keeps_each_type_share():
    sample = sample_questions(make_questions(), n=10)
    assert len(sample) == 10
    assert sample["type"].value_counts().to_dict() == {"a": 5, "b": 3, "c": 2}


def test_sample_rounds_to_exactly_n():
    sample = sample_questions(make_questions(), n=7)
    assert len(sample) == 7
    assert sample["qid"].is_unique


def test_sample_is_fixed_by_the_seed():
    questions = make_questions()
    assert sample_questions(questions, 20, seed=1).equals(sample_questions(questions, 20, seed=1))
    assert not sample_questions(questions, 20, seed=1).equals(sample_questions(questions, 20, seed=2))


def test_sample_larger_than_the_set_fails():
    with pytest.raises(ValueError):
        sample_questions(make_questions(), n=101)


def test_grid_has_54_distinct_configurations():
    configs = configurations()
    assert len(configs) == 54
    assert len({config_name(**c) for c in configs}) == 54


def test_config_name_is_readable():
    assert config_name(256, "bge-base", True, 5) == "bge-base 256 +rerank k5"
    assert config_name(128, "bm25", False, 3) == "bm25 128 k3"
