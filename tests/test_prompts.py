from ragdecisions.prompts import DECLINE, build_closed_book_prompt, build_prompt, passage_header

PASSAGES = [("A", "first text"), ("B", "second text"), ("C", "third text")]


def test_passages_are_numbered_in_rank_order():
    prompt = build_prompt("Who?", PASSAGES)
    assert prompt.index("[1] A\nfirst text") < prompt.index("[2] B\nsecond text") < prompt.index("[3] C\nthird text")


def test_more_passages_extend_the_same_prompt_start():
    two = build_prompt("Who?", PASSAGES[:2])
    three = build_prompt("Who?", PASSAGES)
    shared = two[: two.index("\n\nQuestion: Who?\nShort answer:")]
    assert three.startswith(shared)


def test_question_appears_before_and_after_the_passages():
    prompt = build_prompt("Who won?", PASSAGES)
    assert prompt.count("Question: Who won?") == 2
    assert prompt.endswith("Question: Who won?\nShort answer:")


def test_decline_wording_is_the_same_in_both_prompts():
    assert DECLINE in build_prompt("Who?", PASSAGES)
    assert DECLINE in build_closed_book_prompt("Who?")
    assert "Passages" not in build_closed_book_prompt("Who?")


def test_header_shows_source_and_date_for_news_only():
    assert passage_header("FTX trial", "The Verge", "2023-09-28") == "FTX trial (The Verge, 2023-09-28)"
    assert passage_header("Trade winds", "Wikipedia", "") == "Trade winds"
