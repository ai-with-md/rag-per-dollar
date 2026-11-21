from ragdecisions.grading import build_judge_prompt, is_decline, parse_judge_reply, rule_grade


def test_decline_is_recognized_with_or_without_the_full_stop():
    assert is_decline("Insufficient information.")
    assert is_decline("insufficient information")
    assert not is_decline("Insufficient information about the trial was given, but it was Google.")


def test_unanswerable_questions_are_right_only_when_declined():
    assert rule_grade("Insufficient information.", ["Insufficient information."], answerable=False) is True
    assert rule_grade("Egypt", ["Insufficient information."], answerable=False) is False


def test_declining_an_answerable_question_is_wrong():
    assert rule_grade("Insufficient information.", ["Google"], answerable=True) is False


def test_exact_match_after_normalizing_is_right():
    assert rule_grade("Sam Bankman-Fried", ["Sam Bankman Fried"], answerable=True) is True
    assert rule_grade("roentgen", ["Wilhelm Conrad Röntgen"], answerable=True) is None


def test_yes_no_is_decided_by_the_first_word():
    assert rule_grade("Yes, both articles agree.", ["Yes"], answerable=True) is True
    assert rule_grade("No", ["yes"], answerable=True) is False
    assert rule_grade("Both say so", ["Yes"], answerable=True) is None


def test_anything_else_goes_to_the_judge():
    assert rule_grade("February to June", ["February"], answerable=True) is None


def test_judge_prompt_lists_every_reference_and_the_answer():
    prompt = build_judge_prompt("Who?", ["MFSK", "Olivia"], "Olivia")
    assert "- MFSK\n- Olivia" in prompt
    assert prompt.endswith("Answer to grade: Olivia\n\nGrade:")


def test_judge_reply_is_parsed_strictly():
    assert parse_judge_reply("Correct") is True
    assert parse_judge_reply("Incorrect.") is False
    assert parse_judge_reply("The answer is correct") is None
