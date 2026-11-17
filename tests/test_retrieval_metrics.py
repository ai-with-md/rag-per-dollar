import math

from ragdecisions.metrics import answer_in_texts, evidence_recall

GOLD = [{"doc_id": "a", "start": 10, "end": 20}, {"doc_id": "b", "start": 0, "end": 10}]


def test_all_evidence_retrieved():
    assert evidence_recall(GOLD, [("a", 0, 50), ("b", 0, 30)]) == (1.0, True)


def test_half_the_evidence_retrieved():
    assert evidence_recall(GOLD, [("a", 0, 50), ("c", 0, 30)]) == (0.5, False)


def test_evidence_must_come_from_its_own_document():
    assert evidence_recall(GOLD, [("b", 0, 50)]) == (0.5, False)  # b's span covers a's offsets, but in b


def test_question_without_evidence_gives_nan():
    share, everything = evidence_recall([], [("a", 0, 10)])
    assert math.isnan(share) and math.isnan(everything)


def test_answer_found_as_whole_words():
    assert answer_in_texts(["Röntgen"], ["The prize went to Wilhelm Rontgen in 1901."])
    assert answer_in_texts(["May 18, 2018"], ["released on May 18 2018 worldwide"])


def test_answer_inside_a_longer_word_does_not_count():
    assert not answer_in_texts(["1901"], ["It cost 19015 dollars."])
    assert not answer_in_texts(["art"], ["The party started."])


def test_any_of_several_answers_counts():
    assert answer_in_texts(["MFSK", "Olivia"], ["", "Modes include Olivia and others."])


def test_empty_answer_never_matches():
    assert not answer_in_texts(["", "..."], ["anything at all"])
