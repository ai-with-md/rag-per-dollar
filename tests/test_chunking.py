import pytest

from ragdecisions.chunking import chunk_spans, evidence_found, sentence_spans, whitespace_tokens

TEXT = (
    "The first sentence has five words. Second one is short. "
    "Third sentence is a little bit longer than the others.\n"
    "A new line starts here. End."
)


def chunk_texts(text, max_tokens):
    return [text[s:e] for s, e in chunk_spans(text, max_tokens)]


def test_sentences_are_split_at_punctuation_and_line_breaks():
    sentences = [TEXT[s:e] for s, e in sentence_spans(TEXT)]
    assert sentences == [
        "The first sentence has five words.",
        "Second one is short.",
        "Third sentence is a little bit longer than the others.",
        "A new line starts here.",
        "End.",
    ]


def test_no_chunk_is_over_the_limit():
    for limit in [3, 6, 10, 20, 100]:
        for chunk in chunk_texts(TEXT, limit):
            assert whitespace_tokens(chunk) <= limit


def test_chunks_keep_every_word_in_order_without_overlap():
    for limit in [3, 6, 10, 20, 100]:
        spans = chunk_spans(TEXT, limit)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        assert " ".join(TEXT[s:e] for s, e in spans).split() == TEXT.split()


def test_whole_sentences_are_packed_together_when_they_fit():
    assert chunk_texts(TEXT, 10) == [
        "The first sentence has five words. Second one is short.",
        "Third sentence is a little bit longer than the others.",
        "A new line starts here. End.",
    ]


def test_a_large_limit_gives_one_chunk():
    assert chunk_spans(TEXT, 1000) == [(0, len(TEXT))]


def test_long_sentence_is_cut_between_words():
    assert chunk_texts("one two three four five six seven.", 3) == ["one two three", "four five six", "seven."]


def test_word_longer_than_limit_is_kept_whole():
    def count(text):
        return len(text)  # one token per character
    assert chunk_spans("abcdefgh ij", 4, count) == [(0, 8), (9, 11)]


def test_empty_text_gives_no_chunks():
    assert chunk_spans("", 10) == []
    assert chunk_spans("   \n  ", 10) == []


def test_limit_must_be_positive():
    with pytest.raises(ValueError):
        chunk_spans(TEXT, 0)


def test_evidence_inside_one_retrieved_chunk_is_found():
    assert evidence_found((10, 20), [(0, 50)])


def test_evidence_in_a_chunk_that_was_not_retrieved_is_missed():
    assert not evidence_found((10, 20), [(30, 50)])


def test_evidence_cut_across_two_chunks_needs_most_of_it_retrieved():
    assert evidence_found((10, 20), [(0, 15), (15, 30)])  # both halves
    assert evidence_found((10, 20), [(0, 16)])  # the larger part, 6 of 10 characters
    assert not evidence_found((10, 20), [(0, 14)])  # only 4 of 10 characters
    assert evidence_found((10, 20), [(0, 14)], min_share=0.4)


def test_empty_evidence_is_never_found():
    assert not evidence_found((5, 5), [(0, 10)])
