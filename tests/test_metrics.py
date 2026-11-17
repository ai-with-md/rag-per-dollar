from ragdecisions.metrics import normalize_text


def test_case_punctuation_and_spacing_are_ignored():
    assert normalize_text("  Hit-points,  OR Health ") == "hit points or health"


def test_accents_are_removed():
    assert normalize_text("Wilhelm Conrad Röntgen") == "wilhelm conrad rontgen"


def test_numbers_are_kept():
    assert normalize_text("May 18, 2018") == "may 18 2018"


def test_empty_text_stays_empty():
    assert normalize_text(" ... ") == ""
