"""Grading answers: a matching rule for clear cases and a judge prompt for the rest.

Written before any answers were graded and not tuned on results.
"""

from ragdecisions.metrics import normalize_text
from ragdecisions.prompts import DECLINE

YES_NO = {"yes", "no"}

JUDGE_INSTRUCTION = (
    "You are grading a short answer to a question against the accepted reference answers. "
    "The answer is correct if it means the same as at least one reference answer. "
    "Different wording, spelling, abbreviations or extra detail that does not contradict the reference are fine. "
    "The answer is incorrect if it names something different, is only partly right, gives several options "
    "without committing to one, or says the information is missing. "
    "Reply with exactly one word: Correct or Incorrect."
)


def is_decline(answer: str) -> bool:
    """True if the model gave the agreed reply for missing information."""
    return normalize_text(answer) == normalize_text(DECLINE)


def first_word(text: str) -> str:
    words = normalize_text(text).split()
    return words[0] if words else ""


def rule_grade(answer: str, references, answerable: bool):
    """True or False when the rule can decide, None when the judge has to.

    - A question the documents cannot answer is right only if the model declines.
    - An answerable question is wrong if the model declines.
    - An answer equal to a reference after normalizing is right.
    - For Yes or No references, the first word of the answer decides.
    """
    if not answerable:
        return is_decline(answer)
    if is_decline(answer):
        return False
    normalized = normalize_text(answer)
    targets = {normalize_text(r) for r in references if normalize_text(r)}
    if normalized in targets:
        return True
    if targets and targets <= YES_NO:
        word = first_word(answer)
        if word in YES_NO:
            return word in targets
    return None


def build_judge_prompt(question: str, references, answer: str) -> str:
    accepted = "\n".join(f"- {r}" for r in references)
    return (f"{JUDGE_INSTRUCTION}\n\nQuestion: {question}\n\nReference answers:\n{accepted}\n\n"
            f"Answer to grade: {answer}\n\nGrade:")


def parse_judge_reply(reply: str):
    """True for Correct, False for Incorrect, None if the reply is neither."""
    word = first_word(reply)
    if word == "correct":
        return True
    if word == "incorrect":
        return False
    return None
