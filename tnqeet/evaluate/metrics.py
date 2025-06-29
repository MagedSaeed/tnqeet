from rapidfuzz.distance import Levenshtein
from tnqeet import constants


def wer(reference, hypothesis):
    if not reference and not hypothesis:
        return 0.0  # noqa: E701
    if not hypothesis:
        return 1.0
    ref_words = reference.split() if reference else []
    hyp_words = hypothesis.split() if hypothesis else []
    return len(hyp_words) if not ref_words else Levenshtein.distance(ref_words, hyp_words) / len(ref_words)


def cer(reference, hypothesis):
    if not reference and not hypothesis:
        return 0.0  # noqa: E701
    if not hypothesis:
        return 1.0
    return len(hypothesis) if not reference else Levenshtein.distance(reference, hypothesis) / len(reference)


def doer(reference, hypothesis):
    if not reference and not hypothesis:
        return 0.0  # noqa: E701
    if not hypothesis:
        return 1.0
    reference_rasms = "".join(
        c for c in reference if c not in constants.ARABIC_LETTERS or c not in constants.ARABIC_LETTERS_WITHOUT_DOTS
    )
    return len(hypothesis) if not reference else Levenshtein.distance(reference, hypothesis) / len(reference_rasms)
