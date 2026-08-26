"""Where do each model's ATB errors fall? Splits character errors into the
normalized-letter artifact vs. genuine dotting mistakes.

Reads the per-example predictions written by ``evaluate.py`` and, per model,
counts at the character level:

  * raw char-errors      -- prediction vs. gold, verbatim
  * errors on ا/ه        -- errors at the two letters ATB normalized away
                            (hamza-on-alef -> ا, taa-marbuta -> ه); these are
                            correct restorations our models make that the
                            stripped gold marks as wrong
  * genuine errors       -- after folding BOTH sides (the real dotting mistakes)

It also lists each model's most frequent genuine-error letters and the
CANINE-specific confusions (CANINE wrong while BiLSTM and Transformer are both
right). ``fold`` mirrors the one in ``evaluate.py`` (kept local because importing
that module would run its evaluation).
"""

import glob
import json
from collections import Counter

import pyarabic.araby as araby
from tabulate import tabulate

RESULTS = "tnqeet_scripts/bigru_evaluation/evaluation_results/ATB"
MODELS = ["ngram", "bilstm", "transformer", "canine"]

_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه"})


def fold(t):
    return araby.strip_tatweel(araby.strip_tashkeel(t)).translate(_FOLD)


def load(m):
    return json.load(open(glob.glob(f"{RESULTS}/{m}/*/*.json")[0], encoding="utf-8"))


def main():
    data = {m: load(m) for m in MODELS}

    rows, genuine_top = [], {}
    for m in MODELS:
        raw = on_norm = genuine = 0
        gerr = Counter()
        for r in data[m]:
            g, p = r["original_dotted_text"], r["predicted_dotted_text"]
            if len(g) == len(p):
                for a, b in zip(g, p):
                    if a != b:
                        raw += 1
                        if a in ("ا", "ه"):
                            on_norm += 1
            gf, pf = fold(g), fold(p)
            if len(gf) == len(pf):
                for a, b in zip(gf, pf):
                    if a != b:
                        genuine += 1
                        gerr[a] += 1
        rows.append([m, f"{raw:,}", f"{on_norm:,}", f"{genuine:,}"])
        genuine_top[m] = gerr.most_common(6)

    print(
        tabulate(
            rows,
            headers=["Model", "Raw char-errors", "Errors on ا/ه", "Genuine (post-fold) errors"],
            tablefmt="github",
            colalign=("left", "right", "right", "right"),
        )
    )

    print("\nMost frequent genuine-error letters (gold letter : count):")
    for m in MODELS:
        print(f"  {m:<12}", ", ".join(f"{c}:{k}" for c, k in genuine_top[m]))

    # CANINE-specific confusions: CANINE wrong, BiLSTM and Transformer both right.
    n = min(len(data[m]) for m in MODELS)
    conf = Counter()
    for i in range(n):
        g = fold(data["canine"][i]["original_dotted_text"])
        pc = fold(data["canine"][i]["predicted_dotted_text"])
        pb = fold(data["bilstm"][i]["predicted_dotted_text"])
        pt = fold(data["transformer"][i]["predicted_dotted_text"])
        if len(g) == len(pc) == len(pb) == len(pt):
            for a, cc, cb, ct in zip(g, pc, pb, pt):
                if cc != a and cb == a and ct == a:
                    conf[(a, cc)] += 1

    print("\nTop CANINE-specific confusions (gold -> CANINE, both others correct):")
    for (a, c), k in conf.most_common(10):
        print(f"  {a} -> {c}   x{k}")


main()
