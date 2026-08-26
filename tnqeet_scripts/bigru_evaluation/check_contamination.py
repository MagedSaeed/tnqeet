"""Check contamination between the BiGRU paper's test sets and our train set.

The BiGRU datasets (ATB / Quran / poem / tshkela) live
under ``BiGRUDatasets/<source>/<prefix>_Y_test.txt`` (dotted gold references, one
instance per line). This script measures how many of those test instances already
appear in our training corpus (``MagedSaeed/tnqeet-training-datasets``).

Method: word-level n-gram overlap (the standard decontamination technique used by
GPT-3 / The Pile / OLMo), rather than exact-line or raw-substring matching:

  * Exact-line match under-reports -- our train rows are paragraphs while the BiGRU
    tests are single lines, so identical strings almost never occur.
  * Raw substring containment over-reports -- short lines like "اهلا بكم" appear
    inside thousands of unrelated documents.

A shared *long* word n-gram, by contrast, is strong evidence the same passage was
seen. We build a hashed set of every n-gram in the (normalized) train corpus, then
score each test line by the fraction of its n-grams present in that set. A line is
flagged "contaminated" when that fraction is >= ``--threshold``.

Normalization is applied identically to both sides (strip tashkeel/tatweel, fold
Arabic-presentation forms, unify alef/hamza/yaa/taa-marbuta, drop non-letters,
collapse whitespace) so orthographic style differences between the two corpora do
not create spurious misses. Pass ``--rasm`` to compare in dotless (Rasm) space
instead, which folds away all dotting/orthography and guarantees alignment.

Runtime note: string hashing uses Python's built-in ``hash`` (fast, C-level). It is
consistent within a run but salted across runs; set ``PYTHONHASHSEED=0`` for a
byte-reproducible report.

Usage:
    uv run python tnqeet_scripts/bigru_evaluation/check_contamination.py
    PYTHONHASHSEED=0 uv run python tnqeet_scripts/bigru_evaluation/check_contamination.py --n 8 13
"""

import argparse
import glob
import json
import os
import re
import statistics
from pathlib import Path

import numpy as np
import pyarabic.araby as araby
from tqdm import tqdm

from tnqeet import constants, remove_dots
from tnqeet.data import train_dataset

# --- configuration ----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
BIGRU_ROOT = REPO_ROOT / "BiGRUDatasets"

# source -> subdirectory (test files are matched with "*_Y_test.txt" inside it)
BIGRU_SOURCES = {
    "ATB": "ATB",
    "Quran": "Quran",
    "poem": "poem",
    "tshkela": "tshkela",
}

DEFAULT_N_VALUES = (4, 8, 13)
DEFAULT_THRESHOLD = 0.8
MAX_EXAMPLES = 8  # example contaminated lines kept per source in the report
BUILD_BATCH = 50_000  # train texts hashed per batch when building the index
MASK64 = (1 << 64) - 1

# --- normalization ----------------------------------------------------------

_PRESENTATION_TABLE = str.maketrans(constants.UNICODE_LETTERS_MAPPING)
_ORTHO_TABLE = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ٲ": "ا",
        "ٳ": "ا",
        "ى": "ي",  # alef maqsura -> yaa
        "ة": "ه",  # taa marbuta -> haa
        "ؤ": "و",
        "ئ": "ي",
    }
)
# after the tables above, keep only bare Arabic letters (U+0621..U+064A) and space
_NON_ARABIC = re.compile(r"[^ء-ي ]+")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str, rasm: bool = False) -> str:
    """Canonicalize Arabic text for cross-corpus comparison.

    Applied identically to both the train corpus and the BiGRU test lines so that
    tashkeel and orthographic-style differences do not manufacture false misses.
    """
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = text.translate(_PRESENTATION_TABLE)
    text = text.translate(_ORTHO_TABLE)
    text = _NON_ARABIC.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if rasm:
        text = remove_dots(text)
    return text


def ngram_hashes(words: list[str], n: int) -> list[int]:
    """64-bit hashes of every contiguous n-word window in ``words``."""
    return [hash(" ".join(words[i : i + n])) & MASK64 for i in range(len(words) - n + 1)]


# --- train-corpus index -----------------------------------------------------


def build_index(normalized_texts: list[str], n: int) -> np.ndarray:
    """Sorted, de-duplicated uint64 array of every n-gram hash in the corpus.

    ``normalized_texts`` must already be normalized (see :func:`normalize`); this
    keeps normalization out of the per-n loop so it runs only once.
    """
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(normalized_texts), BUILD_BATCH), desc=f"index n={n}"):
        batch_hashes: list[int] = []
        for text in normalized_texts[start : start + BUILD_BATCH]:
            words = text.split()
            if len(words) >= n:
                batch_hashes.extend(ngram_hashes(words, n))
        if batch_hashes:
            chunks.append(np.fromiter(batch_hashes, dtype=np.uint64, count=len(batch_hashes)))
    if not chunks:
        return np.empty(0, dtype=np.uint64)
    index = np.unique(np.concatenate(chunks))  # unique() also sorts (needed for searchsorted)
    return index


def contains(index: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Boolean membership of ``queries`` in the sorted ``index`` (vectorized)."""
    if index.size == 0 or queries.size == 0:
        return np.zeros(queries.size, dtype=bool)
    pos = np.searchsorted(index, queries)
    pos = np.clip(pos, 0, index.size - 1)
    return index[pos] == queries


# --- BiGRU test loading -----------------------------------------------------


def load_bigru_tests() -> dict[str, list[str]]:
    """Return {source: [test line, ...]} from the *_Y_test.txt files."""
    tests: dict[str, list[str]] = {}
    for source, subdir in BIGRU_SOURCES.items():
        matches = glob.glob(str(BIGRU_ROOT / subdir / "*_Y_test.txt"))
        if not matches:
            print(f"  [warn] no *_Y_test.txt found for source {source} in {subdir}")
            continue
        lines: list[str] = []
        for path in sorted(matches):
            with open(path, encoding="utf-8") as handle:
                lines.extend(line.strip() for line in handle if line.strip())
        tests[source] = lines
    return tests


# --- scoring ----------------------------------------------------------------


def score_source(
    lines: list[str],
    normalized_lines: list[str],
    index: np.ndarray,
    n: int,
    threshold: float,
) -> dict:
    """Score one source's test lines against the train n-gram index.

    ``lines`` holds the raw text (kept for readable examples); ``normalized_lines``
    is the parallel list of pre-normalized text actually compared.
    """
    overlaps: list[float] = []  # per-scored-line fraction of n-grams found in train
    short = 0  # lines with fewer than n words (cannot form an n-gram)
    fully_contained = 0  # lines whose every n-gram is present (verbatim overlap)
    contaminated = 0
    examples: list[dict] = []

    for line, normalized in zip(lines, normalized_lines):
        words = normalized.split()
        if len(words) < n:
            short += 1
            continue
        q = np.fromiter(ngram_hashes(words, n), dtype=np.uint64)
        hit = contains(index, q)
        frac = float(hit.mean())
        overlaps.append(frac)
        if frac >= 1.0:
            fully_contained += 1
        if frac >= threshold:
            contaminated += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append({"overlap": round(frac, 3), "line": line})

    scored = len(overlaps)
    return {
        "n": n,
        "threshold": threshold,
        "total_lines": len(lines),
        "scored_lines": scored,
        "short_lines_excluded": short,
        "contaminated": contaminated,
        "contaminated_pct": round(100 * contaminated / scored, 2) if scored else 0.0,
        "fully_contained": fully_contained,
        "fully_contained_pct": round(100 * fully_contained / scored, 2) if scored else 0.0,
        "mean_overlap": round(statistics.mean(overlaps), 4) if overlaps else 0.0,
        "median_overlap": round(statistics.median(overlaps), 4) if overlaps else 0.0,
        "examples": examples,
    }


# --- driver -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=list(DEFAULT_N_VALUES),
        help="word n-gram size(s) to check (default: 4 8 13)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="fraction of a line's n-grams that must be in train to flag it (default: 0.8)",
    )
    parser.add_argument(
        "--rasm",
        action="store_true",
        help="compare in dotless (Rasm) space instead of dotted normalized text",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only use the first N train examples (for a quick smoke test)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BIGRU_ROOT / "contamination_report.json",
        help="where to write the JSON report",
    )
    args = parser.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        print("[note] PYTHONHASHSEED != 0; report is valid but not byte-reproducible across runs.")

    print("Loading train corpus text ...")
    train_texts = train_dataset["text"]  # type: ignore
    if args.limit:
        train_texts = train_texts[: args.limit]
    print(f"Train examples: {len(train_texts):,}")

    # Normalize the corpus once, up front, and reuse across every n value.
    print("Normalizing train corpus ...")
    train_norm = [normalize(t, rasm=args.rasm) for t in tqdm(train_texts, desc="normalize train")]
    del train_texts

    tests = load_bigru_tests()
    tests_norm = {
        source: [normalize(line, rasm=args.rasm) for line in lines]
        for source, lines in tests.items()
    }
    print("BiGRU test instances:")
    for source, lines in tests.items():
        print(f"  {source:8s}: {len(lines):,}")

    report: dict = {
        "config": {
            "n_values": args.n,
            "threshold": args.threshold,
            "space": "rasm" if args.rasm else "dotted-normalized",
            "train_examples": len(train_norm),
        },
        "results": {},
    }

    for n in args.n:
        print(f"\n=== n = {n} ===")
        index = build_index(train_norm, n=n)
        print(f"  unique train {n}-grams: {index.size:,}")
        for source, lines in tests.items():
            res = score_source(lines, tests_norm[source], index, n=n, threshold=args.threshold)
            report["results"].setdefault(source, {})[str(n)] = res
            print(
                f"  {source:8s}: {res['contaminated']:>6,}/{res['scored_lines']:<6,} "
                f"contaminated ({res['contaminated_pct']:5.2f}%)  "
                f"verbatim {res['fully_contained_pct']:5.2f}%  "
                f"mean-overlap {res['mean_overlap']:.3f}  "
                f"(short excluded: {res['short_lines_excluded']})"
            )
        del index  # free ~GB before building the next n's index

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
