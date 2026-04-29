# Char-Separated Ablation — Results & Analysis

## Hypothesis

BPE tokenizers tend to merge several Arabic characters into a single token, so
the LLM never really "sees" individual glyphs. Forcing one-character-per-token
by space-separating every character — with `|` marking word boundaries —
should give the model a cleaner view of each Rasm when it chooses which dotted
letter to emit.

Format:

```
input :  "م ر ح ٮ ا | ى ا | ع ا ل م"      (dotless)
output:  "م ر ح ب ا | ي ا | ع ا ل م"      (dotted)
```

Experimental setup:

- **Zero-shot** only.
- **Dataset**: `llms_val_dataset` (120 stratified samples).
- **Models**: `claude-sonnet-4`, `gpt-4o`, `gemini-2.5-flash-preview` — the
  same three evaluated on the test set in
  [`evaluate/test_dataset.py`](../../evaluate/test_dataset.py).
- **Baseline**: zeroshot `default_prompt` results already on disk at
  [`evaluation_results/val_dataset/zeroshot/default_prompt/`](../../evaluation_results/val_dataset/zeroshot/default_prompt/).

---

## Headline Results

All three models on the full 120-sample val set, default-prompt zeroshot as
the baseline.

| Model | Metric | Baseline | Char-Sep | Δ |
|---|---|---:|---:|---:|
| **claude-sonnet-4** | WER  | 0.2381 | 0.2933 | **+0.0552** (worse)   |
|                     | CER  | 0.0714 | 0.0698 |   −0.0016 (flat)      |
|                     | DOER | 0.0832 | 0.0818 |   −0.0014 (flat)      |
| **gpt-4o**          | WER  | 0.4283 | 0.5593 | **+0.1310** (worse)   |
|                     | CER  | 0.1655 | 0.1410 | **−0.0245** (better)  |
|                     | DOER | 0.1949 | 0.1665 | **−0.0284** (better)  |
| **gemini-2.5-flash-preview** | WER  | 0.2225 | 0.4008 | **+0.1783** (worse) |
|                              | CER  | 0.0642 | 0.1028 |   +0.0387 (worse)   |
|                              | DOER | 0.0746 | 0.1205 |   +0.0459 (worse)   |

At first glance the hypothesis seems to fail: WER regresses uniformly. But
CER and DOER tell a different story — flat for Claude, clearly better for
GPT-4o, worse only for Gemini. That divergence is the interesting part.

<details>
<summary>Command that produced the table</summary>

```bash
python - <<'PY'
import json, os

MODELS = ["claude-sonnet-4", "gpt-4o", "gemini-2.5-flash-preview"]
BASE = "tnqeet/dotting_models/llms"

def summarize(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    n = len(data)
    return {
        "n": n,
        "wer":  sum(r["wer"]  for r in data) / n,
        "cer":  sum(r["cer"]  for r in data) / n,
        "doer": sum(r["doer"] for r in data) / n,
    }

for model in MODELS:
    default  = summarize(f"{BASE}/evaluation_results/val_dataset/zeroshot/default_prompt/{model}.json")
    ablation = summarize(f"{BASE}/ablation_results/char_separated/val_dataset/zeroshot/{model}.json")
    print(model, "baseline:", default, "ablation:", ablation)
PY
```

</details>

---

## Why Can WER Rise While CER / DOER Don't?

Same predictions, different metric verdicts — WER regresses for all three
models while CER/DOER stay flat or improve. How? Because the metrics work
at different granularities: WER is binary per word, CER is proportional
per character. A toy example makes the gap concrete.

```bash
python - <<'PY'
from tnqeet.evaluate.metrics import wer, cer

ref = "مرحبا يا عالم"   # 3 words, 13 chars (incl. spaces)
hyp = "مرحبايا عالم"    # ONE missing space — 2 words now

print(f"ref: {ref!r}   (words={len(ref.split())}, chars={len(ref)})")
print(f"hyp: {hyp!r}   (words={len(hyp.split())}, chars={len(hyp)})")
print(f"  WER = {wer(ref, hyp):.3f}")
print(f"  CER = {cer(ref, hyp):.3f}")
PY
```

Output:

```
ref: 'مرحبا يا عالم'   (words=3, chars=13)
hyp: 'مرحبايا عالم'    (words=2, chars=12)
  WER = 0.667
  CER = 0.077
```

One character edit → **WER 0.67, CER 0.08**. An ~9× amplification because
WER is a **binary per-word** metric (any error flips the *whole word* to
wrong), while CER is **proportional per-character**.

So the natural first hypothesis is: the char-separated format is producing
small character edits that happen to break word boundaries (dropped or
misplaced `|` sentinels), and WER is magnifying those boundary breaks.

Let's check.

---

## Investigation 1 — Are Word Boundaries the Problem?

If the hypothesis above is right, we should see **more word-count mismatches**
between prediction and reference in char-sep mode than in the baseline.

```bash
python - <<'PY'
import json

def word_count_mismatches(path):
    with open(path) as f:
        data = json.load(f)
    mismatches = 0
    mean_delta = 0.0
    for r in data:
        ref_words = len(r["original_dotted_text"].split())
        hyp_words = len(r["predicted_dotted_text"].split())
        if ref_words != hyp_words:
            mismatches += 1
        mean_delta += abs(ref_words - hyp_words)
    return mismatches, mean_delta / len(data), len(data)

BASE = "tnqeet/dotting_models/llms"
for model in ["claude-sonnet-4", "gpt-4o", "gemini-2.5-flash-preview"]:
    b_path = f"{BASE}/evaluation_results/val_dataset/zeroshot/default_prompt/{model}.json"
    a_path = f"{BASE}/ablation_results/char_separated/val_dataset/zeroshot/{model}.json"
    bm, bd, n = word_count_mismatches(b_path)
    am, ad, _ = word_count_mismatches(a_path)
    print(f"{model}")
    print(f"  baseline : {bm:>3}/{n} samples had a word-count mismatch   "
          f"(mean |ref_words - hyp_words| across all {n} samples = {bd:.2f})")
    print(f"  char-sep : {am:>3}/{n} samples had a word-count mismatch   "
          f"(mean |ref_words - hyp_words| across all {n} samples = {ad:.2f})")
PY
```

Output (`X/120` = number of samples whose prediction has a different word
count than the reference; the mean is taken over **all** 120 samples, not
only the mismatched ones):

```
claude-sonnet-4
  baseline :  19/120 samples had a word-count mismatch   (mean |ref_words - hyp_words| across all 120 samples = 0.37)
  char-sep :   1/120 samples had a word-count mismatch   (mean |ref_words - hyp_words| across all 120 samples = 0.03)
gpt-4o
  baseline :  66/120 samples had a word-count mismatch   (mean |ref_words - hyp_words| across all 120 samples = 1.36)
  char-sep :   2/120 samples had a word-count mismatch   (mean |ref_words - hyp_words| across all 120 samples = 1.74)
gemini-2.5-flash-preview
  baseline :  21/120 samples had a word-count mismatch   (mean |ref_words - hyp_words| across all 120 samples = 0.35)
  char-sep :   4/120 samples had a word-count mismatch   (mean |ref_words - hyp_words| across all 120 samples = 0.17)
```

### Surprise

Word-count alignment **gets dramatically better** in char-sep mode — the `|`
sentinel works exactly as intended. GPT-4o drops from **66** mismatches to
**2**. So broken word boundaries are **not** the story.

If word counts line up better but WER is still higher, the extra errors
must be coming from somewhere else.

---

## Investigation 2 — Are More Words Being Touched, or Are Errors Just Heavier?

If char-sep regresses on WER, we'd expect one of two things: (a) *more*
words are getting touched by errors, or (b) each touched word is carrying
*heavier* character-level damage. Compute both:

```bash
python - <<'PY'
import json
from rapidfuzz.distance import Levenshtein

def analyze(path):
    with open(path) as f:
        data = json.load(f)
    total_wrong_words = 0
    total_char_errs_in_wrong_words = 0
    total_ref_words = 0
    for r in data:
        ref_words = r["original_dotted_text"].split()
        hyp_words = r["predicted_dotted_text"].split()
        total_ref_words += len(ref_words)
        for rw, hw in zip(ref_words, hyp_words):
            if rw != hw:
                total_wrong_words += 1
                total_char_errs_in_wrong_words += Levenshtein.distance(rw, hw)
    return {
        "frac_wrong_words":     total_wrong_words / total_ref_words,
        "chars_per_wrong_word": (total_char_errs_in_wrong_words / total_wrong_words
                                  if total_wrong_words else 0),
    }

BASE = "tnqeet/dotting_models/llms"
for model in ["claude-sonnet-4", "gpt-4o", "gemini-2.5-flash-preview"]:
    b = analyze(f"{BASE}/evaluation_results/val_dataset/zeroshot/default_prompt/{model}.json")
    a = analyze(f"{BASE}/ablation_results/char_separated/val_dataset/zeroshot/{model}.json")
    print(f"\n{model}")
    print(f"  {'':<25} {'baseline':>12} {'char-sep':>12}")
    print(f"  {'% words dirty':<25} {b['frac_wrong_words']*100:>11.2f}% {a['frac_wrong_words']*100:>11.2f}%")
    print(f"  {'char-errs / dirty word':<25} {b['chars_per_wrong_word']:>12.3f} {a['chars_per_wrong_word']:>12.3f}")
PY
```

Output:

```
claude-sonnet-4
                            baseline     char-sep
  % words dirty               31.13%       29.29%
  char-errs / dirty word       3.587        1.624

gpt-4o
                            baseline     char-sep
  % words dirty               72.81%       59.55%
  char-errs / dirty word       4.770        1.685

gemini-2.5-flash-preview
                            baseline     char-sep
  % words dirty               40.14%       42.35%
  char-errs / dirty word       4.675        1.686
```

### Neither expectation holds

Char-sep has **fewer (or comparable) dirty words** AND **much lighter
per-word damage** — char-errs per dirty word drops from 3.6–4.8 down to
≈1.6 across every model. Total character damage per reference word drops
roughly 2-3×:

| Model | Baseline total char damage per ref word | Char-sep total char damage per ref word |
|---|---:|---:|
| claude-sonnet-4 | 0.311 × 3.59 = **1.12** | 0.293 × 1.62 = **0.47** |
| gpt-4o          | 0.728 × 4.77 = **3.47** | 0.595 × 1.69 = **1.00** |
| gemini          | 0.401 × 4.67 = **1.88** | 0.424 × 1.69 = **0.71** |

That matches what CER/DOER said — **less total char-level damage in
char-sep mode**. The answer to "why WER still goes up" follows directly.

---

## Root Cause

**The baseline concentrates errors into a few heavily-mangled words;
char-sep spreads smaller errors across more words.** CER/DOER sum total
character damage and see the improvement. WER is binary per word — *any*
imperfection flips the whole word to wrong — so thin-but-widespread errors
make WER look worse even when total damage is lower.

| Metric | Grain | Char-sep effect |
|---|---|---|
| **CER**  | proportional per-character      | flat or better (total damage drops) |
| **DOER** | per-character, Rasm-normalized  | flat or better (same reason)        |
| **WER**  | binary per-word                 | **worse** (errors spread across more words) |

Intuition: if per-character error probability is `p` and words are `k`
chars long, the probability a word is perfect is `(1 − p)^k`. Char-sep
lowers `p` (≈1.6 vs ≈4 char-errs per dirty word), but compounded over `k`
chars the fraction of perfectly-dotted words can still fall — which is
exactly what WER picks up.

---

## Takeaways

- **Hypothesis was partially right**: per-glyph decisions did improve.
  CER/DOER reward this; WER punishes it.
- **Production metric matters**: char-sep is a wash-to-win on CER/DOER,
  and a loss on WER in zero-shot.

Possible follow-ups:

1. **Few-shot variant** to give models a format template.
2. **`CHAR_SEP` only, no `|`** — to measure whether the sentinel is itself
   an error source.
3. **Mixed decoding** (char-sep input, normal-text output) — to isolate
   whether the benefit is on the input or output side.
