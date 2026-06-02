import heapq
from collections import defaultdict

from tnqeet import constants
from tnqeet.weights import resolve_weight

_KENLM_INSTALL_HINT = (
    "The n-gram dotter requires KenLM, which is not installed by default because "
    "it compiles from source. Install it with:\n\n"
    '    MAX_ORDER=8 pip install "git+https://github.com/kpu/kenlm.git"\n\n'
    "MAX_ORDER must be >= the n-gram order you load (orders up to 8 are "
    "published). See the tnqeet README for details."
)


def _import_kenlm():
    """Import KenLM lazily, raising a helpful error if it is missing."""
    try:
        import kenlm
    except ImportError as e:
        raise ImportError(_KENLM_INSTALL_HINT) from e
    return kenlm


def to_chars(text):
    chars = list()
    for c in text:
        if c.isspace():
            chars.append("<SPACE>")
        else:
            chars.append(c)
    chars = " ".join(chars)
    return chars


class NgramDotter:
    """Beam-search dotter backed by a KenLM character-level n-gram model.

    Each beam path is ranked by the cumulative conditional log-probability of
    its character sequence under the language model.

    Implementation notes:

    - Stateful KenLM scoring. Each beam path carries a kenlm.State and extends
      via BaseScore, so each step adds the new token's conditional log-prob in
      O(1) rather than rescoring the whole prefix.
    - heapq.nlargest for top-k pruning over the cartesian-product expansion at
      ambiguous positions; for forced positions (single-candidate characters)
      the expansion is already at most beam_size and the prune is skipped.
    """

    def __init__(self, model, beam_size: int = 10):
        self._kenlm = _import_kenlm()
        self.model = model
        self.beam_size = beam_size
        self.rasm_to_letters = defaultdict(list)
        for dotted_letter, rasm in constants.LETTERS_MAPPING.items():
            self.rasm_to_letters[rasm].append(dotted_letter)
        self.rasm_to_letters = dict(self.rasm_to_letters)

    @classmethod
    def from_pretrained(
        cls, order=None, beam_size: int = 10, revision=None, weights_dir=None
    ):
        """Load a pretrained KenLM n-gram dotter by order (e.g. ``6``).

        Downloads the binary from the Hugging Face Hub on demand, or reads from
        a local ``trained_models`` tree when ``weights_dir`` is given.
        """
        kenlm = _import_kenlm()
        binary_path = resolve_weight(
            "ngram", size=order, revision=revision, weights_dir=weights_dir
        )
        return cls(model=kenlm.LanguageModel(binary_path), beam_size=beam_size)

    def _initial_state(self):
        state = self._kenlm.State()
        self.model.BeginSentenceWrite(state)
        return state

    def restore_dots(self, dotless_text: str) -> str:
        tokens = to_chars(dotless_text).split()

        # Each beam entry: (score, sequence_list, kenlm_state)
        beam = [(0.0, [], self._initial_state())]

        for char in tokens:
            candidates = self.rasm_to_letters.get(char, [char])
            new_beam = []
            for score, sequence, state in beam:
                for candidate in candidates:
                    out_state = self._kenlm.State()
                    delta = self.model.BaseScore(state, candidate, out_state)
                    new_beam.append((score + delta, sequence + [candidate], out_state))

            if len(new_beam) > self.beam_size:
                beam = heapq.nlargest(self.beam_size, new_beam, key=lambda x: x[0])
            else:
                beam = new_beam

        best_sequence = max(beam, key=lambda x: x[0])[1]
        return "".join(best_sequence).replace(" ", "").replace("<SPACE>", " ")
