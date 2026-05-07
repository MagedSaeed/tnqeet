import heapq
from collections import defaultdict

import kenlm

from tnqeet import constants


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
        self.model = model
        self.beam_size = beam_size
        self.rasm_to_letters = defaultdict(list)
        for dotted_letter, rasm in constants.LETTERS_MAPPING.items():
            self.rasm_to_letters[rasm].append(dotted_letter)
        self.rasm_to_letters = dict(self.rasm_to_letters)

    def _initial_state(self) -> kenlm.State:
        state = kenlm.State()
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
                    out_state = kenlm.State()
                    delta = self.model.BaseScore(state, candidate, out_state)
                    new_beam.append((score + delta, sequence + [candidate], out_state))

            if len(new_beam) > self.beam_size:
                beam = heapq.nlargest(self.beam_size, new_beam, key=lambda x: x[0])
            else:
                beam = new_beam

        best_sequence = max(beam, key=lambda x: x[0])[1]
        return "".join(best_sequence).replace(" ", "").replace("<SPACE>", " ")
