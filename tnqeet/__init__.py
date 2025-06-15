import re

from tnqeet import constants
from tnqeet import utils


def remove_dots(text: str) -> str:
    dotless_text = ""
    for word in re.split(r"\s+", text):  # match every white space
        last_letter = utils.last_arabic_letter(word)
        if last_letter == "ن":
            word = word[:-1] + constants.NOON_RASM
        if word.endswith("ق"):
            word = word[:-1] + constants.QAF_RASM
        if word.startswith("ي") and len(word) > 1:
            word = constants.BAA_RASM + word[1:]
        if len(word) > 1:
            # Find last Arabic letter position to avoid replacing it
            last_arabic_pos = -1
            for i in range(len(word) - 1, -1, -1):
                if word[i] in constants.ARABIC_LETTERS:
                    last_arabic_pos = i
                    break
            if last_arabic_pos > 0:
                word = word[:last_arabic_pos].replace("ي", constants.BAA_RASM) + word[last_arabic_pos:]
        word = word.translate(
            word.maketrans(
                "".join(constants.LETTERS_MAPPING.keys()),
                "".join(constants.LETTERS_MAPPING.values()),
            )
        )
        dotless_text += word
        if not word.isspace():
            dotless_text += " "
    return dotless_text.strip()
