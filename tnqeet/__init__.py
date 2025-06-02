import re

from tnqeet import constants

def remove_dots(text:str) -> str:
    dotless_text = ""
    for word in re.split(r"\s+", text):  # match every white space
        if word.endswith("ن"):
            word = word[:-1] + constants.NOON_RASM
        if word.endswith("ق"):
            word = word[:-1] + constants.QAF_RASM
        if word.startswith("ي") and len(word) > 1:
            word = constants.BAA_RASM + word[1:]
        if len(word) > 1:
            word = word[:-1].replace("ي", constants.BAA_RASM) + word[-1]
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