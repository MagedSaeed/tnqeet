from tnqeet import constants


def last_arabic_letter(word: str) -> str:
    for char in reversed(word):
        if char in constants.ARABIC_LETTERS:
            return char
    return ""
