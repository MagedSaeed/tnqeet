def split_text_by_threshold(text, threshold=1024):
    if not text:
        return []
    if len(text) <= threshold:
        return [text]
    results = []
    start_index = 0
    end_index = threshold
    while start_index < len(text):
        piece = text[start_index:end_index]
        piece_last_space = piece.rfind(" ")
        if piece_last_space != -1:
            piece = piece[: piece_last_space + 1]
        while end_index <= len(text) and not text[end_index - 1].isspace():
            start_index += 1
            end_index += 1
        start_index += threshold
        end_index += threshold
        if piece.strip():
            results.append(piece)
    return results


# Test with your example
# text = "abcde cd efadk how are you doing today? I hope you are doing well. This is a test string to check the functionality of the split_string_by_threshold function."
# threshold = 16
# from tnqeet.data import test_dataset
# from tqdm.auto import tqdm

# c = 0
# for example in tqdm(test_dataset):
#     text = example["text"]
#     result = split_string_by_threshold(text)
#     if len(result) > 1:
#         c += 1
#         # print(f"Original text: {text}")
#         print(f"Number of pieces: {len(result)}")
# print(c)
