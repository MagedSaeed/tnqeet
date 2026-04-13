def split_text_by_threshold(text, threshold=1024):
    if not text:
        return []
    if len(text) <= threshold:
        return [text]
    results = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + threshold
        if end_index >= len(text):
            results.append(text[start_index:])
            break
        # Try to split at the last space within the chunk.
        piece = text[start_index:end_index]
        last_space = piece.rfind(" ")
        if last_space != -1:
            split_at = start_index + last_space + 1
        else:
            # No space found — hard cut at threshold.
            split_at = end_index
        if piece.strip():
            results.append(text[start_index:split_at])
        start_index = split_at
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
