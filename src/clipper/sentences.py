"""
Sentence building logic for clipper transcribe pipeline.
"""


from .models import Sentence, Word


def build_sentences(words: list[Word]) -> list[Sentence]:
    """
    Build sentences from a list of words following specific rules.

    Args:
        words: List of Word objects in chronological order

    Returns:
        List of Sentence objects
    """
    if not words:
        return []

    # Preprocessing: filter out empty words
    valid_words = [word for word in words if word.text.strip()]

    if not valid_words:
        return []

    sentences = []
    current_sentence_words = []
    current_start_index = 0

    for i, word in enumerate(valid_words):
        if current_sentence_words:
            prev_word = current_sentence_words[-1]
            # Punctuation-based split: previous word ended a sentence (. ? !)
            # with a non-terminal word following it.
            # Gap-based split: > 0.6s silence between words.
            gap = word.start - prev_word.end
            ends_punct = prev_word.text.strip().endswith(('.', '?', '!'))
            word_ends_sentence = word.text.strip().endswith(('.', '?', '!'))
            if (ends_punct and not word_ends_sentence) or gap > 0.6:
                sentences.append(
                    _create_sentence(current_sentence_words, current_start_index)
                )
                current_sentence_words = []
                current_start_index = i

        current_sentence_words.append(word)

    # Add any remaining words as a sentence (if not empty)
    if current_sentence_words:
        sentences.append(_create_sentence(current_sentence_words, current_start_index))

    # Second pass: split long sentences (> 12.0s) recursively at largest internal gap
    final_sentences = []
    for sentence in sentences:
        if sentence.end - sentence.start <= 12.0:
            final_sentences.append(sentence)
        else:
            # Split the long sentence
            split_sentences = _split_long_sentence(sentence, words)
            final_sentences.extend(split_sentences)

    # Remove empty sentences (just in case)
    return [s for s in final_sentences if s.text.strip()]


def _create_sentence(word_list: list[Word], start_global_index: int) -> Sentence:
    """Create a sentence from a list of words."""
    if not word_list:
        return Sentence(
            start=0.0,
            end=0.0,
            text="",
            word_idx=(0, 0)
        )

    start = word_list[0].start
    end = word_list[-1].end
    text = " ".join(word.text.strip() for word in word_list if word.text.strip())

    # End index is the last word's index (inclusive)
    end_global_index = start_global_index + len(word_list) - 1

    return Sentence(
        start=start,
        end=end,
        text=text,
        word_idx=(start_global_index, end_global_index)
    )


def _split_long_sentence(sentence: Sentence, all_words: list[Word]) -> list[Sentence]:
    """Recursively split a sentence that exceeds 12.0s duration."""
    # If it's already short enough or has only one word, return as-is
    if sentence.end - sentence.start <= 12.0:
        return [sentence]

    # Get the words corresponding to this sentence based on its indices
    start_idx = sentence.word_idx[0]
    end_idx = sentence.word_idx[1] + 1  # +1 because it's inclusive

    # Find all words that correspond to this sentence range
    sent_words = []
    for i, word in enumerate(all_words):
        if start_idx <= i < end_idx:
            sent_words.append(word)

    if not sent_words or len(sent_words) <= 1:
        return [sentence]

    # Find the largest internal gap
    max_gap = 0.0
    split_point = -1

    # Find gaps between consecutive words within this sentence
    for i in range(len(sent_words) - 1):
        gap = sent_words[i + 1].start - sent_words[i].end
        if gap > max_gap:
            max_gap = gap
            split_point = i + 1

    # If no good split point found, return the original sentence
    if split_point == -1 or max_gap <= 0.6:
        return [sentence]

    # Create two sentences from the split
    first_part_words = sent_words[:split_point]
    second_part_words = sent_words[split_point:]

    # Get proper indices for each part
    first_part = first_part_words[0] if first_part_words else None
    first_start_idx = (
        all_words.index(first_part) if first_part is not None else start_idx
    )
    second_part = second_part_words[0] if second_part_words else None
    second_start_idx = (
        all_words.index(second_part) if second_part is not None else end_idx - 1
    )

    # Create the two new sentences
    first_sentence = _create_sentence(first_part_words, first_start_idx)
    second_sentence = _create_sentence(second_part_words, second_start_idx)

    # Recursive splitting of each part
    result = []
    result.extend(_split_long_sentence(first_sentence, all_words))
    result.extend(_split_long_sentence(second_sentence, all_words))

    return result
