"""
test_qna_tool.py
-----------------
Tests against the actual QNATool interface (mirroring qna_tool.h), using a
tiny hand-built corpus fed directly via insert_sentence -- no files, no API
keys needed. Run with:  python -m pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.node import linked_list_to_list
from src.qna_tool import QNATool


GENERAL_FREQ = {
    "the": 1_000_000, "and": 1_000_000, "of": 1_000_000, "he": 1_000_000,
    "was": 1_000_000, "a": 1_000_000, "for": 1_000_000, "is": 1_000_000,
    "gandhi": 5, "partition": 3, "nonviolence": 2, "sunny": 50,
}


def build_toy_tool() -> QNATool:
    tool = QNATool(general_freq=GENERAL_FREQ)

    # Book 1, Page 1, Paragraph 1 -- two sentences
    tool.insert_sentence(1, 1, 1, 1, "Gandhi spoke often about nonviolence and truth.")
    tool.insert_sentence(1, 1, 1, 2, "He believed nonviolence was the only path to freedom.")

    # Book 1, Page 1, Paragraph 2
    tool.insert_sentence(1, 1, 2, 1, "The partition of India caused Gandhi deep sorrow and pain.")

    # Book 1, Page 2, Paragraph 1
    tool.insert_sentence(1, 2, 1, 1, "He fasted many times for peace between communities.")

    # Book 2, Page 1, Paragraph 1 -- irrelevant paragraph
    tool.insert_sentence(2, 1, 1, 1, "The weather today is sunny and warm.")

    # Book 2, Page 1, Paragraph 2
    tool.insert_sentence(2, 1, 2, 1, "Gandhi believed partition was a tragedy that could have been avoided.")

    return tool


def test_get_paragraph_reconstructs_sentences_in_order():
    tool = build_toy_tool()
    text = tool.get_paragraph(1, 1, 1)
    assert text == (
        "Gandhi spoke often about nonviolence and truth. "
        "He believed nonviolence was the only path to freedom."
    )


def test_get_top_k_para_returns_linked_list_of_relevant_paragraphs():
    tool = build_toy_tool()
    head = tool.get_top_k_para("Gandhi partition", 2)
    results = linked_list_to_list(head)

    assert len(results) == 2
    for node in results:
        assert node.book_code is not None and node.page is not None and node.paragraph is not None
        text = tool.get_paragraph(node.book_code, node.page, node.paragraph)
        assert "partition" in text.lower()


def test_get_top_k_para_returns_none_for_no_matches():
    tool = build_toy_tool()
    head = tool.get_top_k_para("xyzzy nonexistentword", 3)
    assert head is None


def test_word_score_rewards_corpus_specific_words():
    tool = build_toy_tool()
    s_gandhi = tool._word_score("gandhi")
    s_the = tool._word_score("the")
    # "gandhi" is frequent here but rare in general English -> higher score.
    assert s_gandhi > s_the


if __name__ == "__main__":
    test_get_paragraph_reconstructs_sentences_in_order()
    test_get_top_k_para_returns_linked_list_of_relevant_paragraphs()
    test_get_top_k_para_returns_none_for_no_matches()
    test_word_score_rewards_corpus_specific_words()
    print("All tests passed.")
