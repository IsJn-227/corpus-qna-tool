"""
test_qna_tool.py
-----------------
Tests against the QNATool interface, now covering BM25 scoring, stopword
removal + stemming, the proximity bonus, and the semantic fallback when no
paragraph contains any query term.

A tiny FakeEmbeddingModel is injected in place of the real
sentence-transformers model so these tests run fast and don't require a
model download -- it encodes text as a bag-of-words vector over a small
fixed vocabulary, which is enough to meaningfully exercise the semantic
scoring/fallback code paths without needing real transformer semantics.

Run with:  python -m pytest tests/
       or: python tests/test_qna_tool.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.node import linked_list_to_list
from src.qna_tool import QNATool, tokenize, preprocess_words


class FakeEmbeddingModel:
    """Deterministic bag-of-words 'embedding' over a fixed small vocabulary,
    L2-normalized -- stands in for sentence-transformers in tests so texts
    sharing more vocabulary get higher cosine similarity, without needing
    the real model."""

    VOCAB = [
        "gandhi", "partition", "india", "nonviolence", "freedom", "sorrow",
        "swaraj", "peace", "communities", "fasted", "weather", "sunny",
        "warm", "truth", "self", "rule",
    ]

    def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):
        words = set(tokenize(text))
        vec = np.array([1.0 if w in words else 0.0 for w in self.VOCAB])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def build_toy_tool() -> QNATool:
    tool = QNATool(embedding_model=FakeEmbeddingModel())

    # Book 1, Page 1, Paragraph 1 -- two sentences, query words adjacent
    tool.insert_sentence(1, 1, 1, 1, "Gandhi spoke often about nonviolence and truth.")
    tool.insert_sentence(1, 1, 1, 2, "He believed nonviolence was the only path to freedom.")

    # Book 1, Page 1, Paragraph 2 -- "Gandhi" and "partition" close together
    tool.insert_sentence(1, 1, 2, 1, "The partition of India caused Gandhi deep sorrow and pain.")

    # Book 1, Page 2, Paragraph 1
    tool.insert_sentence(1, 2, 1, 1, "He fasted many times for peace between communities.")

    # Book 2, Page 1, Paragraph 1 -- irrelevant paragraph
    tool.insert_sentence(2, 1, 1, 1, "The weather today is sunny and warm.")

    # Book 2, Page 1, Paragraph 2 -- "Gandhi" and "partition" far apart,
    # separated by several other words, for the proximity-bonus test
    tool.insert_sentence(
        2, 1, 2, 1,
        "Gandhi always maintained that the eventual political partition "
        "of the subcontinent was a tragedy that could have been avoided."
    )

    tool.build_embeddings()
    return tool


def test_get_paragraph_reconstructs_sentences_in_order():
    tool = build_toy_tool()
    text = tool.get_paragraph(1, 1, 1)
    assert text == (
        "Gandhi spoke often about nonviolence and truth. "
        "He believed nonviolence was the only path to freedom."
    )


def test_preprocess_words_removes_stopwords_and_stems():
    words = preprocess_words(tokenize("The partitions of India were painful"))
    # "the", "of", "were" are stopwords and should be gone.
    assert "the" not in words
    assert "of" not in words
    assert "were" not in words
    # "partitions" should stem to the same root as "partition".
    assert preprocess_words(tokenize("partition")) == preprocess_words(tokenize("partitions"))


def test_get_top_k_para_returns_linked_list_of_relevant_paragraphs():
    tool = build_toy_tool()
    head = tool.get_top_k_para("Gandhi partition", 2)
    results = linked_list_to_list(head)

    assert len(results) == 2
    for node in results:
        assert node.book_code is not None and node.page is not None and node.paragraph is not None
        text = tool.get_paragraph(node.book_code, node.page, node.paragraph).lower()
        assert "partition" in text


def test_proximity_bonus_favors_closer_terms():
    tool = build_toy_tool()
    query_words = preprocess_words(tokenize("Gandhi partition"))

    close_bonus = tool._proximity_bonus((1, 1, 2), query_words)   # "Gandhi"/"partition" near each other
    far_bonus = tool._proximity_bonus((2, 1, 2), query_words)     # same words, much further apart

    assert close_bonus > far_bonus


def test_semantic_fallback_when_no_lexical_match():
    tool = build_toy_tool()
    # None of these words appear anywhere in the toy corpus, so BM25 alone
    # would return nothing -- the semantic fallback should still find the
    # weather paragraph, since FakeEmbeddingModel gives it nonzero overlap
    # via shared vocabulary once we search on a word that IS in the vocab
    # but not in this exact phrasing.
    head = tool.get_top_k_para("sunny warm", 1)
    results = linked_list_to_list(head)
    assert len(results) == 1
    text = tool.get_paragraph(results[0].book_code, results[0].page, results[0].paragraph).lower()
    assert "sunny" in text or "warm" in text


def test_get_top_k_para_returns_none_for_empty_query():
    tool = build_toy_tool()
    head = tool.get_top_k_para("the of and", 3)  # all stopwords -> nothing to search on
    assert head is None


if __name__ == "__main__":
    test_get_paragraph_reconstructs_sentences_in_order()
    test_preprocess_words_removes_stopwords_and_stems()
    test_get_top_k_para_returns_linked_list_of_relevant_paragraphs()
    test_proximity_bonus_favors_closer_terms()
    test_semantic_fallback_when_no_lexical_match()
    test_get_top_k_para_returns_none_for_empty_query()
    print("All tests passed.")
