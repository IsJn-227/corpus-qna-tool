"""
corpus_reader.py
-----------------
Generic loader that reads plain-text book files and feeds them into a
QNATool instance sentence-by-sentence via insert_sentence(), matching the
ingestion contract described in qna_tool.h.

Assumed file layout (adjust to match your actual corpus format from A6):

    data/raw_corpus/
        <book_code>_<title>.txt

Within each file:
    - Pages separated by a form-feed character '\\x0c' OR the literal
      marker '[PAGE]' on its own line. If neither is present, the whole
      file is treated as a single page (page = 1).
    - Paragraphs separated by one or more blank lines.
    - Sentences within a paragraph are split on '.', '!', '?' followed by
      whitespace (a simple heuristic -- swap in nltk.sent_tokenize for
      better accuracy if abbreviations are causing bad splits).

book_code is taken from the leading integer in the filename
(e.g. "12_satyagraha_in_south_africa.txt" -> book_code = 12). If the
filename has no leading integer, book_code is assigned in sorted-filename
order starting from 0.
"""

from __future__ import annotations

import os
import re
from typing import Iterator, Tuple

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LEADING_INT_RE = re.compile(r"^(\d+)")


def _book_code_from_filename(filename: str, fallback: int) -> int:
    m = _LEADING_INT_RE.match(filename)
    return int(m.group(1)) if m else fallback


def _split_pages(raw_text: str):
    if "\x0c" in raw_text:
        return raw_text.split("\x0c")
    if "[PAGE]" in raw_text:
        return raw_text.split("[PAGE]")
    return [raw_text]


def _split_paragraphs(page_text: str):
    page_text = page_text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = re.split(r"\n\s*\n", page_text)
    return [c.strip() for c in chunks if c.strip()]


def _split_sentences(paragraph_text: str):
    paragraph_text = " ".join(paragraph_text.split())
    sentences = _SENTENCE_SPLIT_RE.split(paragraph_text)
    return [s.strip() for s in sentences if s.strip()]


def iter_corpus_sentences(corpus_dir: str) -> Iterator[Tuple[int, int, int, int, str]]:
    """Yields (book_code, page, paragraph, sentence_no, sentence) for every
    sentence in every .txt book found under corpus_dir."""
    filenames = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".txt"))

    for fallback_code, filename in enumerate(filenames):
        book_code = _book_code_from_filename(filename, fallback_code)
        path = os.path.join(corpus_dir, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            raw_text = fh.read()

        for page_no, page_text in enumerate(_split_pages(raw_text), start=1):
            for para_no, para_text in enumerate(_split_paragraphs(page_text), start=1):
                for sent_no, sentence in enumerate(_split_sentences(para_text), start=1):
                    yield book_code, page_no, para_no, sent_no, sentence


def load_into_tool(tool, corpus_dir: str) -> None:
    """Reads corpus_dir and calls tool.insert_sentence(...) for every sentence."""
    for book_code, page, paragraph, sentence_no, sentence in iter_corpus_sentences(corpus_dir):
        tool.insert_sentence(book_code, page, paragraph, sentence_no, sentence)
