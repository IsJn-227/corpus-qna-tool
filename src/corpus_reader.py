"""
corpus_reader.py
-----------------
The header's ingestion contract is:

    void insert_sentence(int book_code, int page, int paragraph,
                          int sentence_no, string sentence);

called once per sentence in the corpus -- presumably by a driver you (or the
provided A7 harness) already have from A6, which parses the raw book files
into (book_code, page, paragraph, sentence_no, sentence) tuples.

This module is a generic, adjustable version of that driver, since we don't
have your exact A6 corpus file format. It reads plain-text book files and
splits them into pages -> paragraphs -> sentences using simple heuristics.
ADAPT THE SPLITTING LOGIC BELOW to match whatever format your actual corpus
files (the 98 Gandhi books / Ramana books) are provided in -- e.g. if the
grader gives you a CSV/JSON with book_code,page,paragraph,sentence_no,sentence
already laid out, skip this file entirely and call
QNATool.insert_sentence(...) directly from that structured data instead.

Assumed plain-text layout (change as needed):
    data/raw_corpus/
        <book_code>_<title>.txt        # book_code parsed from leading digits,
                                        # else assigned by sorted file order

    Within a file:
      - Pages separated by a form-feed character '\\x0c', or the literal
        marker '[PAGE]' on its own line. If neither appears, the whole file
        is one page.
      - Paragraphs separated by one or more blank lines.
      - Sentences split on '.', '!', '?' followed by whitespace. This is a
        simple heuristic; swap in `nltk.sent_tokenize` for better accuracy
        if you have nltk available and abbreviations are causing bad splits.
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
