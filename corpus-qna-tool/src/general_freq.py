"""
general_freq.py
----------------
Loads a CSV of general-English word frequencies, used as the "background"
corpus frequency in the scoring formula:

    s(w) = (freq_specific_corpus(w) + 1) / (freq_general_corpus(w) + 1)

A good, freely available source for this is the "unigram_freq.csv" dataset
(word frequencies derived from the Google Web Trillion Word Corpus), which
has exactly two columns: `word,count`. Drop such a file at
`data/general_freq.csv` and this loader will pick it up.

If you were given a different CSV by your course (different column names),
adjust `word_col` / `count_col` below.
"""

from __future__ import annotations

import csv
from typing import Dict


def load_general_freq(csv_path: str, word_col: str = "word", count_col: str = "count") -> Dict[str, int]:
    freq: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        # Fall back gracefully if the CSV has no header (just word,count rows).
        if reader.fieldnames is None or word_col not in reader.fieldnames:
            fh.seek(0)
            raw_reader = csv.reader(fh)
            for row in raw_reader:
                if len(row) < 2:
                    continue
                word, count = row[0], row[1]
                try:
                    freq[word.strip().lower()] = int(count)
                except ValueError:
                    continue
            return freq

        for row in reader:
            word = row[word_col].strip().lower()
            try:
                freq[word] = int(row[count_col])
            except (ValueError, KeyError):
                continue
    return freq
