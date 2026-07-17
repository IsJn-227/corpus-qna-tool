"""
qna_tool.py
-----------
Python translation of the C++ interface declared in qna_tool.h.

    C++ (qna_tool.h)                            Python (this file)
    -------------------------------------------  --------------------------------------
    QNA_tool()                                   QNATool()
    ~QNA_tool()                                  (garbage collected automatically)
    void insert_sentence(book_code, page,        insert_sentence(book_code, page,
                          paragraph, sentence_no,               paragraph, sentence_no,
                          sentence)                              sentence)
    Node* get_top_k_para(question, k)            get_top_k_para(question, k) -> Node|None
    string get_paragraph(book_code, page,        get_paragraph(book_code, page,
                          paragraph)                              paragraph) -> str
    void query(question, filename)               query(question, filename)
    void query_llm(filename, root, k,             query_llm(filename, root, k,
                    API_KEY, question)                          api_key, question)

`para_node` in the header (book_code/page/paragraph/score/para_dict) is
represented here implicitly: we key everything off (book_code, page,
paragraph) tuples in dictionaries rather than a separate class + a
`vector<para_node*> p_dict`, since Python dicts already give us O(1)
amortized lookup/insert without needing to hand-roll one (the `Dict`/hashmap
you built in A6). `self.p_dict` is kept as a list of known paragraph keys
for interface parity / debugging.

Design of the scoring, exactly as specified in the assignment:

    s(w)      = (freq_specific_corpus(w) + 1) / (freq_general_corpus(w) + 1)
    score(p)  = sum over query words w of  f_p(w) * s(w)

where f_p(w) is how many times w occurs in paragraph p.
"""

from __future__ import annotations

import heapq
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.node import Node
from src.llm_query import build_prompt, query_llm_backend, SYSTEM_PROMPT

ParaKey = Tuple[int, int, int]  # (book_code, page, paragraph)

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def tokenize(text: str) -> List[str]:
    """Lowercase + extract alphabetic word tokens. Query preprocessing described
    in the assignment ("you will need to pre-process the input query") funnels
    through this same function used at ingestion time, so query words and
    indexed words are normalized identically."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class QNATool:

    def __init__(self, general_freq: Optional[Dict[str, int]] = None):
        # word -> {para_key: count_in_that_paragraph}      (the "Dict *corpus" hashmap)
        self.corpus: Dict[str, Dict[ParaKey, int]] = defaultdict(dict)
        # word -> total occurrences across the WHOLE corpus (needed for s(w))
        self._corpus_word_freq: Dict[str, int] = defaultdict(int)
        # para_key -> {sentence_no: sentence_text}, lets us reconstruct a
        # paragraph in original order regardless of insertion order
        self._para_sentences: Dict[ParaKey, Dict[int, str]] = defaultdict(dict)
        # list of every paragraph key we've seen (interface parity with
        # `vector<para_node*> p_dict;` in the header)
        self.p_dict: List[ParaKey] = []

        # background/general word-frequency table for the s(w) formula;
        # can also be supplied later via set_general_freq(...)
        self.general_freq: Dict[str, int] = general_freq or {}

    def __del__(self):
        # Python's garbage collector reclaims everything automatically;
        # kept only for interface parity with ~QNA_tool().
        pass

    def set_general_freq(self, general_freq: Dict[str, int]) -> None:
        self.general_freq = general_freq

    # ------------------------------------------------------------------
    # Corpus ingestion
    # ------------------------------------------------------------------
    def insert_sentence(self, book_code: int, page: int, paragraph: int, sentence_no: int, sentence: str) -> None:
        """void insert_sentence(int book_code, int page, int paragraph,
                                 int sentence_no, string sentence);

        Called once per sentence in the corpus. Updates the inverted index
        (self.corpus) and stashes the sentence text so get_paragraph() can
        reconstruct the full paragraph later."""
        key: ParaKey = (book_code, page, paragraph)

        if key not in self._para_sentences:
            self.p_dict.append(key)
        self._para_sentences[key][sentence_no] = sentence

        for word in tokenize(sentence):
            self.corpus[word][key] = self.corpus[word].get(key, 0) + 1
            self._corpus_word_freq[word] += 1

    # ------------------------------------------------------------------
    # Section 1: scoring + top-k retrieval
    # ------------------------------------------------------------------
    def _word_score(self, word: str) -> float:
        """s(w) = (freq_specific(w) + 1) / (freq_general(w) + 1)"""
        freq_specific = self._corpus_word_freq.get(word, 0)
        freq_general = self.general_freq.get(word, 0)
        return (freq_specific + 1) / (freq_general + 1)

    def _score_paragraphs(self, query_words: List[str]) -> Dict[ParaKey, float]:
        """score(p) = sum_i f_p(w_i) * s(w_i), computed only over paragraphs
        that actually contain at least one query word (via inverted-index
        lookups), not the whole corpus."""
        scores: Dict[ParaKey, float] = {}
        seen_words = set()
        for w in query_words:
            if w in seen_words:
                continue
            seen_words.add(w)

            s_w = self._word_score(w)
            if s_w <= 0:
                continue

            for key, f_p in self.corpus.get(w, {}).items():
                scores[key] = scores.get(key, 0.0) + f_p * s_w
        return scores

    def get_top_k_para(self, question: str, k: int) -> Optional[Node]:
        """Node* get_top_k_para(string question, int k);

        Preprocesses `question` into words, scores every paragraph that
        matches at least one word, and returns the head of a linked list of
        (up to) k Nodes -- each with book_code / page / paragraph set --
        sorted by descending score. Uses a bounded heap (O(n log k)) instead
        of sorting every matched paragraph (O(n log n)).

        Returns None if there are no matches (Python's version of nullptr)."""
        query_words = tokenize(question)
        if not query_words:
            return None

        scores = self._score_paragraphs(query_words)
        if not scores:
            return None

        top_items = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])

        head: Optional[Node] = None
        tail: Optional[Node] = None
        for (book_code, page, paragraph), score in top_items:
            node = Node(book_code=book_code, page=page, paragraph=paragraph)
            node.score = score
            if head is None:
                head = tail = node
            else:
                tail.next = node
                tail = node
        return head

    # ------------------------------------------------------------------
    # Paragraph reconstruction
    # ------------------------------------------------------------------
    def get_paragraph(self, book_code: int, page: int, paragraph: int) -> str:
        """string get_paragraph(int book_code, int page, int paragraph);
        Searches through the corpus and reconstructs the full paragraph text
        from its stored sentences, in sentence_no order."""
        key: ParaKey = (book_code, page, paragraph)
        sentences = self._para_sentences.get(key)
        if not sentences:
            return ""
        return " ".join(sentences[i] for i in sorted(sentences.keys()))

    # ------------------------------------------------------------------
    # Section 2: querying the LLM
    # ------------------------------------------------------------------
    def query_llm(self, filename: str, root: Optional[Node], k: int, api_key: Optional[str],
                   question: str, backend: str = "anthropic", model: Optional[str] = None) -> str:
        """void query_llm(string filename, Node *root, int k, string API_KEY, string question);

        Walks the linked list `root` (up to k nodes), fetches each
        paragraph's text, builds a grounded prompt, and calls the LLM.
        Writes the answer to `filename` and returns it too. In the original
        C++ version this shells out to api_call.py; since we're already in
        Python we call the LLM API directly (see src/llm_query.py) -- feel
        free to swap this out to demonstrate your own prompt engineering."""
        excerpts = []
        node = root
        count = 0
        while node is not None and count < k:
            text = self.get_paragraph(node.book_code, node.page, node.paragraph)
            excerpts.append((node.book_code, node.page, node.paragraph, text))
            node = node.next
            count += 1

        prompt = build_prompt(question, excerpts)

        # Assignment requirement: "Make sure you print the final query or
        # queries you are sending to the LLM to stdout."
        print("----- Query sent to LLM -----")
        print(prompt)
        print("------------------------------")

        answer = query_llm_backend(SYSTEM_PROMPT, prompt, backend=backend, model=model, api_key=api_key)

        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(answer)

        return answer

    def query(self, question: str, filename: str, k: int = 6, backend: str = "anthropic",
              model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        """void query(string question, string filename);

        The crux of A7: given a question and an output filename, retrieve
        the top-k paragraphs, print the query sent to the LLM to stdout
        (via query_llm), and write the final answer to `filename`."""
        root = self.get_top_k_para(question, k)
        if root is None:
            print(f"No relevant paragraphs found for: {question!r}")
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write("No relevant paragraphs were found in the corpus for this question.")
            return

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.query_llm(filename, root, k, api_key, question, backend=backend, model=model)
