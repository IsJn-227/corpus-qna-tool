"""
qna_tool.py
-----------
Python translation of the C++ interface declared in qna_tool.h, now upgraded
to a hybrid retrieval scorer:

    - BM25 (Okapi) term scoring, replacing the assignment's simple
      (freq_specific+1)/(freq_general+1) ratio. BM25 additionally accounts
      for paragraph length and saturates term-frequency contributions, so a
      paragraph that repeats a word many times doesn't dominate.
    - Stopword removal + Porter stemming at both index and query time, so
      "partitioned"/"partition"/"partitions" all collapse to one term.
    - A proximity bonus: query terms that appear close together within a
      paragraph score higher than the same terms scattered far apart.
    - A semantic layer via sentence embeddings (all-MiniLM-L6-v2), so a
      paragraph that expresses the same idea in different words can still
      be found -- something pure keyword matching (including BM25) misses
      entirely. If no paragraph contains ANY query term, we fall back to
      semantic-only retrieval over the whole corpus instead of returning
      nothing.

    C++ (qna_tool.h)                            Python (this file)
    -------------------------------------------  --------------------------------------
    QNA_tool()                                   QNATool()
    ~QNA_tool()                                  (garbage collected automatically)
    void insert_sentence(...)                    insert_sentence(...)
    Node* get_top_k_para(question, k)            get_top_k_para(question, k) -> Node|None
    string get_paragraph(...)                    get_paragraph(...) -> str
    void query(question, filename)                query(question, filename)
    void query_llm(filename, root, k,             query_llm(filename, root, k,
                    API_KEY, question)                          api_key, question)
"""

from __future__ import annotations

import heapq
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from nltk.stem import PorterStemmer

from src.node import Node
from src.llm_query import build_prompt, query_llm_backend, SYSTEM_PROMPT

ParaKey = Tuple[int, int, int]  # (book_code, page, paragraph)

_TOKEN_RE = re.compile(r"[A-Za-z]+")

STOPWORDS = {
    "a", "an", "the",
    "is", "are", "was", "were",
    "am", "be", "been", "being",
    "of", "to", "in", "on", "at",
    "for", "from", "by", "with",
    "and", "or", "but",
    "that", "this", "these", "those",
    "it", "its",
    "as", "if", "than", "then",
    "into", "about", "over", "under",
    "what", "which", "who", "whom",
    "why", "how", "when", "where",
}

_stemmer = PorterStemmer()

# Weights for combining BM25 + proximity + semantic signals into one score.
# Pulled out as constants so they're easy to tune/explain in a viva.
PROXIMITY_WEIGHT = 0.5
SEMANTIC_WEIGHT = 1.5
SHORT_PARAGRAPH_WORD_THRESHOLD = 5
SHORT_PARAGRAPH_PENALTY = 0.2  # multiply score by this if paragraph is very short

DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def tokenize(text: str) -> List[str]:
    """Lowercase + extract alphabetic word tokens."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def preprocess_words(words: List[str]) -> List[str]:
    """Stopword removal + Porter stemming, applied identically at ingestion
    time (insert_sentence) and query time (get_top_k_para) so indexed terms
    and query terms are normalized the same way."""
    return [_stemmer.stem(w) for w in words if w not in STOPWORDS]


class QNATool:

    def __init__(self, embedding_model=None, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
                 k1: float = 1.5, b: float = 0.75):
        """
        embedding_model: optionally inject a pre-built embedding model (must
            expose .encode(text, convert_to_numpy=True, normalize_embeddings=True)
            -> np.ndarray). Mainly useful for tests, so they can pass a tiny
            fake encoder instead of downloading the real ~90MB model. If not
            given, the real SentenceTransformer model is lazily loaded on
            first use (so simply constructing a QNATool, or running BM25-only
            retrieval, never triggers a download).
        embedding_model_name: HuggingFace model name used when lazily
            loading the real SentenceTransformer, if embedding_model wasn't
            injected.
        k1, b: standard BM25 hyperparameters (term-frequency saturation and
            length-normalization strength respectively).
        """
        # word -> {para_key: [positions of word within that paragraph]}
        # (the "Dict *corpus" hashmap; storing positions, not just counts,
        # is what makes the proximity bonus possible)
        self.corpus: Dict[str, Dict[ParaKey, List[int]]] = defaultdict(dict)

        # para_key -> {sentence_no: sentence_text}
        self._para_sentences: Dict[ParaKey, Dict[int, str]] = defaultdict(dict)

        # list of every paragraph key we've seen (interface parity with
        # `vector<para_node*> p_dict;` in the header)
        self.p_dict: List[ParaKey] = []

        # BM25 bookkeeping
        self.total_paragraphs = 0
        self.para_length: Dict[ParaKey, int] = {}
        self.k1 = k1
        self.b = b

        # Semantic layer (lazy-loaded so constructing a QNATool never forces
        # a model download unless you actually call build_embeddings() /
        # get_top_k_para())
        self._embedding_model = embedding_model
        self._embedding_model_name = embedding_model_name
        self.para_embeddings: Dict[ParaKey, np.ndarray] = {}

    def __del__(self):
        pass

    # ------------------------------------------------------------------
    # Embedding model (lazy)
    # ------------------------------------------------------------------
    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _encode(self, text: str) -> np.ndarray:
        model = self._get_embedding_model()
        return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def _average_para_length(self) -> float:
        if self.total_paragraphs == 0:
            return 0.0
        return sum(self.para_length.values()) / self.total_paragraphs

    # ------------------------------------------------------------------
    # Corpus ingestion
    # ------------------------------------------------------------------
    def insert_sentence(self, book_code: int, page: int, paragraph: int, sentence_no: int, sentence: str) -> None:
        """void insert_sentence(int book_code, int page, int paragraph,
                                 int sentence_no, string sentence);

        Called once per sentence in the corpus. Updates the inverted index
        (self.corpus, storing word positions for the proximity bonus) and
        stashes the sentence text so get_paragraph() can reconstruct the
        full paragraph later. Does NOT build embeddings -- call
        build_embeddings() once after all sentences are inserted."""
        key: ParaKey = (book_code, page, paragraph)
        words = preprocess_words(tokenize(sentence))

        if key not in self._para_sentences:
            self.p_dict.append(key)
        self._para_sentences[key][sentence_no] = sentence

        for position, word in enumerate(words):
            if key not in self.corpus[word]:
                self.corpus[word][key] = []
            self.corpus[word][key].append(position)

        if key not in self.para_length:
            self.total_paragraphs += 1
            self.para_length[key] = 0
        self.para_length[key] += len(words)

    def build_embeddings(self) -> None:
        """Build one embedding per paragraph in the corpus. Call this once
        after all insert_sentence() calls have finished, before running any
        queries (get_top_k_para relies on self.para_embeddings for both the
        semantic-score term and the no-lexical-match fallback)."""
        self.para_embeddings.clear()

        for book_code, page, paragraph in self.p_dict:
            text = self.get_paragraph(book_code, page, paragraph)
            if not text.strip():
                continue
            self.para_embeddings[(book_code, page, paragraph)] = self._encode(text)

    # ------------------------------------------------------------------
    # Section 1: scoring + top-k retrieval
    # ------------------------------------------------------------------
    def _bm25_score(self, word: str, para_key: ParaKey, tf: int, avgdl: float) -> float:
        """Okapi BM25 term score for one (word, paragraph) pair."""
        if avgdl == 0:
            return 0.0

        df = len(self.corpus.get(word, {}))
        N = self.total_paragraphs
        dl = self.para_length[para_key]

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        return idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))

    def _semantic_score(self, para_key: ParaKey, query_embedding: np.ndarray) -> float:
        """Cosine similarity (both vectors are pre-normalized, so a plain
        dot product suffices) between the query and paragraph embeddings."""
        paragraph_embedding = self.para_embeddings.get(para_key)
        if paragraph_embedding is None:
            return 0.0
        return float(np.dot(query_embedding, paragraph_embedding))

    def _proximity_bonus(self, para_key: ParaKey, query_words: List[str]) -> float:
        """Rewards paragraphs where consecutive query words appear close
        together (measured over each adjacent pair of DISTINCT query
        words), not just where they both happen to occur somewhere."""
        seen = set()
        distinct_words = []
        for word in query_words:
            if word not in seen:
                seen.add(word)
                distinct_words.append(word)

        if len(distinct_words) < 2:
            return 0.0

        total_distance = 0
        pair_count = 0

        for i in range(len(distinct_words) - 1):
            word1, word2 = distinct_words[i], distinct_words[i + 1]

            if (word1 not in self.corpus or para_key not in self.corpus[word1]
                    or word2 not in self.corpus or para_key not in self.corpus[word2]):
                continue

            positions1 = self.corpus[word1][para_key]
            positions2 = self.corpus[word2][para_key]

            min_distance = min(abs(p1 - p2) for p1 in positions1 for p2 in positions2)
            total_distance += min_distance
            pair_count += 1

        if pair_count == 0:
            return 0.0

        avg_distance = total_distance / pair_count
        return 1.0 / (1.0 + avg_distance)

    def _score_paragraphs(self, query_words: List[str], query_embedding: np.ndarray) -> Dict[ParaKey, float]:
        """Combined score = BM25 (summed over query terms) + proximity bonus
        + semantic similarity, with a small penalty for very short
        paragraphs (which can otherwise score artificially high on BM25 due
        to length normalization). If NO paragraph contains any query term,
        falls back to semantic-only retrieval over the whole corpus."""
        scores: Dict[ParaKey, float] = {}
        seen_words = set()
        avgdl = self._average_para_length()

        for w in query_words:
            if w in seen_words:
                continue
            seen_words.add(w)

            for key, position_list in self.corpus.get(w, {}).items():
                tf = len(position_list)
                bm25 = self._bm25_score(w, key, tf, avgdl)
                scores[key] = scores.get(key, 0.0) + bm25

        for key in scores:
            score = (
                scores[key]
                + PROXIMITY_WEIGHT * self._proximity_bonus(key, query_words)
                + SEMANTIC_WEIGHT * self._semantic_score(key, query_embedding)
            )

            book_code, page, paragraph_no = key
            paragraph_text = self.get_paragraph(book_code, page, paragraph_no)
            if len(paragraph_text.split()) < SHORT_PARAGRAPH_WORD_THRESHOLD:
                score *= SHORT_PARAGRAPH_PENALTY

            scores[key] = score

        if not scores:
            # No lexical matches anywhere in the corpus -- fall back to
            # semantic-only retrieval instead of returning nothing.
            for key in self.para_embeddings:
                scores[key] = SEMANTIC_WEIGHT * self._semantic_score(key, query_embedding)

        return scores

    def get_top_k_para(self, question: str, k: int) -> Optional[Node]:
        """Node* get_top_k_para(string question, int k);

        Preprocesses `question` (tokenize -> stopword removal -> stemming),
        computes an embedding for it, scores every candidate paragraph via
        _score_paragraphs, and returns the head of a linked list of (up to)
        k Nodes sorted by descending score. Uses a bounded heap
        (O(n log k)) instead of sorting every candidate.

        Returns None if there are no candidate paragraphs at all (e.g. an
        empty corpus or embeddings never built)."""
        query_words = preprocess_words(tokenize(question))
        if not query_words:
            return None

        query_embedding = self._encode(question)

        scores = self._score_paragraphs(query_words, query_embedding)
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
        Reconstructs the full paragraph text from its stored sentences, in
        sentence_no order."""
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
        """void query_llm(string filename, Node *root, int k, string API_KEY, string question);"""
        excerpts = []
        node = root
        count = 0
        while node is not None and count < k:
            text = self.get_paragraph(node.book_code, node.page, node.paragraph)
            excerpts.append((node.book_code, node.page, node.paragraph, text))
            node = node.next
            count += 1

        prompt = build_prompt(question, excerpts)

        print("----- Query sent to LLM -----")
        print(prompt)
        print("------------------------------")

        answer = query_llm_backend(SYSTEM_PROMPT, prompt, backend=backend, model=model, api_key=api_key)

        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(answer)

        return answer

    def query(self, question: str, filename: str, k: int = 6, backend: str = "anthropic",
              model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        """void query(string question, string filename);"""
        root = self.get_top_k_para(question, k)
        if root is None:
            print(f"No relevant paragraphs found for: {question!r}")
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write("No relevant paragraphs were found in the corpus for this question.")
            return

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.query_llm(filename, root, k, api_key, question, backend=backend, model=model)
