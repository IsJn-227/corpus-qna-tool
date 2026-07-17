# Corpus Q&A Tool (Python translation of `qna_tool.h`)

A Python implementation of the Assignment 7 interface: given a corpus fed in
sentence-by-sentence (as `(book_code, page, paragraph, sentence_no,
sentence)`), find the top-k most relevant paragraphs for a natural-language
query, then feed them to an LLM to synthesize a grounded, cited answer.

This mirrors the C++ header you were given almost 1:1:

| `qna_tool.h` (C++)                                                | This project (Python)                            |
|---------------------------------------------------------------------|---------------------------------------------------|
| `QNA_tool()` / `~QNA_tool()`                                          | `QNATool()` (GC'd automatically)                   |
| `insert_sentence(book_code, page, paragraph, sentence_no, s)`        | `QNATool.insert_sentence(...)` — same signature    |
| `Node* get_top_k_para(question, k)`                                   | `QNATool.get_top_k_para(question, k) -> Node`      |
| `string get_paragraph(book_code, page, paragraph)`                    | `QNATool.get_paragraph(...) -> str`                |
| `void query(question, filename)`                                     | `QNATool.query(question, filename)`                |
| `void query_llm(filename, root, k, API_KEY, question)` (private)     | `QNATool.query_llm(...)` (called internally)       |
| `Node` (linked list: book_code/page/paragraph/next)                   | `src/node.py: Node` — same fields, real `.next`    |

`get_top_k_para` genuinely returns a **linked list** (a `Node` with
`.next`), not a Python list, so the shape of the interface -- and the code
you'd write to walk it in `query_llm` -- matches the original assignment.

## How it works

```
insert_sentence(...) called once per sentence while loading the corpus
        |
        v
inverted index (word -> {(book_code,page,paragraph): count})   [QNATool.corpus]
sentence store (para_key -> {sentence_no: text})                [QNATool._para_sentences]
        |
        v  get_top_k_para(question, k)
tokenize question -> look up each word in the inverted index -> score every
matching paragraph:
        s(w)     = (freq_corpus(w) + 1) / (freq_general(w) + 1)
        score(p) = sum f_p(w) * s(w)
take top-k with a bounded heap (O(n log k)) -> build Node linked list
        |
        v  query(question, filename)  ->  query_llm(...)
walk the linked list, reconstruct each paragraph's text via get_paragraph(),
build a grounded, citation-requiring prompt, call the LLM, print the query
to stdout, write the answer to `filename`
```

### Files

```
src/
├── node.py            # Node class: book_code, page, paragraph, score, next
├── corpus_reader.py   # reads raw book files -> calls insert_sentence per sentence
│                       # (ADAPT this to your actual A6 corpus file format --
│                       #  see the docstring at the top of the file)
├── general_freq.py    # loads the background word-frequency CSV for s(w)
├── qna_tool.py         # QNATool class: the direct translation of qna_tool.h
├── llm_query.py        # prompt construction + LLM backends (Anthropic/OpenAI/HF stub)
└── cli.py              # command-line driver tying it all together
tests/
└── test_qna_tool.py    # tests against a tiny hand-built corpus, no files/API needed
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. **Corpus ingestion.** The header expects `insert_sentence(book_code,
   page, paragraph, sentence_no, sentence)` to be called once per sentence.
   `src/corpus_reader.py` is a generic driver that reads plain-text `.txt`
   files from `data/raw_corpus/` and splits them into pages -> paragraphs ->
   sentences with simple heuristics (form-feed / `[PAGE]` markers for pages,
   blank lines for paragraphs, `.!?` for sentences). **If your actual A6
   corpus format is different** (e.g. a structured file the course gave you
   with book/page/paragraph/sentence numbers already laid out), skip this
   file and call `tool.insert_sentence(...)` directly from that structured
   data instead -- that's the intended extension point.
2. Get a general-English word-frequency CSV (e.g. `unigram_freq.csv`,
   derived from the Google Web Trillion Word Corpus, freely available on
   Kaggle) and save it as `data/general_freq.csv` with columns `word,count`.
3. Copy `.env.example` to `.env`, fill in your `ANTHROPIC_API_KEY` and/or
   `OPENAI_API_KEY`, and export them into your shell.

## Usage

```bash
# Retrieval only (Part 1), no LLM call / API key needed:
python -m src.cli --query "Gandhi's views on the partition of India" --k 8 --no-llm

# Full pipeline (Part 2): retrieves top-k, prints the exact prompt sent to
# the LLM to stdout, writes the answer to answer.txt:
python -m src.cli --query "Gandhi's views on the partition of India" --k 8 \
    --backend anthropic --out answer.txt
```

Or use `QNATool` directly in your own script, exactly like the C++ version
would be driven from a `main.cpp`:

```python
from src.qna_tool import QNATool
from src.general_freq import load_general_freq

general_freq = load_general_freq("data/general_freq.csv")
tool = QNATool(general_freq=general_freq)

# however you parse your corpus, call insert_sentence once per sentence:
tool.insert_sentence(book_code=3, page=42, paragraph=2, sentence_no=1,
                      sentence="Gandhi began his fast on the 12th of January.")
# ... (repeat for every sentence in the corpus)

tool.query("What was Gandhi's state of mind around Independence?", "answer.txt", k=8)
```

Run the tests (no API keys or real corpus required):

```bash
python -m pytest tests/
# or, without pytest installed:
python tests/test_qna_tool.py
```

## Design notes / viva talking points

**Why dicts instead of a custom `Dict`/hashmap class from A6?**
Python's built-in `dict` *is* a hash table with O(1) amortized
get/insert/delete -- reimplementing open addressing / chaining by hand in
Python would add code without adding correctness or much speed.
`QNATool.corpus` (word -> paragraph -> count) plays exactly the role your A6
`Dict` played in the C++ version; if your viva wants you to show a
hand-rolled hash table, it's a drop-in swap for `defaultdict` in
`qna_tool.py`.

**Why a real linked list for `get_top_k_para`?**
The header's signature (`Node* get_top_k_para(...)`) strongly implies the
grader/viva may walk the returned structure as a linked list (e.g. inside
`query_llm`). `src/node.py`'s `Node` has a genuine `.next` pointer and
`get_top_k_para` builds an honest singly-linked list, not a Python list
dressed up to look like one -- so translating this back to C++, or
explaining it in the viva, is straightforward.

**Choosing k.** More is not always better: paragraphs past the top few tend
to be lower-scoring and often tangential, diluting the LLM's context and
costing more tokens for no benefit. k in the 5-10 range is a reasonable
starting point; narrow queries need fewer, broad ones need more. Worth
tuning per corpus and reporting in your writeup.

**Is the retrieval algorithm optimal? Can it be better?**
The current scoring is bag-of-words: no notion of synonyms or paraphrase, so
a paragraph expressing the same idea in different words is invisible to it.
A natural two-stage improvement: retrieve a larger candidate set (say k=50)
cheaply via the inverted index as now, then **rerank** by embedding
similarity (sentence-transformers or an embeddings API) before truncating to
the final k -- combining the speed of hashmap lookups with semantic recall.
Another improvement: swap the `s(w)`/`score(p)` formulas for **BM25**, which
additionally normalizes for paragraph length and saturates term-frequency
contributions -- a drop-in change to `QNATool._word_score` /
`QNATool._score_paragraphs`.

**Prompt engineering.** See `llm_query.SYSTEM_PROMPT`: forbids outside
knowledge (reduces hallucination), requires per-claim citation of
`(book_code, page)`, and asks the model to surface disagreement across
sources instead of flattening decades of Gandhi's (or Ramana's) writing into
one voice.

**Multiple / open-source LLMs.** `llm_query.BACKENDS` is a dict of
pluggable backends specifically so you can run the same retrieved context
through more than one model and compare answers, or fill in
`_call_huggingface_local` to try a free open-source model against
Claude/ChatGPT on identical evidence.
