# Corpus Q&A Tool (Python translation of `qna_tool.h`)

A Python implementation of the Assignment 7 interface: given a corpus fed in
sentence-by-sentence (as `(book_code, page, paragraph, sentence_no,
sentence)`), find the top-k most relevant paragraphs for a natural-language
query, then feed them to an LLM to synthesize a grounded, cited answer.

Retrieval is a **hybrid scorer**, extending well past the assignment's
original simple frequency-ratio formula:

- **BM25** (Okapi) term scoring instead of a plain frequency ratio --
  accounts for paragraph length and saturates term-frequency contributions,
  so a paragraph that just repeats a word many times doesn't dominate.
- **Stopword removal + Porter stemming**, applied identically at indexing
  and query time, so "partition"/"partitions"/"partitioned" collapse to one
  term and function words ("the", "of", "and", ...) don't pollute scores.
- **A proximity bonus**: paragraphs where the query's terms appear close
  together score higher than paragraphs where the same terms are scattered
  far apart.
- **Semantic embeddings** (`sentence-transformers`, `all-MiniLM-L6-v2`):
  every paragraph and every query is embedded, and cosine similarity is
  blended into the score. This is what lets the tool find a paragraph that
  expresses the same idea in *different words* -- something pure keyword
  matching (BM25 included) structurally cannot do.
- **Semantic fallback**: if literally no paragraph contains any query term,
  the tool falls back to semantic-only retrieval over the whole corpus
  instead of returning nothing.

## Interface mapping

| `qna_tool.h` (C++)                                                | This project (Python)                            |
|---------------------------------------------------------------------|-----------------------------------------------------|
| `QNA_tool()` / `~QNA_tool()`                                          | `QNATool()` (GC'd automatically)                   |
| `insert_sentence(book_code, page, paragraph, sentence_no, s)`        | `QNATool.insert_sentence(...)` -- same signature   |
| `Node* get_top_k_para(question, k)`                                   | `QNATool.get_top_k_para(question, k) -> Node`      |
| `string get_paragraph(book_code, page, paragraph)`                    | `QNATool.get_paragraph(...) -> str`                |
| `void query(question, filename)`                                     | `QNATool.query(question, filename)`                |
| `void query_llm(filename, root, k, API_KEY, question)` (private)     | `QNATool.query_llm(...)` (called internally)       |
| `Node` (linked list: book_code/page/paragraph/next)                   | `src/node.py: Node` -- same fields, real `.next`   |

`get_top_k_para` genuinely returns a **linked list** (a `Node` with
`.next`), not a Python list.

## How it works

```
insert_sentence(...) called once per sentence while loading the corpus
        |
        v
inverted index: word -> {(book_code,page,paragraph): [positions of word]}
sentence store: para_key -> {sentence_no: text}
        |
        v  build_embeddings()  (call once, after all sentences are inserted)
one sentence-transformer embedding computed per paragraph
        |
        v  get_top_k_para(question, k)
question -> tokenize -> remove stopwords -> stem -> embed
for every paragraph containing >=1 query term:
    score = BM25(query terms, paragraph)
          + 0.5 * proximity_bonus(query terms, paragraph)
          + 1.5 * cosine_similarity(query embedding, paragraph embedding)
    (short paragraphs get a small penalty to counter BM25 length bias)
if NO paragraph contains any query term:
    score = 1.5 * cosine_similarity(...) over every paragraph  (semantic-only fallback)
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
├── qna_tool.py         # QNATool class: BM25 + proximity + semantic hybrid scorer
├── llm_query.py        # prompt construction + LLM backends (Anthropic/OpenAI/HF stub)
└── cli.py              # command-line driver tying it all together
tests/
└── test_qna_tool.py    # tests against a tiny hand-built corpus, no model download needed
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** `sentence-transformers` pulls in `torch`, so this install is
noticeably heavier (~1-2GB) than a BM25-only setup. If you want to avoid
that dependency entirely, you can inject a lightweight custom
`embedding_model` (any object exposing `.encode(text, ...) -> np.ndarray`)
into `QNATool(embedding_model=...)`, or strip the semantic layer out of
`_score_paragraphs` and rely on BM25 + proximity alone.

1. **Corpus ingestion.** The header expects `insert_sentence(book_code,
   page, paragraph, sentence_no, sentence)` to be called once per sentence.
   `src/corpus_reader.py` is a generic driver that reads plain-text `.txt`
   files from `data/raw_corpus/` and splits them into pages -> paragraphs ->
   sentences with simple heuristics. **If your actual A6 corpus format is
   different**, skip this file and call `tool.insert_sentence(...)` directly
   from your own parsed data instead.
2. Copy `.env.example` to `.env`, fill in your `ANTHROPIC_API_KEY` and/or
   `OPENAI_API_KEY`, and export them into your shell.

## Usage

```bash
# Retrieval only (no LLM call / API key needed):
python -m src.cli --query "Gandhi's views on the partition of India" --k 8 --no-llm

# Full pipeline: retrieves top-k, prints the exact prompt sent to the LLM to
# stdout, writes the answer to answer.txt:
python -m src.cli --query "Gandhi's views on the partition of India" --k 8 \
    --backend anthropic --out answer.txt
```

Or use `QNATool` directly in your own script:

```python
from src.qna_tool import QNATool

tool = QNATool()  # lazily loads the embedding model on first use

# however you parse your corpus, call insert_sentence once per sentence:
tool.insert_sentence(book_code=3, page=42, paragraph=2, sentence_no=1,
                      sentence="Gandhi began his fast on the 12th of January.")
# ... (repeat for every sentence in the corpus)

tool.build_embeddings()  # call once, after all sentences are inserted

tool.query("What was Gandhi's state of mind around Independence?", "answer.txt", k=8)
```

Run the tests (fast -- no model download, no API keys, no real corpus
needed; a small fake embedding model is injected instead of the real one):

```bash
python -m pytest tests/
# or, without pytest installed:
python tests/test_qna_tool.py
```

## Design notes / viva talking points

**Why BM25 instead of the assignment's frequency-ratio formula?**
The original `s(w) = (freq_specific+1)/(freq_general+1)` scoring has no
notion of paragraph length, so a long paragraph that happens to contain a
query word once scores the same per-occurrence as a short, tightly-focused
one. BM25 fixes this with length normalization (`b`) and diminishing
returns on repeated terms (`k1`), which is why it's the standard scoring
function behind most production keyword search (including Elasticsearch's
default).

**Why add semantic embeddings on top of BM25?**
BM25 (and the original formula) are both purely lexical: a paragraph that
discusses "the division of the subcontinent" is invisible to a query about
"partition" unless the exact word appears. Embedding-based cosine
similarity captures meaning rather than exact wording, closing that gap --
at the cost of being fuzzier and more expensive to compute, hence combining
both rather than replacing one with the other.

**Why the proximity bonus?**
BM25 sums independent per-term scores, so a paragraph mentioning "Gandhi"
in one sentence and "partition" three paragraphs later scores identically
to one where they appear in the same sentence, discussing the same event.
The proximity bonus differentiates these using the token *positions* stored
in the inverted index.

**Why fall back to semantic-only retrieval?**
Without it, a query using different vocabulary than the corpus (e.g. asking
about "the 1947 split of British India" when the corpus always says
"partition") could return nothing at all under BM25, even though clearly
relevant paragraphs exist. The fallback trades precision for recall in that
specific (no-lexical-match) case.

**Choosing k.** More is not always better: paragraphs past the top few tend
to be lower-scoring and often tangential, diluting the LLM's context and
costing more tokens for no benefit. k in the 5-10 range is a reasonable
starting point.

**Prompt engineering.** See `llm_query.SYSTEM_PROMPT`: forbids outside
knowledge (reduces hallucination), requires per-claim citation of
`(book_code, page)`, and asks the model to surface disagreement across
sources instead of flattening decades of writing into one voice.

**Further extensions.** `llm_query.BACKENDS` is a dict of pluggable
backends so you can run the same retrieved context through more than one
model and compare answers, or fill in `_call_huggingface_local` to try a
free open-source model against Claude/ChatGPT on identical evidence.
