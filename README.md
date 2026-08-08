# Corpus Q&A Tool (Python translation of `qna_tool.h`)

A Python implementation of given corpus fed in
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

