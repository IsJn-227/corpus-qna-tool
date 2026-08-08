"""
cli.py
------
Command-line entry point wiring: corpus_reader -> QNATool -> query().

Usage examples:

    # Retrieval only, no LLM call needed (good for testing retrieval alone):
    python -m src.cli --query "Gandhi's views on the partition of India" --k 8 --no-llm

    # Full pipeline, writing the LLM's answer to answer.txt:
    python -m src.cli --query "Gandhi's views on the partition of India" --k 8 \
        --backend anthropic --out answer.txt

Note: building embeddings (tool.build_embeddings()) downloads and runs a
sentence-transformers model on every paragraph, so indexing a large corpus
will take noticeably longer than the old BM25/frequency-only version. For a
98-book corpus, expect this step to dominate runtime.
"""

from __future__ import annotations

import argparse
import os

from src.corpus_reader import load_into_tool
from src.node import linked_list_to_list
from src.qna_tool import QNATool

DEFAULT_CORPUS_DIR = "data/raw_corpus"


def main():
    parser = argparse.ArgumentParser(description="Corpus Q&A tool (BM25 + semantic retrieval + LLM).")
    parser.add_argument("--query", required=True, help="The question to ask the corpus.")
    parser.add_argument("--k", type=int, default=6, help="Number of top paragraphs to retrieve.")
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--no-llm", action="store_true", help="Only show retrieved paragraphs; skip the LLM call.")
    parser.add_argument("--backend", default="anthropic", choices=["anthropic", "openai", "huggingface"])
    parser.add_argument("--model", default=None, help="Override the default model for the chosen backend.")
    parser.add_argument("--out", default="answer.txt", help="File to write the LLM's answer to.")
    args = parser.parse_args()

    if not os.path.isdir(args.corpus_dir) or not any(f.endswith(".txt") for f in os.listdir(args.corpus_dir)):
        raise SystemExit(
            f"No corpus found at '{args.corpus_dir}'. Drop .txt book files there first "
            f"(see README.md)."
        )

    print("Building index from corpus (insert_sentence per sentence)...")
    tool = QNATool()
    load_into_tool(tool, args.corpus_dir)
    print(f"  {len(tool.p_dict)} paragraphs indexed.")

    print("Building paragraph embeddings (this downloads/runs a sentence-transformers model)...")
    tool.build_embeddings()

    print(f"\nSearching for: {args.query!r}\n")
    head = tool.get_top_k_para(args.query, args.k)
    results = linked_list_to_list(head)

    if not results:
        print("No matching paragraphs found.")
        return

    print(f"Top {len(results)} paragraphs:\n" + "=" * 60)
    for i, node in enumerate(results, start=1):
        text = tool.get_paragraph(node.book_code, node.page, node.paragraph)
        preview = text if len(text) <= 400 else text[:400] + "..."
        print(f"\n#{i} | score={node.score:.3f} | book={node.book_code} page={node.page} para={node.paragraph}\n{preview}")
    print("\n" + "=" * 60)

    if args.no_llm:
        return

    print(f"\nSending to LLM backend '{args.backend}', writing answer to '{args.out}'...\n")
    tool.query(args.query, args.out, k=args.k, backend=args.backend, model=args.model)
    print(f"\nDone. Answer written to {args.out}")


if __name__ == "__main__":
    main()
