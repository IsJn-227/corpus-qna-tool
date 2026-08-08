"""
llm_query.py
------------
Prompt construction + LLM backends, used by QNATool.query_llm() /
QNATool.query() (see src/qna_tool.py) to implement Part 2: "Querying the
LLM".
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

Excerpt = Tuple[int, int, int, str]

SYSTEM_PROMPT = """You are a careful research assistant answering questions about a specific \
text corpus using ONLY the excerpts provided to you below. \
Do not use any outside knowledge or make up information. \
If the excerpts do not contain enough information to answer the question, say so plainly \
rather than guessing. \
When you use information from an excerpt, cite it as (Book book_code, Page page). \
Write a clear, well-organized answer in your own words -- do not just copy the excerpts. \
Where the excerpts show differing or evolving views, point that out rather than \
flattening them into one answer."""


def build_prompt(question: str, excerpts: List[Excerpt]) -> str:
    blocks = []
    for i, (book_code, page, paragraph, text) in enumerate(excerpts, start=1):
        blocks.append(
            f"[Excerpt {i} -- Book {book_code}, Page {page}, Paragraph {paragraph}]\n{text}"
        )
    excerpts_text = "\n\n".join(blocks)

    return f"""Question: {question}

Here are the most relevant excerpts retrieved from the corpus:

{excerpts_text}

Using only the excerpts above, answer the question. Cite the source \
(book code and page) for each claim you make. If the excerpts only \
partially answer the question, answer what you can and note what's missing."""


def _call_anthropic(system_prompt: str, user_prompt: str, api_key: Optional[str], model: str = "claude-sonnet-4-6") -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai(system_prompt: str, user_prompt: str, api_key: Optional[str], model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def _call_huggingface_local(system_prompt: str, user_prompt: str, api_key: Optional[str], model: str = "meta-llama/Llama-3.1-8B-Instruct") -> str:
    raise NotImplementedError(
        "Wire this up with transformers.pipeline('text-generation', model=model) "
        "if you want to compare an open-source model."
    )


BACKENDS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "huggingface": _call_huggingface_local,
}


def query_llm_backend(system_prompt: str, user_prompt: str, backend: str = "anthropic",
                       model: Optional[str] = None, api_key: Optional[str] = None) -> str:
    fn = BACKENDS.get(backend)
    if fn is None:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {list(BACKENDS)}.")
    kwargs = {"model": model} if model else {}
    return fn(system_prompt, user_prompt, api_key, **kwargs)
