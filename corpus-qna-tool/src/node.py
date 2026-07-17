"""
node.py
-------
Python equivalent of Node.h. The C++ header declares:

    Node* get_top_k_para(string question, int k);

which returns the head of a singly linked list, where each Node carries
(book_code, page, paragraph). We mirror that exactly here (rather than just
returning a Python list) so the shape of the interface -- and the code you'd
write to walk it -- matches the original assignment.
"""

from __future__ import annotations

from typing import Optional


class Node:
    def __init__(self, book_code: Optional[int] = None, page: Optional[int] = None, paragraph: Optional[int] = None):
        self.book_code = book_code
        self.page = page
        self.paragraph = paragraph
        self.score: float = 0.0        # not part of the original Node.h contract,
                                        # but handy to carry the score along for
                                        # debugging / printing without a second lookup
        self.next: Optional["Node"] = None

    def __repr__(self) -> str:
        return f"Node(book_code={self.book_code}, page={self.page}, paragraph={self.paragraph}, score={self.score:.4f})"


def linked_list_to_list(head: Optional[Node]) -> list:
    """Utility: walk a Node linked list into a plain Python list (debugging/tests)."""
    out = []
    node = head
    while node is not None:
        out.append(node)
        node = node.next
    return out
