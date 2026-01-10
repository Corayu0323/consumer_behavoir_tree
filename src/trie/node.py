# -*- coding: utf-8 -*-
"""
Trie / Skip-Trie Node definitions (HANDWRITTEN)

Design goals (for report):
1) children uses hash table (Python dict) because category_id space is sparse.
2) count stores "number of sequences passing through this node" (prefix frequency),
   enabling conditional probability:
       P(next | prefix) = next_node.count / current_node.count
3) key is (category_id, type_code) rather than timestamp to avoid exploding branches.
4) SkipTrieNode adds skip_pointers for "space-for-time" optimization (Skip List idea).

Key conventions:
- Each step token in a sequence: (cat_id, type_code, ts)
- Node transition key: (cat_id, type_code)  <-- timestamp excluded by default
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

StepKey = Tuple[int, int]  # (category_id, type_code)


@dataclass
class TrieNode:
    """
    Basic Bayesian Trie node.
    """
    # count: how many sequences pass through this node (prefix frequency)
    count: int = 0

    # children: hash index for next step
    # Key: (category_id, type_code) -> Value: TrieNode
    children: Dict[StepKey, "TrieNode"] = None

    # is_end: whether any sequence ends at this node (optional)
    is_end: bool = False

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = {}

    def get_child(self, key: StepKey) -> Optional["TrieNode"]:
        return self.children.get(key)

    def ensure_child(self, key: StepKey) -> "TrieNode":
        """
        Get existing child if present; otherwise create it.
        """
        child = self.children.get(key)
        if child is None:
            child = TrieNode()
            self.children[key] = child
        return child

    def add_count(self, delta: int = 1) -> None:
        self.count += delta


@dataclass
class SkipTrieNode(TrieNode):
    """
    Skip-Trie node:
    - Inherits basic TrieNode fields
    - Adds skip_pointers: fast-forward pointers for frequent fixed-length patterns

    skip_pointers:
      Key: pattern tuple of StepKey(s), e.g. ((catA,tA),(catB,tB),(catC,tC))
      Value: reference to the end node after jumping (direct object pointer)
    """
    skip_pointers: Dict[Tuple[StepKey, ...], "SkipTrieNode"] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.skip_pointers is None:
            self.skip_pointers = {}

    def get_skip(self, pattern: Tuple[StepKey, ...]) -> Optional["SkipTrieNode"]:
        return self.skip_pointers.get(pattern)

    def set_skip(self, pattern: Tuple[StepKey, ...], end_node: "SkipTrieNode") -> None:
        self.skip_pointers[pattern] = end_node


def step_to_key(step: Any) -> StepKey:
    """
    Convert a raw step into a key used in Trie edges.

    Supported step formats:
    - (cat_id, type_code)          -> use directly
    - (cat_id, type_code, ts)      -> ignore ts
    - [cat_id, type_code, ts]      -> ignore ts
    """
    if isinstance(step, (tuple, list)):
        if len(step) < 2:
            raise ValueError(f"Invalid step (len<2): {step}")
        cat_id = int(step[0])
        type_code = int(step[1])
        return (cat_id, type_code)
    raise TypeError(f"Unsupported step type: {type(step)} -> {step}")


if __name__ == "__main__":
    # ===== minimal sanity check =====
    print("Running TrieNode sanity check...")

    root = TrieNode()

    # 模拟一个行为 step: (category_id, type_code)
    step1 = (1001, 1)   # pv
    step2 = (1001, 3)   # cart

    # 插入两步路径
    root.add_count()
    node1 = root.ensure_child(step1)
    node1.add_count()
    node2 = node1.ensure_child(step2)
    node2.add_count()

    # 输出结构信息
    print("Root count:", root.count)
    print("Children of root:", root.children.keys())
    print("Children of node1:", node1.children.keys())
    print("Node2 count:", node2.count)

    print("Sanity check passed.")
