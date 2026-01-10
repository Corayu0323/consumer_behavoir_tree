# -*- coding: utf-8 -*-
"""
Baseline Bayesian Trie

- Insert user sequences into Trie
- Provide conditional probability:
      P(next | prefix) = count(prefix->next) / count(prefix)
- Provide walk / score helpers for OOD detector later

Input sequence step format (from your jsonl):
  [cat_id, type_code, ts]
Trie edge key uses (cat_id, type_code) only (timestamp excluded).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any

from .node import TrieNode, StepKey, step_to_key


@dataclass
class WalkResult:
    ok: bool
    first_missing_step: int  # -1 if ok
    final_node: TrieNode
    visited_nodes: int


class BayesianTrie:
    def __init__(self) -> None:
        self.root = TrieNode()

    # ---------- build ----------
    def insert(self, seq: Iterable[Any]) -> None:
        """
        Insert one sequence.
        Count semantics:
          - root.count increments once per sequence
          - each visited node.count increments once per sequence passing through it
        """
        node = self.root
        node.add_count(1)

        for step in seq:
            key = step_to_key(step)  # (cat_id, type_code)
            node = node.ensure_child(key)
            node.add_count(1)

        node.is_end = True

    def bulk_insert(self, sequences: Iterable[Iterable[Any]], limit: int = 0) -> int:
        """
        Insert multiple sequences.
        Returns number of sequences inserted.
        """
        n = 0
        for seq in sequences:
            self.insert(seq)
            n += 1
            if limit and n >= limit:
                break
        return n

    # ---------- query ----------
    def walk(self, seq: Iterable[Any]) -> WalkResult:
        """
        Walk along sequence keys without creating nodes.
        If missing, returns first_missing_step index (0-based).
        """
        node = self.root
        visited = 1
        for i, step in enumerate(seq):
            key = step_to_key(step)
            nxt = node.get_child(key)
            if nxt is None:
                return WalkResult(ok=False, first_missing_step=i, final_node=node, visited_nodes=visited)
            node = nxt
            visited += 1
        return WalkResult(ok=True, first_missing_step=-1, final_node=node, visited_nodes=visited)

    def transition_prob(self, node: TrieNode, next_key: StepKey, eps: float = 1e-12) -> float:
        """
        P(next_key | node_prefix) = child.count / node.count
        - If transition missing, returns 0.0
        - eps is only used to avoid log(0) in scoring helpers (if you choose to clamp)
        """
        child = node.get_child(next_key)
        if child is None or node.count <= 0:
            return 0.0
        p = child.count / node.count
        # do not force eps here; leave exact 0 for hard OOD logic
        return float(p)

    def neg_loglik(self, seq: Iterable[Any], clamp_eps: float = 1e-12) -> Tuple[bool, int, float, int]:
        """
        Compute negative log-likelihood score:
            S = sum_{t} -log P(x_t | prefix)
        Returns:
            (ok, first_missing_step, score, visited_nodes)
        Note:
          - If a step is missing -> ok=False (Hard OOD) and stop at first missing.
          - If present but p extremely small, score will be large.
        """
        node = self.root
        visited = 1
        score = 0.0

        for i, step in enumerate(seq):
            key = step_to_key(step)
            child = node.get_child(key)
            if child is None:
                return False, i, score, visited

            p = child.count / node.count if node.count > 0 else 0.0
            # clamp for numeric stability
            p = max(p, clamp_eps)
            score += -math.log(p)

            node = child
            visited += 1

        return True, -1, score, visited


# ---------- utilities: load sequences ----------
def iter_user_sequences_jsonl(path: str) -> Iterable[List[List[int]]]:
    """
    Yields seq for each line:
      {"user_id": ..., "seq": [[cat, type_code, ts], ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["seq"]


# ---------- minimal runnable test ----------
if __name__ == "__main__":
    # Adjust this path if you run from a different working directory
    SEQ_PATH = "data/cluster_data/processed/user_sequences_clustered.jsonl"

    trie = BayesianTrie()
    n = trie.bulk_insert(iter_user_sequences_jsonl(SEQ_PATH), limit=2000)

    print(f"[OK] inserted sequences: {n}")
    print(f"root.count = {trie.root.count}")
    print(f"root children = {len(trie.root.children)}")

    # take the first sequence to test walk + score
    first_seq = next(iter_user_sequences_jsonl(SEQ_PATH))
    wr = trie.walk(first_seq)
    ok, miss_i, score, visited = trie.neg_loglik(first_seq)

    print(f"walk ok={wr.ok}, visited_nodes={wr.visited_nodes}")
    print(f"nll ok={ok}, miss_i={miss_i}, score={score:.4f}, visited_nodes={visited}")
