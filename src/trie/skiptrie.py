# -*- coding: utf-8 -*-
"""
Skip-Trie for BayesianTrie (space-for-time)

Core idea:
- After building the baseline Trie, add a single "skip pointer" at some nodes.
- A skip pointer represents a deterministic/high-confidence chain of edges:
      node --k1--> n1 --k2--> n2 ... --kL--> target
- When scoring, if upcoming keys match (k1..kL), we jump in one step:
      score += precomputed_nll_sum
      visited_nodes += 1   (count only the target visit for "skip")
      skip_hits += 1

This DOES NOT change:
- counts, probabilities, NLL definition
- Hard OOD semantics (missing edge)
It only changes the traversal to reduce repeated dict lookups and node visits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

from .trie import BayesianTrie
from .node import TrieNode, StepKey, step_to_key


@dataclass
class SkipPointer:
    keys: Tuple[StepKey, ...]   # the sequence of keys to match
    target: TrieNode            # node after following all keys
    nll_sum: float              # sum_{j=1..L} -log(child.count / parent.count)
    steps: int                  # L


class SkipBayesianTrie(BayesianTrie):
    def __init__(self) -> None:
        super().__init__()
        self._skip_pointer_count = 0

    @property
    def skip_pointer_count(self) -> int:
        return int(self._skip_pointer_count)

    def count_skip_pointers(self) -> int:
        return int(self._skip_pointer_count)

    def build_skip_pointers(
        self,
        min_count: int = 20,
        min_prob: float = 0.95,
        max_jump: int = 4,
    ) -> None:
        """
        Build ONE best skip pointer for each node when possible.

        Rules (minimal & safe):
        - Only consider nodes with count >= min_count.
        - Follow the most-likely child repeatedly while:
            P(child|node) >= min_prob
          up to max_jump steps.
        - This gives a deterministic-ish high-frequency prefix chain.

        Notes:
        - You can tune min_count/min_prob/max_jump in experiments.
        - If a node cannot form a valid chain (length < 2), no skip pointer is added.
        """
        self._skip_pointer_count = 0

        def best_child(n: TrieNode) -> Optional[Tuple[StepKey, TrieNode, float]]:
            if not getattr(n, "children", None) or n.count <= 0:
                return None
            # find max prob child
            best_k, best_c, best_p = None, None, 0.0
            for k, c in n.children.items():
                if c is None:
                    continue
                p = (c.count / n.count) if n.count > 0 else 0.0
                if p > best_p:
                    best_k, best_c, best_p = k, c, p
            if best_k is None:
                return None
            return best_k, best_c, float(best_p)

        # DFS stack
        stack = [self.root]
        while stack:
            node = stack.pop()

            # push children for traversal
            if getattr(node, "children", None):
                for ch in node.children.values():
                    stack.append(ch)

            # attempt to create skip pointer at this node
            if node.count < min_count:
                setattr(node, "skip", None)
                continue

            keys: List[StepKey] = []
            nll_sum = 0.0
            cur = node
            steps = 0

            while steps < max_jump:
                bc = best_child(cur)
                if bc is None:
                    break
                k, child, p = bc
                if p < min_prob:
                    break
                # accumulate exact nll for this transition
                nll_sum += -math.log(max(p, 1e-12))
                keys.append(k)
                cur = child
                steps += 1

            # Require at least 2-step jump to be meaningful
            if steps >= 2:
                setattr(node, "skip", SkipPointer(keys=tuple(keys), target=cur, nll_sum=float(nll_sum), steps=steps))
                self._skip_pointer_count += 1
            else:
                setattr(node, "skip", None)

    def neg_loglik(self, seq: Iterable[Any], clamp_eps: float = 1e-12) -> Tuple[bool, int, float, int, int]:
        """
        Skip-accelerated NLL.
        Returns:
          (ok, first_missing_step, score, visited_nodes, skip_hits)

        visited_nodes semantics:
        - root counts as 1
        - normal step: +1 per visited node
        - skip jump: +1 for the target node (intermediate nodes not counted)
        """
        steps = list(seq)
        keys = [step_to_key(s) for s in steps]

        node = self.root
        visited = 1
        score = 0.0
        i = 0
        skip_hits = 0

        while i < len(keys):
            sp: Optional[SkipPointer] = getattr(node, "skip", None)

            if sp is not None:
                L = sp.steps
                if i + L <= len(keys) and tuple(keys[i:i+L]) == sp.keys:
                    # take the jump
                    score += sp.nll_sum
                    node = sp.target
                    visited += 1
                    skip_hits += 1
                    i += L
                    continue

            # fallback to baseline single-step
            k = keys[i]
            child = node.get_child(k)
            if child is None:
                return False, i, float(score), int(visited), int(skip_hits)

            p = (child.count / node.count) if node.count > 0 else 0.0
            p = max(p, clamp_eps)
            score += -math.log(p)

            node = child
            visited += 1
            i += 1

        return True, -1, float(score), int(visited), int(skip_hits)
