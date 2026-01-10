# -*- coding: utf-8 -*-
"""
OOD Detector for Bayesian Trie (Baseline)

Definitions:
- Hard OOD: a step transition is missing in Trie (path breaks)
- Soft OOD: path exists, but anomaly score is high
    score S = sum_t -log P(x_t | prefix)

Outputs:
- Per-sequence results: result_type, first_bad_step, score, time_ms, visited_nodes
"""

from __future__ import annotations

import os
import csv
import time
import math
import json
import argparse
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Any, Dict, Optional

from src.trie.trie import BayesianTrie, iter_user_sequences_jsonl
from src.trie.node import step_to_key


@dataclass
class OODResult:
    seq_id: int
    user_id: int
    length: int
    result_type: str            # "ID" | "Hard_OOD" | "Soft_OOD"
    first_bad_step: int         # -1 if none
    score: float                # anomaly score (NLL)
    visited_nodes: int
    time_ms: float


class TrieOODDetector:
    """
    Uses a trained BayesianTrie to score a sequence.
    """
    def __init__(self, trie: BayesianTrie, soft_threshold: float = 10.0, clamp_eps: float = 1e-12):
        self.trie = trie
        self.soft_threshold = float(soft_threshold)
        self.clamp_eps = float(clamp_eps)

    def predict(self, seq: Iterable[Any]) -> Tuple[str, int, float, int]:
        """
        Returns: (result_type, first_bad_step, score, visited_nodes)

        - If a step is missing -> Hard_OOD, score is partial NLL up to before missing
        - Else compute full score:
            score > soft_threshold -> Soft_OOD
            else -> ID
        """
        ok, miss_i, score, visited = self.trie.neg_loglik(seq, clamp_eps=self.clamp_eps)

        if not ok:
            return "Hard_OOD", int(miss_i), float(score), int(visited)

        if score > self.soft_threshold:
            return "Soft_OOD", -1, float(score), int(visited)

        return "ID", -1, float(score), int(visited)


def iter_user_sequences_with_uid(path: str) -> Iterable[Tuple[int, List[List[int]]]]:
    """
    Yields (user_id, seq) for each line:
      {"user_id": ..., "seq": [[cat, type_code, ts], ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield int(obj["user_id"]), obj["seq"]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def percentile(sorted_vals: List[float], q: float) -> float:
    """
    q in [0,1], expects sorted list.
    """
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    idx = int(math.ceil(q * len(sorted_vals))) - 1
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq_path",
        default="data/cluster_data/processed/user_sequences_clustered.jsonl")
    parser.add_argument("--out_csv", default="results/baseline_ood.csv")
    parser.add_argument("--train_ratio", type=float, default=0.8)

    # build trie
    parser.add_argument("--build_limit", type=int, default=0, help="0 = use all sequences to build trie")
    # evaluation
    parser.add_argument("--eval_limit", type=int, default=0, help="0 = score all sequences")

    # soft ood threshold
    parser.add_argument("--soft_th", type=float, default=10.0)

    args = parser.parse_args()

    ensure_dir(os.path.dirname(args.out_csv) or ".")

    # 1) load all (uid, seq) once
    all_data = list(iter_user_sequences_with_uid(args.seq_path))
    all_users = sorted({uid for uid, _ in all_data})

    # 2) split users (deterministic)
    n_train_users = int(len(all_users) * args.train_ratio)
    train_users = set(all_users[:n_train_users])
    test_users = set(all_users[n_train_users:])

    train_data = [(uid, seq) for uid, seq in all_data if uid in train_users]
    test_data = [(uid, seq) for uid, seq in all_data if uid in test_users]

    # optional limits
    if args.build_limit and args.build_limit > 0:
        train_data = train_data[:args.build_limit]
    if args.eval_limit and args.eval_limit > 0:
        test_data = test_data[:args.eval_limit]

    # 3) build trie on TRAIN only
    trie = BayesianTrie()
    for _, seq in train_data:
        trie.insert(seq)

    print(
        f"[OK] built trie with TRAIN users: {len(train_users)}  train_sequences: {len(train_data)} (root.count={trie.root.count})")
    print(f"[OK] TEST users: {len(test_users)}  test_sequences: {len(test_data)}")

    # 4) score TEST only
    detector = TrieOODDetector(trie, soft_threshold=args.soft_th)

    rows = []
    scores = []
    type_counts = {"ID": 0, "Hard_OOD": 0, "Soft_OOD": 0}

    t0 = time.perf_counter()
    for seq_id, (uid, seq) in enumerate(test_data):
        start = time.perf_counter()
        result_type, first_bad, score, visited = detector.predict(seq)
        time_ms = (time.perf_counter() - start) * 1000.0

        rows.append(
            OODResult(
                seq_id=seq_id,
                user_id=uid,
                length=len(seq),
                result_type=result_type,
                first_bad_step=first_bad,
                score=score,
                visited_nodes=visited,
                time_ms=time_ms,
            )
        )
        type_counts[result_type] += 1
        scores.append(score)

    total_ms = (time.perf_counter() - t0) * 1000.0

    # 3) write csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "user_id", "length", "result_type", "first_bad_step", "score", "visited_nodes", "time_ms"])
        for r in rows:
            w.writerow([r.seq_id, r.user_id, r.length, r.result_type, r.first_bad_step,
                        f"{r.score:.6f}", r.visited_nodes, f"{r.time_ms:.3f}"])

    # 4) print summary
    scores_sorted = sorted(scores)
    avg_score = sum(scores_sorted) / len(scores_sorted) if scores_sorted else 0.0

    print(f"[OK] scored sequences: {len(rows)}  total_time_ms={total_ms:.1f}")
    print("counts:", type_counts)
    print(f"score mean={avg_score:.4f}  p50={percentile(scores_sorted, 0.50):.4f}  "
          f"p95={percentile(scores_sorted, 0.95):.4f}  max={scores_sorted[-1]:.4f}" if scores_sorted else "score: empty")
    print(f"[OK] saved: {args.out_csv}")


if __name__ == "__main__":
    main()
