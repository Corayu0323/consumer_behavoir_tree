# -*- coding: utf-8 -*-
"""
OOD Detector for Bayesian Trie (Baseline)

Definitions:
- Hard OOD: a step transition is missing in Trie (path breaks)
- Causal / Logic OOD: path exists but violates stage logic rules
- Soft OOD: path exists, but anomaly score is high
    score S = sum_t -log P(x_t | prefix)

Outputs:
- Per-sequence results: result_type, first_bad_step, score, time_ms, visited_nodes
"""

from __future__ import annotations

import os
import csv
import json
import argparse
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Any

from src.trie.trie import BayesianTrie
from src.ood.timing import perf_counter, elapsed_ms, summarize


@dataclass
class OODResult:
    seq_id: int
    user_id: int
    length: int
    result_type: str            # "ID" | "Hard_OOD" | "Soft_OOD" | "Causal_OOD"
    first_bad_step: int         # -1 if none
    score: float                # anomaly score (NLL)
    visited_nodes: int
    time_ms: float


class TrieOODDetector:
    """Uses a trained BayesianTrie to score a sequence."""
    def __init__(self, trie: BayesianTrie, soft_threshold: float = 10.0, clamp_eps: float = 1e-12):
        self.trie = trie
        self.soft_threshold = float(soft_threshold)
        self.clamp_eps = float(clamp_eps)

    def detect_causal_ood(self, seq: List[Any]) -> bool:
        """
        Rule-based Causal / Logic OOD detection.
        seq: List[(cluster_id, type_code, ...)]
        """
        if not seq:
            return False

        type_codes = [x[1] for x in seq]  # second field is type_code

        # Rule 1: strong stage reversal (e.g., 4 -> 1, 3 -> 1)
        for i in range(1, len(type_codes)):
            if type_codes[i - 1] - type_codes[i] >= 2:
                return True

        # Rule 2: too-early purchase (buy happens in first 2 steps)
        if 4 in type_codes:
            first_buy_pos = type_codes.index(4)
            if first_buy_pos <= 1:
                return True

        return False

    def predict(self, seq: Iterable[Any]) -> Tuple[str, int, float, int]:
        """
        Returns: (result_type, first_bad_step, score, visited_nodes)

        Priority:
          Hard_OOD -> Causal_OOD -> Soft_OOD -> ID
        """
        ok, miss_i, score, visited = self.trie.neg_loglik(seq, clamp_eps=self.clamp_eps)

        if not ok:
            return "Hard_OOD", int(miss_i), float(score), int(visited)

        if self.detect_causal_ood(list(seq) if not isinstance(seq, list) else seq):
            return "Causal_OOD", -1, float(score), int(visited)

        if score > self.soft_threshold:
            return "Soft_OOD", -1, float(score), int(visited)

        return "ID", -1, float(score), int(visited)


def iter_user_sequences_with_uid(path: str):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_path", default="data/cluster_data/processed/user_sequences_clustered.jsonl")
    parser.add_argument("--out_csv", default="results/baseline/baseline_ood.csv")
    parser.add_argument("--train_ratio", type=float, default=0.8)

    # build trie
    parser.add_argument("--build_limit", type=int, default=0, help="0 = use all sequences to build trie")
    # evaluation
    parser.add_argument("--eval_limit", type=int, default=0, help="0 = score all sequences")

    # soft ood threshold
    parser.add_argument("--soft_th", type=float, default=10.0)

    args = parser.parse_args()
    ensure_dir(os.path.dirname(args.out_csv) or ".")

    # 1) load all data once
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

    print(f"[OK] built trie with TRAIN users: {len(train_users)}  train_sequences: {len(train_data)} (root.count={trie.root.count})")
    print(f"[OK] TEST users: {len(test_users)}  test_sequences: {len(test_data)}")

    # 4) score TEST only
    detector = TrieOODDetector(trie, soft_threshold=args.soft_th)

    rows: List[OODResult] = []
    scores: List[float] = []
    time_ms_list: List[float] = []
    type_counts = {"ID": 0, "Hard_OOD": 0, "Soft_OOD": 0, "Causal_OOD": 0}

    t0 = perf_counter()
    for seq_id, (uid, seq) in enumerate(test_data):
        start = perf_counter()
        result_type, first_bad, score, visited = detector.predict(seq)
        time_ms = elapsed_ms(start)

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

        type_counts[result_type] = type_counts.get(result_type, 0) + 1
        scores.append(score)
        time_ms_list.append(time_ms)

    total_ms = elapsed_ms(t0)

    # 5) write csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "user_id", "length", "result_type", "first_bad_step", "score", "visited_nodes", "time_ms"])
        for r in rows:
            w.writerow([r.seq_id, r.user_id, r.length, r.result_type, r.first_bad_step,
                        f"{r.score:.6f}", r.visited_nodes, f"{r.time_ms:.3f}"])

    # 6) print summary
    score_stat = summarize(scores)
    time_stat = summarize(time_ms_list)

    print(f"[OK] scored sequences: {len(rows)}  total_time_ms={total_ms:.1f}")
    print("counts:", type_counts)
    print(f"score mean={score_stat['mean']:.4f}  p50={score_stat['p50']:.4f}  p95={score_stat['p95']:.4f}  max={score_stat['max']:.4f}")
    print(f"time  mean={time_stat['mean']:.3f}ms  p50={time_stat['p50']:.3f}ms  p95={time_stat['p95']:.3f}ms  max={time_stat['max']:.3f}ms")
    print(f"[OK] saved: {args.out_csv}")


if __name__ == "__main__":
    main()
