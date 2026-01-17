# -*- coding: utf-8 -*-
"""
OOD Detector for Skip-Trie (B)

Goal:
- Same判定逻辑、同一CSV字段（便于A/B对照）
- 仅在“查询/访问方式”上使用 Skip Pointer 加速
- 额外输出（可选）：skip_hits、pointer_count（用于7.2表格）

Assumptions (尽量做成兼容式)：
- SkipTrie 类提供 insert(seq)
- neg_loglik(seq, clamp_eps=...) 返回：
    (ok, miss_i, score, visited_nodes)  或
    (ok, miss_i, score, visited_nodes, skip_hits)
- 若你的 SkipTrie API 不同，只需改动 build 与 neg_loglik 调用的几行
"""

from __future__ import annotations

import os
import csv
import json
import argparse
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Any, Optional

from src.ood.timing import perf_counter, elapsed_ms, summarize

# 你需要在 src/trie/ 里提供一个 skiptrie.py（或改成你实际文件名）
# 其中包含 SkipBayesianTrie（或改成你实际类名）
try:
    from src.trie.skiptrie import SkipBayesianTrie  # <-- 如类名不同就改这里
except Exception as e:
    raise ImportError(
        "Cannot import SkipBayesianTrie. Please create src/trie/skiptrie.py "
        "and define class SkipBayesianTrie (or modify detector_skip.py import)."
    ) from e


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
    skip_hits: int              # skip pointer hit count (0 for baseline / unsupported)


class SkipTrieOODDetector:
    """Uses a trained SkipBayesianTrie to score a sequence."""
    def __init__(self, trie: SkipBayesianTrie, soft_threshold: float = 10.0, clamp_eps: float = 1e-12):
        self.trie = trie
        self.soft_threshold = float(soft_threshold)
        self.clamp_eps = float(clamp_eps)

    def detect_causal_ood(self, seq: List[Any]) -> bool:
        """Same rule-based Causal / Logic OOD as baseline."""
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

    def _neg_loglik_with_optional_skip(self, seq: Iterable[Any]) -> Tuple[bool, int, float, int, int]:
        """
        Try to read skip_hits if SkipTrie.neg_loglik returns it.
        Compatible with:
          - (ok, miss_i, score, visited)
          - (ok, miss_i, score, visited, skip_hits)
        """
        res = self.trie.neg_loglik(seq, clamp_eps=self.clamp_eps)

        if not isinstance(res, tuple):
            raise TypeError("trie.neg_loglik must return a tuple.")

        if len(res) == 4:
            ok, miss_i, score, visited = res
            return bool(ok), int(miss_i), float(score), int(visited), 0
        if len(res) == 5:
            ok, miss_i, score, visited, skip_hits = res
            return bool(ok), int(miss_i), float(score), int(visited), int(skip_hits)

        raise ValueError("Unsupported neg_loglik return format. Expect 4-tuple or 5-tuple.")

    def predict(self, seq: Iterable[Any]) -> Tuple[str, int, float, int, int]:
        """
        Returns: (result_type, first_bad_step, score, visited_nodes, skip_hits)

        Priority:
          Hard_OOD -> Causal_OOD -> Soft_OOD -> ID
        """
        # Ensure we can run causal rules without consuming generators
        seq_list = seq if isinstance(seq, list) else list(seq)

        ok, miss_i, score, visited, skip_hits = self._neg_loglik_with_optional_skip(seq_list)

        if not ok:
            return "Hard_OOD", int(miss_i), float(score), int(visited), int(skip_hits)

        if self.detect_causal_ood(seq_list):
            return "Causal_OOD", -1, float(score), int(visited), int(skip_hits)

        if score > self.soft_threshold:
            return "Soft_OOD", -1, float(score), int(visited), int(skip_hits)

        return "ID", -1, float(score), int(visited), int(skip_hits)


def iter_user_sequences_with_uid(path: str):
    """Yields (user_id, seq) per line: {"user_id": ..., "seq": [[cat, type_code, ts], ...]}"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield int(obj["user_id"]), obj["seq"]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def try_get_skip_pointer_count(trie: Any) -> Optional[int]:
    """
    Best-effort pointer count for 7.2 'space overhead':
    - supports trie.skip_pointer_count attr
    - or trie.count_skip_pointers() method
    """
    if hasattr(trie, "skip_pointer_count"):
        try:
            return int(getattr(trie, "skip_pointer_count"))
        except Exception:
            return None
    if hasattr(trie, "count_skip_pointers") and callable(getattr(trie, "count_skip_pointers")):
        try:
            return int(trie.count_skip_pointers())
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_path", default="data/cluster_data/processed/user_sequences_clustered.jsonl")
    parser.add_argument("--out_csv", default="results/skiptrie/skip_ood.csv")
    parser.add_argument("--train_ratio", type=float, default=0.8)

    # build trie
    parser.add_argument("--build_limit", type=int, default=0, help="0 = use all sequences to build trie")
    # evaluation
    parser.add_argument("--eval_limit", type=int, default=0, help="0 = score all sequences")

    # soft ood threshold
    parser.add_argument("--soft_th", type=float, default=10.0)

    # (可选) Skip 指针构建参数：只要你的 SkipTrie 支持，就会被调用；不支持则忽略
    parser.add_argument("--skip_min_count", type=int, default=0,
                        help="min prefix count to create a skip pointer (0 = use SkipTrie default / disable)")
    parser.add_argument("--skip_max_jump", type=int, default=0,
                        help="max jump length for skip pointer (0 = use SkipTrie default)")

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

    if args.build_limit and args.build_limit > 0:
        train_data = train_data[:args.build_limit]
    if args.eval_limit and args.eval_limit > 0:
        test_data = test_data[:args.eval_limit]

    # 3) build Skip-Trie on TRAIN only
    trie = SkipBayesianTrie()
    for _, seq in train_data:
        trie.insert(seq)

    # (可选) build skip pointers if your SkipTrie supports it
    # e.g., trie.build_skip_pointers(min_count=..., max_jump=...)
    if hasattr(trie, "build_skip_pointers") and callable(getattr(trie, "build_skip_pointers")):
        kwargs = {}
        if args.skip_min_count and args.skip_min_count > 0:
            kwargs["min_count"] = args.skip_min_count
        if args.skip_max_jump and args.skip_max_jump > 0:
            kwargs["max_jump"] = args.skip_max_jump
        try:
            trie.build_skip_pointers(**kwargs)
        except TypeError:
            # signature mismatch; ignore to keep "minimal change"
            trie.build_skip_pointers()

    print(f"[OK] built skip-trie with TRAIN users: {len(train_users)}  train_sequences: {len(train_data)}")
    print(f"[OK] TEST users: {len(test_users)}  test_sequences: {len(test_data)}")

    pointer_count = try_get_skip_pointer_count(trie)
    if pointer_count is not None:
        print(f"[OK] skip pointer count: {pointer_count}")

    # 4) score TEST only
    detector = SkipTrieOODDetector(trie, soft_threshold=args.soft_th)

    rows: List[OODResult] = []
    scores: List[float] = []
    time_ms_list: List[float] = []
    skip_hits_list: List[int] = []
    type_counts = {"ID": 0, "Hard_OOD": 0, "Soft_OOD": 0, "Causal_OOD": 0}

    t0 = perf_counter()
    for seq_id, (uid, seq) in enumerate(test_data):
        start = perf_counter()
        result_type, first_bad, score, visited, skip_hits = detector.predict(seq)
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
                skip_hits=skip_hits,
            )
        )
        type_counts[result_type] = type_counts.get(result_type, 0) + 1
        scores.append(score)
        time_ms_list.append(time_ms)
        skip_hits_list.append(skip_hits)

    total_ms = elapsed_ms(t0)

    # 5) write csv (same baseline columns + one extra column: skip_hits)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "seq_id", "user_id", "length", "result_type", "first_bad_step",
            "score", "visited_nodes", "time_ms", "skip_hits"
        ])
        for r in rows:
            w.writerow([
                r.seq_id, r.user_id, r.length, r.result_type, r.first_bad_step,
                f"{r.score:.6f}", r.visited_nodes, f"{r.time_ms:.3f}", r.skip_hits
            ])

    # 6) print summary
    score_stat = summarize(scores)
    time_stat = summarize(time_ms_list)

    # Skip hit rate: fraction of sequences with >=1 skip hit (simple, stable definition)
    hit_rate = 0.0
    if skip_hits_list:
        hit_rate = sum(1 for x in skip_hits_list if x > 0) / len(skip_hits_list)

    print(f"[OK] scored sequences: {len(rows)}  total_time_ms={total_ms:.1f}")
    print("counts:", type_counts)
    print(f"score mean={score_stat['mean']:.4f}  p50={score_stat['p50']:.4f}  p95={score_stat['p95']:.4f}  max={score_stat['max']:.4f}")
    print(f"time  mean={time_stat['mean']:.3f}ms  p50={time_stat['p50']:.3f}ms  p95={time_stat['p95']:.3f}ms  max={time_stat['max']:.3f}ms")
    print(f"skip hit rate={hit_rate * 100:.2f}%  (seq-level, skip_hits>0)")
    if pointer_count is not None:
        print(f"space overhead: skip_pointers={pointer_count}")
    print(f"[OK] saved: {args.out_csv}")


if __name__ == "__main__":
    main()
