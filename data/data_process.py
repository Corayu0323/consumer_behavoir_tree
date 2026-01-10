# -*- coding: utf-8 -*-
"""
Taobao Tianchi UserBehavior preprocessing (minimal, reproducible)

Input (default):
  cluster_data/UserBehavior.csv

Expected columns (either name set is OK):
  1) user_id, item_id, category_id, behavior_type, timestamp
  OR
  2) user_id, item_id, cat_id, type, ts

Outputs (default -> cluster_data/processed/):
  - events_sorted.parquet (or events_sorted.csv fallback)
  - user_sequences.jsonl  (one user per line)
  - stats.json            (summary stats for report)

Run:
  python cluster_data/data_process.py
  python cluster_data/data_process.py --max_users 20000 --min_len 2 --max_len 200
"""

import os
import json
import argparse
from typing import Dict, Tuple, List, Optional

import pandas as pd


# --------- column normalization ---------
CANON_MAP = {
    # canonical -> possible names
    "user_id": ["user_id", "uid", "user"],
    "item_id": ["item_id", "iid", "item"],
    "category_id": ["category_id", "cat_id", "category", "cid"],
    "behavior_type": ["behavior_type", "type", "behavior", "action"],
    "timestamp": ["timestamp", "ts", "time", "t"],
}

# behavior normalization + rank/code (tie-breaker if same timestamp)
# Tianchi commonly uses: pv, fav, cart, buy
BEHAVIOR_ALIAS = {
    "pv": "pv",
    "click": "pv",
    "fav": "fav",
    "collect": "fav",
    "cart": "cart",
    "add_to_cart": "cart",
    "buy": "buy",
    "order": "buy",
}

BEHAVIOR_CODE = {"pv": 1, "fav": 2, "cart": 3, "buy": 4}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols_lower)

    rename = {}
    existing = set(df.columns)
    for canon, candidates in CANON_MAP.items():
        for c in candidates:
            if c in existing:
                rename[c] = canon
                break

    df = df.rename(columns=rename)

    required = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Your columns are: {list(df.columns)[:20]} ..."
        )
    return df


def _normalize_behavior(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    s = s.map(lambda x: BEHAVIOR_ALIAS.get(x, x))
    return s


def _safe_to_parquet(df: pd.DataFrame, out_path: str) -> Tuple[str, bool]:
    try:
        df.to_parquet(out_path, index=False)
        return out_path, True
    except Exception:
        # fallback to csv
        csv_path = out_path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        return csv_path, False


def build_sequences(
    df: pd.DataFrame,
    min_len: int,
    max_len: int,
) -> pd.Series:
    """
    Returns: pandas Series indexed by user_id, values are list of (cat_id, type_code, ts)
    """
    # group by user_id -> list tuples
    seqs = (
        df.groupby("user_id", sort=False)
          .apply(lambda x: list(zip(x["category_id"], x["type_code"], x["timestamp"])))
    )

    # filter by length
    seqs = seqs[seqs.map(len) >= min_len]

    # truncate long sequences (keep most recent max_len)
    if max_len > 0:
        seqs = seqs.map(lambda lst: lst[-max_len:] if len(lst) > max_len else lst)

    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="cluster_data/UserBehavior.csv")
    parser.add_argument("--out_dir", default="cluster_data/processed")

    # basic cleaning
    parser.add_argument("--dedup", action="store_true", default=True)
    parser.add_argument("--no_dedup", action="store_true", default=False)

    # sequence constraints
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=200)

    # optional sampling (for fast dev)
    parser.add_argument("--max_users", type=int, default=0, help="0 means no limit")
    parser.add_argument("--sample_users_seed", type=int, default=42)

    args = parser.parse_args()
    if args.no_dedup:
        args.dedup = False

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- load ----------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # cluster_data/
    IN_CSV = os.path.join(BASE_DIR, "UserBehavior.csv")

    df = pd.read_csv(
        IN_CSV,
        nrows=100_000,
        header=None,
        names=["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
    )

    # ---------- normalize types ----------
    df["behavior_type"] = _normalize_behavior(df["behavior_type"])
    df = df[df["behavior_type"].isin(BEHAVIOR_CODE.keys())].copy()

    # timestamp + ids (keep as int where possible)
    for col in ["user_id", "item_id", "category_id", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["user_id", "category_id", "behavior_type", "timestamp"])
    df["user_id"] = df["user_id"].astype("int64")
    df["item_id"] = df["item_id"].astype("int64")
    df["category_id"] = df["category_id"].astype("int64")
    df["timestamp"] = df["timestamp"].astype("int64")

    # ---------- dedup ----------
    if args.dedup:
        df = df.drop_duplicates(subset=["user_id", "item_id", "category_id", "behavior_type", "timestamp"])

    # ---------- add behavior code (tie-breaker) ----------
    df["type_code"] = df["behavior_type"].map(BEHAVIOR_CODE).astype("int8")

    # ---------- sort within user by (timestamp, type_code) ----------
    df = df.sort_values(by=["user_id", "timestamp", "type_code"], ascending=[True, True, True])

    # ---------- optional user sampling ----------
    if args.max_users and args.max_users > 0:
        users = df["user_id"].drop_duplicates().sample(
            n=min(args.max_users, df["user_id"].nunique()),
            random_state=args.sample_users_seed
        )
        df = df[df["user_id"].isin(set(users))].copy()
        df = df.sort_values(by=["user_id", "timestamp", "type_code"], ascending=[True, True, True])

    # ---------- save sorted events ----------
    events_path, is_parquet = _safe_to_parquet(df, os.path.join(args.out_dir, "events_sorted.parquet"))

    # ---------- build sequences ----------
    seqs = build_sequences(df, min_len=args.min_len, max_len=args.max_len)
    import numpy as np
    from collections import defaultdict
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans

    def build_category_cooc(seqs, window=5, max_pairs=2_000_000):
        """
        seqs: iterable of list[(cat_id, type_code, ts)] or list[list[int]]
        return: (cat2idx, cooc_sparse_dict)
          cooc_sparse_dict[(i,j)] = count   (i<=j)
        """
        cat_set = set()
        for _, seq in seqs:
            for step in seq:
                cat_set.add(int(step[0]))
        cats = sorted(cat_set)
        cat2idx = {c: i for i, c in enumerate(cats)}

        cooc = defaultdict(int)
        pairs = 0
        for _, seq in seqs:
            cats_seq = [cat2idx[int(s[0])] for s in seq]
            L = len(cats_seq)
            for i in range(L):
                a = cats_seq[i]
                jmax = min(L, i + window + 1)
                for j in range(i + 1, jmax):
                    b = cats_seq[j]
                    x, y = (a, b) if a <= b else (b, a)
                    cooc[(x, y)] += 1
                    pairs += 1
                    if max_pairs and pairs >= max_pairs:
                        return cat2idx, cooc
        return cat2idx, cooc

    def cooc_to_embeddings(cat2idx, cooc, dim=32, min_count=3):
        """
        Build symmetric sparse matrix in (i,j)->count dict, then SVD to get embeddings.
        """
        n = len(cat2idx)
        # build CSR manually via lists (symmetric)
        rows, cols, vals = [], [], []
        for (i, j), c in cooc.items():
            if c < min_count:
                continue
            rows.append(i);
            cols.append(j);
            vals.append(c)
            if i != j:
                rows.append(j);
                cols.append(i);
                vals.append(c)

        # if too sparse or empty, fallback: one-hot-ish random init
        if len(vals) < 10:
            rng = np.random.default_rng(42)
            return rng.normal(size=(n, dim)).astype(np.float32)

        from scipy.sparse import csr_matrix
        X = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)

        svd = TruncatedSVD(n_components=dim, random_state=42)
        E = svd.fit_transform(X)  # (n, dim)
        return E.astype(np.float32)

    def cluster_categories(E, k=200):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(E)
        return labels.astype(int)

    # ---------- write jsonl ----------
    jsonl_path = os.path.join(args.out_dir, "user_sequences.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for uid, seq in seqs.items():
            obj = {
                "user_id": int(uid),
                # each step: [cat_id, type_code, ts]
                "seq": [[int(c), int(t), int(ts)] for (c, t, ts) in seq],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    # ---- category clustering (optional) ----
    DO_CLUSTER = True
    CLUSTER_K = 20  # 你可以先 50/100/200 试
    WINDOW = 5
    EMB_DIM = 32
    MIN_COOC = 3

    if DO_CLUSTER:
        # need (uid, seq) list
        seq_items = [(int(uid), [[int(c), int(t), int(ts)] for (c, t, ts) in seq]) for uid, seq in seqs.items()]

        cat2idx, cooc = build_category_cooc(seq_items, window=WINDOW)
        E = cooc_to_embeddings(cat2idx, cooc, dim=EMB_DIM, min_count=MIN_COOC)
        labels = cluster_categories(E, k=CLUSTER_K)

        # idx -> cat_id
        idx2cat = {i: c for c, i in cat2idx.items()}
        cat2cluster = {int(idx2cat[i]): int(labels[i]) for i in range(len(labels))}

        # save mapping
        map_path = os.path.join(args.out_dir, "category2cluster.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(cat2cluster, f, ensure_ascii=False)

        # write clustered sequences
        clustered_path = os.path.join(args.out_dir, "user_sequences_clustered.jsonl")
        with open(clustered_path, "w", encoding="utf-8") as f:
            for uid, seq in seq_items:
                new_seq = [[cat2cluster[s[0]], s[1], s[2]] for s in seq]
                f.write(json.dumps({"user_id": uid, "seq": new_seq}, ensure_ascii=False) + "\n")

        print(f"[OK] saved: {map_path}")
        print(f"[OK] saved: {clustered_path}")

    # ---------- stats for report ----------
    lengths = seqs.map(len)
    stats = {
        "input_csv": args.in_csv,
        "events_saved": events_path,
        "events_format": "parquet" if is_parquet else "csv",
        "n_events": int(len(df)),
        "n_users_total_in_events": int(df["user_id"].nunique()),
        "n_users_with_sequences": int(len(seqs)),
        "min_len": int(args.min_len),
        "max_len": int(args.max_len),
        "seq_len_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "seq_len_median": float(lengths.median()) if len(lengths) else 0.0,
        "seq_len_p95": float(lengths.quantile(0.95)) if len(lengths) else 0.0,
        "behavior_counts": df["behavior_type"].value_counts().to_dict(),
        "behavior_code_map": BEHAVIOR_CODE,
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[OK] Saved:")
    print(f"  - {events_path}")
    print(f"  - {jsonl_path}")
    print(f"  - {os.path.join(args.out_dir, 'stats.json')}")


if __name__ == "__main__":
    main()
