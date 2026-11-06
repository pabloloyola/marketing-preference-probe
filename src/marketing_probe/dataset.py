"""Load REAL Upworthy A/B comparisons from the exploratory packages CSV.

Do not make up labels. Use only pairs within a randomized test with the SAME image
(and same lede when present) so that headline text is the main varying feature.
"""
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def prepare(csv_path, output_path, n=100, seed=42, minimum_impressions=1000, allow_partial=False):
    by_test = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                impressions = int(float(row.get("impressions") or 0))
                clicks = int(float(row.get("clicks") or 0))
            except ValueError:
                continue
            headline = (row.get("headline") or "").strip()
            test_id = (row.get("clickability_test_id") or "").strip()
            if (not test_id or len(headline) < 12 or impressions < minimum_impressions
                    or clicks < 0 or clicks > impressions or str(row.get("problem", "0")).strip() in {"1", "1.0", "True"}):
                continue
            by_test[test_id].append({"headline": headline, "impressions": impressions,
                                    "clicks": clicks, "image": (row.get("eyecatcher_id") or "").strip(),
                                    "lede": (row.get("lede") or "").strip()})
    candidates = []
    for tid, arms in by_test.items():
        seen = set()
        for i, a in enumerate(arms):
            for b in arms[i+1:]:
                if not a["image"] or a["image"] != b["image"]:
                    continue
                if a["lede"] != b["lede"] or a["headline"] == b["headline"]:
                    continue
                pair_key = tuple(sorted([a["headline"], b["headline"]]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                candidates.append({"test_id": tid, "a": a, "b": b})
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen, used_tests = [], set()
    for item in candidates:
        if item["test_id"] in used_tests:
            continue
        used_tests.add(item["test_id"])
        a, b = item["a"], item["b"]
        if rng.random() < .5:
            a, b = b, a
        chosen.append({"pair_id": hashlib.sha256((item["test_id"]+a["headline"]+b["headline"]).encode()).hexdigest()[:16],
                       "test_id": item["test_id"], "headline_A": a["headline"], "headline_B": b["headline"],
                       "impressions_A": a["impressions"], "clicks_A": a["clicks"],
                       "impressions_B": b["impressions"], "clicks_B": b["clicks"],
                       "ctr_A": a["clicks"]/a["impressions"], "ctr_B": b["clicks"]/b["impressions"],
                       "winner": "A" if a["clicks"]/a["impressions"] > b["clicks"]/b["impressions"] else "B"})
        if len(chosen) >= n:
            break
    if len(chosen) < n and not allow_partial:
        raise ValueError(f"Only {len(chosen)} eligible distinct-test headline pairs found; requested {n}. Try a smaller n or lower min impressions.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for item in chosen:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(chosen), len(candidates)


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
