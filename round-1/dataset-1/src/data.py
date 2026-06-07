#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["loguru"]
# ///
"""
Load FOLIO, ProofWriter (RelNeg-OWA), and MALLS NL-FOL datasets from
temp/datasets/, standardize to exp_sel_data_out schema, and save to
full_data_out.json grouped by dataset source.
"""

import json
import random
import re
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WS = Path(__file__).parent
TEMP = WS / "temp" / "datasets"

QUANTIFIER_RE = re.compile(r"[∀∃]")
CONNECTIVE_RE = re.compile(r"[∧∨→⊕]")


def compositional_depth(fol: str) -> int:
    return len(QUANTIFIER_RE.findall(fol)) + len(CONNECTIVE_RE.findall(fol))


def load_folio() -> list[dict]:
    """Load FOLIO train+validation, split premises into individual NL-FOL pairs."""
    pairs: list[dict] = []
    mismatches = 0
    for split in ("train", "validation"):
        fname = TEMP / f"full_tasksource_folio_default_{split}.json"
        logger.info(f"Loading FOLIO {split} ({fname.name})")
        rows = json.loads(fname.read_text())
        logger.info(f"  {len(rows)} story rows")
        for row in rows:
            story_id = str(row["story_id"])
            label = str(row.get("label", ""))
            example_id = row.get("example_id", "")
            premises_nl = [p.strip() for p in row["premises"].split("\n") if p.strip()]
            premises_fol = [f.strip() for f in row["premises-FOL"].split("\n") if f.strip()]
            if len(premises_nl) != len(premises_fol):
                mismatches += 1
                continue
            for i, (nl, fol) in enumerate(zip(premises_nl, premises_fol)):
                pairs.append({
                    "id": f"folio_{split}_{example_id}_p{i}",
                    "nl_sentence": nl,
                    "gold_fol": fol,
                    "depth": compositional_depth(fol),
                    "story_id": story_id,
                    "label": label,
                    "split": split,
                })
            conc_nl = row.get("conclusion", "").strip()
            conc_fol = row.get("conclusion-FOL", "").strip()
            if conc_nl and conc_fol:
                pairs.append({
                    "id": f"folio_{split}_{example_id}_conc",
                    "nl_sentence": conc_nl,
                    "gold_fol": conc_fol,
                    "depth": compositional_depth(conc_fol),
                    "story_id": story_id,
                    "label": label,
                    "split": split,
                })
        logger.info(f"  {len(pairs)} pairs after {split}")
    logger.info(f"FOLIO total pairs: {len(pairs)} (mismatches skipped: {mismatches})")
    return pairs


def load_proofwriter(limit: int = 200) -> list[dict]:
    """Load ProofWriter RelNeg-OWA rows (no gold FOL — depth ablation only)."""
    fname = TEMP / "full_tasksource_proofwriter_default_test.json"
    logger.info(f"Loading ProofWriter ({fname.name})")
    all_rows = json.loads(fname.read_text())
    logger.info(f"  {len(all_rows)} total rows")
    relneg = [r for r in all_rows if "RelNeg" in r.get("id", "")]
    logger.info(f"  {len(relneg)} RelNeg rows, using {min(limit, len(relneg))}")
    sample = relneg[:limit]
    pairs: list[dict] = []
    for i, row in enumerate(sample):
        theory = row.get("theory", "")
        qdep = int(row.get("QDep", 0))
        # Theory is period-separated sentences
        raw_sents = re.split(r"\.\s+", theory)
        sents = [s.strip().rstrip(".") + "." for s in raw_sents if s.strip()]
        relational = [s for s in sents if re.search(r"\b(is|are)\b", s, re.I)]
        for j, sent in enumerate(relational):
            pairs.append({
                "id": f"proofwriter_test_{i}_s{j}",
                "nl_sentence": sent,
                "gold_fol": None,
                "depth": qdep,
                "story_id": None,
                "label": str(row.get("answer", "")),
            })
    logger.info(f"ProofWriter: {len(pairs)} relational sentences extracted")
    return pairs


def load_malls(limit: int = 1000) -> list[dict]:
    """Load MALLS NL-FOL pairs (GPT-4 generated, depth>=2 filter)."""
    fname = TEMP / "full_yuan-yang_MALLS-v0_default_train.json"
    logger.info(f"Loading MALLS ({fname.name})")
    rows = json.loads(fname.read_text())
    logger.info(f"  {len(rows)} total rows, using first {limit}")
    pairs: list[dict] = []
    for i, row in enumerate(rows[:limit]):
        nl = row.get("NL", "").strip()
        fol = row.get("FOL", "").strip()
        if not nl or not fol:
            continue
        depth = compositional_depth(fol)
        if depth < 2:
            continue
        pairs.append({
            "id": f"malls_train_{i}",
            "nl_sentence": nl,
            "gold_fol": fol,
            "depth": depth,
            "story_id": None,
            "label": None,
        })
    logger.info(f"MALLS: {len(pairs)} pairs (depth>=2)")
    return pairs


def assign_folds(pairs: list[dict], rng: random.Random) -> list[dict]:
    """Stratified 30-sentence alignment-validation split from FOLIO pairs."""
    buckets = {
        "low": [p for p in pairs if 2 <= p["depth"] <= 3],
        "mid": [p for p in pairs if 4 <= p["depth"] <= 5],
        "high": [p for p in pairs if p["depth"] >= 6],
    }
    val_ids: set[str] = set()
    for bucket, target in (("low", 12), ("mid", 12), ("high", 6)):
        chosen = rng.sample(buckets[bucket], min(target, len(buckets[bucket])))
        val_ids.update(p["id"] for p in chosen)
        logger.info(f"  align_val {bucket}: {len(chosen)}/{target}")
    for p in pairs:
        p["fold"] = "alignment_validation" if p["id"] in val_ids else "pilot"
    return pairs


def dedup(pairs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for p in pairs:
        key = p["nl_sentence"].strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def to_example(p: dict, fold: str) -> dict:
    ex: dict = {
        "input": p["nl_sentence"],
        "output": p["gold_fol"] if p["gold_fol"] is not None else "",
        "metadata_id": p["id"],
        "metadata_compositional_depth": p["depth"],
        "metadata_fold": fold,
        "metadata_story_id": p.get("story_id") or "",
        "metadata_label": p.get("label") or "",
    }
    return ex


@logger.catch(reraise=True)
def main() -> None:
    rng = random.Random(42)

    # --- FOLIO ---
    folio_raw = load_folio()
    folio_filtered = [p for p in folio_raw if p["depth"] >= 2]
    logger.info(f"FOLIO depth>=2: {len(folio_filtered)}")
    folio_deduped = dedup(folio_filtered)
    logger.info(f"FOLIO deduped: {len(folio_deduped)}")
    folio_with_folds = assign_folds(folio_deduped, rng)
    folio_examples = [to_example(p, p["fold"]) for p in folio_with_folds]

    # --- ProofWriter ---
    try:
        pw_raw = load_proofwriter(limit=200)
        pw_deduped = dedup(pw_raw)
        pw_examples = [to_example(p, "proofwriter_depth_ablation") for p in pw_deduped]
    except Exception:
        logger.warning("ProofWriter failed — skipping")
        pw_examples = []

    # --- MALLS ---
    try:
        malls_raw = load_malls(limit=1000)
        malls_deduped = dedup(malls_raw)
        malls_examples = [to_example(p, "malls_supplement") for p in malls_deduped]
    except Exception:
        logger.warning("MALLS failed — skipping")
        malls_examples = []

    # --- Assertions ---
    pilot_val = [e for e in folio_examples if e["metadata_fold"] in ("pilot", "alignment_validation")]
    assert all(e["output"] != "" for e in pilot_val), "Null gold_fol in pilot/alignment_validation rows"
    assert len(pilot_val) >= 150, f"Only {len(pilot_val)} pilot+val rows (need >= 150)"
    align_val = [e for e in folio_examples if e["metadata_fold"] == "alignment_validation"]
    assert len(align_val) == 30, f"alignment_validation has {len(align_val)} rows (expected 30)"

    # --- Build grouped output ---
    output = {
        "metadata": {
            "description": "FOLIO + ProofWriter + MALLS NL-FOL dataset for locality pilot",
            "sources": {
                "folio": "tasksource/folio (Yale-LILY/FOLIO, EMNLP 2024)",
                "proofwriter": "tasksource/proofwriter (Allen AI, RelNeg-OWA subset)",
                "malls": "yuan-yang/MALLS-v0 (LogicLLaMA paper, ICLR 2024)",
            },
            "depth_filter": ">=2 quantifiers+connectives (FOLIO, MALLS); QDep for ProofWriter",
            "alignment_validation_count": len(align_val),
            "pilot_count": len([e for e in folio_examples if e["metadata_fold"] == "pilot"]),
        },
        "datasets": [
            {"dataset": "folio", "examples": folio_examples},
            {"dataset": "proofwriter", "examples": pw_examples},
            {"dataset": "malls", "examples": malls_examples},
        ],
    }

    out_path = WS / "full_data_out.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    size_mb = out_path.stat().st_size / 1e6
    assert size_mb < 50, f"full_data_out.json is {size_mb:.1f}MB (>50MB)"
    logger.info(f"Saved to {out_path} ({size_mb:.2f}MB)")
    logger.info(f"  folio: {len(folio_examples)} | proofwriter: {len(pw_examples)} | malls: {len(malls_examples)}")


if __name__ == "__main__":
    main()
