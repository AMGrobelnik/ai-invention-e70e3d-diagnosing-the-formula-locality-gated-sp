#!/usr/bin/env python3
"""Build unified NL-FOL dataset from FOLIO, ProofWriter, and MALLS."""

import json
import random
import re
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/build.log", rotation="30 MB", level="DEBUG")

WS = Path("/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/run_bp-Wzi9abxjN/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")
TEMP = WS / "temp" / "datasets"

QUANTIFIER_RE = re.compile(r'[∀∃]')
CONNECTIVE_RE = re.compile(r'[∧∨→⊕]')


def compositional_depth(fol: str) -> int:
    return len(QUANTIFIER_RE.findall(fol)) + len(CONNECTIVE_RE.findall(fol))


def process_folio(split: str) -> list[dict]:
    fname = TEMP / f"full_tasksource_folio_default_{split}.json"
    logger.info(f"Loading FOLIO {split} from {fname}")
    rows = json.loads(fname.read_text())
    logger.info(f"  {len(rows)} stories in FOLIO {split}")

    pairs = []
    mismatches = 0
    for row in rows:
        story_id = str(row["story_id"])
        label = row.get("label")
        premises_nl = [p.strip() for p in row["premises"].split("\n") if p.strip()]
        premises_fol = [f.strip() for f in row["premises-FOL"].split("\n") if f.strip()]

        if len(premises_nl) != len(premises_fol):
            mismatches += 1
            logger.debug(f"Story {story_id}: premises count mismatch {len(premises_nl)} NL vs {len(premises_fol)} FOL — skipping")
            continue

        example_id = row.get("example_id", "")
        for i, (nl, fol) in enumerate(zip(premises_nl, premises_fol)):
            pairs.append({
                "id": f"folio_{split}_{example_id}_p{i}",
                "nl_sentence": nl,
                "gold_fol": fol,
                "compositional_depth": compositional_depth(fol),
                "dataset_source": "folio",
                "metadata_fold": None,
                "story_id": story_id,
                "label": label,
            })

        # Also add conclusion pairs
        conc_nl = row.get("conclusion", "").strip()
        conc_fol = row.get("conclusion-FOL", "").strip()
        if conc_nl and conc_fol:
            pairs.append({
                "id": f"folio_{split}_{example_id}_conc",
                "nl_sentence": conc_nl,
                "gold_fol": conc_fol,
                "compositional_depth": compositional_depth(conc_fol),
                "dataset_source": "folio",
                "metadata_fold": None,
                "story_id": story_id,
                "label": label,
            })

    logger.info(f"  {len(pairs)} NL-FOL pairs extracted (mismatches skipped: {mismatches})")
    return pairs


def process_proofwriter(limit: int = 200) -> list[dict]:
    fname = TEMP / "full_tasksource_proofwriter_default_test.json"
    logger.info(f"Loading ProofWriter test from {fname}")
    all_rows = json.loads(fname.read_text())
    logger.info(f"  {len(all_rows)} total rows")

    # Filter to RelNeg-OWA rows (RelNeg is in the 'id' field, not 'config')
    relneg = [r for r in all_rows if "RelNeg" in r.get("id", "")]
    logger.info(f"  {len(relneg)} RelNeg rows found")
    sample = relneg[:limit] if len(relneg) >= limit else relneg
    logger.info(f"  Using {len(sample)} rows")

    pairs = []
    for i, row in enumerate(sample):
        theory_text = row.get("theory", "")
        qdep = row.get("QDep", 0)
        sentences = [s.strip() for s in theory_text.split("\n") if s.strip()]
        # Keep relational sentences (heuristic: contain 'is' or 'are')
        relational = [s for s in sentences if re.search(r'\b(is|are)\b', s, re.I)]
        for j, sent in enumerate(relational):
            pairs.append({
                "id": f"proofwriter_test_{i}_s{j}",
                "nl_sentence": sent,
                "gold_fol": None,
                "compositional_depth": int(qdep),
                "dataset_source": "proofwriter",
                "metadata_fold": "proofwriter_depth_ablation",
                "story_id": None,
                "label": row.get("answer"),
            })

    logger.info(f"  {len(pairs)} ProofWriter sentences extracted")
    return pairs


def process_malls(limit: int = 1000) -> list[dict]:
    fname = TEMP / "full_yuan-yang_MALLS-v0_default_train.json"
    logger.info(f"Loading MALLS from {fname}")
    rows = json.loads(fname.read_text())
    logger.info(f"  {len(rows)} total MALLS rows")
    sample = rows[:limit]

    pairs = []
    for i, row in enumerate(sample):
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
            "compositional_depth": depth,
            "dataset_source": "malls",
            "metadata_fold": "malls_supplement",
            "story_id": None,
            "label": None,
        })

    logger.info(f"  {len(pairs)} MALLS pairs after depth>=2 filter")
    return pairs


def build_alignment_validation_split(pilot_rows: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Stratified sample of 30 rows for alignment-validation split."""
    buckets = {
        "low": [r for r in pilot_rows if 2 <= r["compositional_depth"] <= 3],
        "mid": [r for r in pilot_rows if 4 <= r["compositional_depth"] <= 5],
        "high": [r for r in pilot_rows if r["compositional_depth"] >= 6],
    }
    targets = {"low": 12, "mid": 12, "high": 6}
    val_ids = set()
    for bucket, target in targets.items():
        pool = buckets[bucket]
        n = min(target, len(pool))
        chosen = rng.sample(pool, n)
        for r in chosen:
            val_ids.add(r["id"])
        logger.info(f"  Alignment validation {bucket}: {n}/{target} sampled")

    val = [r for r in pilot_rows if r["id"] in val_ids]
    pilot = [r for r in pilot_rows if r["id"] not in val_ids]
    return pilot, val


@logger.catch(reraise=True)
def main() -> None:
    rng = random.Random(42)
    (WS / "logs").mkdir(exist_ok=True)

    # --- FOLIO ---
    folio_train = process_folio("train")
    folio_val = process_folio("validation")
    folio_all = folio_train + folio_val

    # Filter by compositional depth >= 2
    folio_filtered = [r for r in folio_all if r["compositional_depth"] >= 2]
    logger.info(f"FOLIO after depth>=2 filter: {len(folio_filtered)} rows")

    if len(folio_filtered) < 150:
        logger.warning("Fewer than 150 rows at depth>=2, lowering threshold to >=1")
        folio_filtered = [r for r in folio_all if r["compositional_depth"] >= 1]
        logger.info(f"FOLIO after depth>=1 filter: {len(folio_filtered)} rows")

    # Deduplicate FOLIO on nl_sentence
    seen_nl = set()
    folio_deduped = []
    for r in folio_filtered:
        key = r["nl_sentence"].strip().lower()
        if key not in seen_nl:
            seen_nl.add(key)
            folio_deduped.append(r)
    logger.info(f"FOLIO after dedup: {len(folio_deduped)} rows")

    # Assign alignment-validation vs pilot fold for FOLIO
    folio_pilot, folio_align_val = build_alignment_validation_split(folio_deduped, rng)
    for r in folio_pilot:
        r["metadata_fold"] = "pilot"
    for r in folio_align_val:
        r["metadata_fold"] = "alignment_validation"

    # --- ProofWriter ---
    try:
        proofwriter_rows = process_proofwriter(limit=200)
    except Exception:
        logger.warning("ProofWriter processing failed — skipping")
        proofwriter_rows = []

    # --- MALLS ---
    try:
        malls_rows = process_malls(limit=1000)
    except Exception:
        logger.warning("MALLS processing failed — skipping")
        malls_rows = []

    # --- Merge & deduplicate ---
    all_rows = folio_pilot + folio_align_val + proofwriter_rows + malls_rows
    logger.info(f"Total before final dedup: {len(all_rows)}")

    final_dedup: list[dict] = []
    all_seen_nl = set()
    for r in all_rows:
        key = r["nl_sentence"].strip().lower()
        if key not in all_seen_nl:
            all_seen_nl.add(key)
            final_dedup.append(r)
    logger.info(f"Total after final dedup: {len(final_dedup)}")

    # --- Validate assertions ---
    pilot_and_val = [r for r in final_dedup if r["metadata_fold"] in ("pilot", "alignment_validation")]
    assert all(r["gold_fol"] is not None for r in pilot_and_val), \
        "Some pilot/alignment_validation rows have null gold_fol"
    assert len(pilot_and_val) >= 150, \
        f"Pilot+validation only {len(pilot_and_val)} rows (need >= 150)"
    logger.info(f"Assertions passed: {len(pilot_and_val)} pilot+val rows with non-null gold_fol")

    # --- Convert to schema format ---
    # Schema: {datasets: [{dataset, examples: [{input, output, metadata_*}]}]}
    # input = nl_sentence, output = gold_fol (empty string if null for ProofWriter)
    examples = []
    for r in final_dedup:
        ex = {
            "input": r["nl_sentence"],
            "output": r["gold_fol"] if r["gold_fol"] is not None else "",
            "metadata_id": r["id"],
            "metadata_compositional_depth": r["compositional_depth"],
            "metadata_dataset_source": r["dataset_source"],
            "metadata_fold": r["metadata_fold"] if r["metadata_fold"] else "pilot",
            "metadata_story_id": r["story_id"] if r["story_id"] else "",
            "metadata_label": r["label"] if r["label"] else "",
        }
        examples.append(ex)

    output_doc = {
        "metadata": {
            "description": "FOLIO + ProofWriter + MALLS NL-FOL dataset for locality pilot",
            "total_rows": len(examples),
            "sources": ["tasksource/folio", "tasksource/proofwriter", "yuan-yang/MALLS-v0"],
            "depth_filter": ">=2 quantifiers+connectives for FOLIO and MALLS",
        },
        "datasets": [
            {
                "dataset": "folio+proofwriter+malls_nl_fol",
                "examples": examples,
            }
        ],
    }

    # --- Save ---
    out_path = WS / "data_out.json"
    out_path.write_text(json.dumps(output_doc, indent=2, ensure_ascii=False))

    size_mb = out_path.stat().st_size / 1e6
    assert size_mb < 50, f"data_out.json is {size_mb:.1f}MB (>50MB limit)"
    logger.info(f"Saved {len(final_dedup)} rows to {out_path} ({size_mb:.2f}MB)")

    # Stats
    by_source = {}
    by_fold = {}
    for r in final_dedup:
        by_source[r["dataset_source"]] = by_source.get(r["dataset_source"], 0) + 1
        by_fold[r["metadata_fold"]] = by_fold.get(r["metadata_fold"], 0) + 1
    logger.info(f"By source: {by_source}")
    logger.info(f"By fold: {by_fold}")

    depth_dist = {}
    for r in final_dedup:
        d = r["compositional_depth"]
        bucket = f"depth_{d}" if d <= 6 else "depth_7+"
        depth_dist[bucket] = depth_dist.get(bucket, 0) + 1
    logger.info(f"Depth distribution: {dict(sorted(depth_dist.items()))}")


if __name__ == "__main__":
    main()
