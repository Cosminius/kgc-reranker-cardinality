"""Wrong first choices of the cross-encoder and the bi-encoder on test queries with 100 or more
known answers: the share of the right type and, with --wikidata (CoDEx-M), the share that are
Wikidata statements and the queries that CARF repairs although the cross-encoder's first choice
is not a Wikidata statement.

Run after scripts/evaluate.py, which saves the per-query scores:

    python scripts/error_analysis.py --config configs/codex-m.yaml --seed 0 --wikidata
"""

import argparse
import json

from _common import REPO_ROOT, load_config, load_scored, results_prefix
from src.datasets import load_candidates, load_triples
from src.errors import (repaired_by_fusion, right_type_share, wikidata_check, wikidata_statements,
                        wikidata_triple, wrong_first_choices)
from src.evaluation import select_and_evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--arm", choices=["unmasked", "masked"], default="unmasked")
    p.add_argument("--wikidata", action="store_true", help="look the answers up in Wikidata (CoDEx-M)")
    args = p.parse_args()
    cfg = load_config(args.config)

    prefix = results_prefix(cfg, args.seed, args.arm)
    data_dir = REPO_ROOT / cfg["simkgc_data_dir"]
    valid, test = load_scored(prefix, "valid"), load_scored(prefix, "test")
    records = {r["query_id"]: r for r in
               load_candidates(REPO_ROOT / cfg["mining_dir"], "test", int(cfg.get("top_k", 50)))}

    wrong = wrong_first_choices(test, records)
    out = {"wrong": {m: len(v) for m, v in wrong.items()},
           "right_type": right_type_share(wrong, records, load_triples(data_dir, "train"))}
    for m in wrong:
        print(f"{m:13s} wrong first choices {out['wrong'][m]:6,d} | right type {out['right_type'][m]:.0%}")

    if args.wikidata:
        relation_ids = {name: pid for pid, name in
                        json.loads((data_dir / "relations.json").read_text(encoding="utf-8")).items()}
        out["wikidata"] = wikidata_check(wrong, records, relation_ids)
        for m, row in out["wikidata"].items():
            print(f"{m:13s} {row['sample']} sampled errors | in Wikidata {row['true']:.0%} "
                  f"| gold answers found {row['gold_found']:.0%}")

        _, configs, _ = select_and_evaluate(valid, test, float(cfg.get("alpha_grid_step", 0.05)))
        repaired = repaired_by_fusion(test, records, configs)
        found = wikidata_statements([wikidata_triple(records[q], e, relation_ids) for q, e in repaired])
        names = {e["entity_id"]: e.get("entity", e["entity_id"]) for e in
                 json.loads((data_dir / "entities.json").read_text(encoding="utf-8"))}
        out["repaired"] = []
        for q, e in repaired:
            if wikidata_triple(records[q], e, relation_ids) in found:
                continue
            rec = records[q]
            row = {"query": [names.get(rec["head_id"]), rec["relation"]], "cardinality": test[q]["n_filtered"] + 1,
                   "gold": names.get(rec["gold_entity_id"]), "cross_encoder": names.get(e),
                   "ranks": [configs[c][q] for c in configs]}
            out["repaired"].append(row)
        print(f"{len(out['repaired'])} queries repaired by CARF whose cross-encoder choice is not in Wikidata")

    with open(f"{prefix}_error_analysis.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
