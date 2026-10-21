"""What the cross-encoder and the bi-encoder rank first when it is not the gold answer, on queries
with many known answers."""

import json
import random
import time
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np

MODELS = {"Cross-encoder": "rerank_scores", "Bi-encoder": "bi_scores"}
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "kgc-reranker-cardinality (academic research)"


def wrong_first_choices(scored, records, min_cardinality=100):
    """{model: [(query_id, entity_id)]} for queries whose gold answer is among the candidates and
    whose first choice is another entity."""
    out = {m: [] for m in MODELS}
    for q in sorted(scored):
        r = scored[q]
        if r["source"] != "reranker" or r["n_filtered"] + 1 < min_cardinality:
            continue
        for m, key in MODELS.items():
            top = int(np.argmax(r[key]))
            if top != r["gold_idx_in_topk"]:
                out[m].append((q, records[q]["top_k_entity_ids"][top]))
    return out


def right_type_share(wrong, records, train_triples):
    """Share of the wrong first choices that answer the same relation for another head in training."""
    answers = defaultdict(set)
    for t in train_triples:
        answers[t["relation"]].add(t["tail_id"])
        answers[f"inverse {t['relation']}"].add(t["head_id"])
    return {m: float(np.mean([e in answers[records[q]["relation"]] for q, e in pairs]))
            for m, pairs in wrong.items()}


def wikidata_triple(record, entity, relation_ids):
    """(subject, property, object) of the query with `entity` as answer, in Wikidata direction."""
    rel = record["relation"]
    if rel.startswith("inverse "):
        return entity, relation_ids[rel[len("inverse "):]], record["head_id"]
    return record["head_id"], relation_ids[rel], entity


def wikidata_statements(triples, batch=100):
    """The triples that are Wikidata statements (any rank)."""
    found = set()
    for i in range(0, len(triples), batch):
        values = " ".join(f"(wd:{s} p:{p} ps:{p} wd:{o})" for s, p, o in triples[i:i + batch])
        query = f"SELECT ?s ?pp ?o WHERE {{ VALUES (?s ?pp ?ps ?o) {{ {values} }} ?s ?pp ?st . ?st ?ps ?o . }}"
        data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
        req = urllib.request.Request(WIKIDATA_SPARQL, data=data, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            rows = json.load(resp)["results"]["bindings"]
        found |= {tuple(b[k]["value"].rsplit("/", 1)[-1] for k in ("s", "pp", "o")) for b in rows}
        time.sleep(2)
    return found


def wikidata_check(wrong, records, relation_ids, sample=300, seed=0):
    """Share of a random sample of wrong first choices that are Wikidata statements, and the share
    of the gold answers of the same queries that the lookup finds."""
    rng = random.Random(seed)
    out = {}
    for m, pairs in wrong.items():
        picked = rng.sample(pairs, sample)
        preds = [wikidata_triple(records[q], e, relation_ids) for q, e in picked]
        golds = [wikidata_triple(records[q], records[q]["gold_entity_id"], relation_ids) for q, _ in picked]
        found = wikidata_statements(preds + golds)
        out[m] = {"sample": sample, "true": sum(t in found for t in preds) / sample,
                  "gold_found": sum(t in found for t in golds) / sample}
    return out


def repaired_by_fusion(scored, records, configs, min_cardinality=100):
    """[(query_id, cross-encoder's first choice)] for queries that the bi-encoder and CARF rank
    correctly and the cross-encoder does not."""
    return [(q, records[q]["top_k_entity_ids"][int(np.argmax(scored[q]["rerank_scores"]))])
            for q in sorted(scored)
            if scored[q]["source"] == "reranker" and scored[q]["n_filtered"] + 1 >= min_cardinality
            and configs["Bi-encoder"][q] == 1 and configs["Per-bin fusion"][q] == 1
            and configs["Reranker"][q] > 1]
