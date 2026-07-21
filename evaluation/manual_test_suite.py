"""One-shot manual test suite for MedSearch AI — prints a compact report."""
import json
import time
import urllib.request

API = "http://localhost:8000/api/v1"


def post(path, payload, timeout=180):
    req = urllib.request.Request(
        f"{API}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), int((time.time() - t0) * 1000), None
    except urllib.error.HTTPError as e:
        return None, int((time.time() - t0) * 1000), f"HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:  # noqa: BLE001
        return None, int((time.time() - t0) * 1000), str(e)[:200]


def show(res):
    for c in res["results"]:
        print(f"  {c['case_id']}  sex={c['sex']} age={c['age']} outcome={c['outcome_class']} score={c['score']:.3f}")
        print(f"    | {c['snippet'][:110]}")


def search(q, filters=None, rerank=False, k=5):
    return post("/search", {"query": q, "k": k, "rerank": rerank, "filters": filters or {}})


print("=" * 70)
print("T1  FILTER: sex=female — 'elderly patient with chest pain'")
r, ms, err = search("elderly patient with chest pain", {"sex": "female"})
if err: print("  ERROR:", err)
else:
    show(r); print(f"  took={ms}ms  PASS={'yes' if all(c['sex']=='female' for c in r['results']) else 'NO — male leaked in'}")

print("=" * 70)
print("T2  FILTER: outcome=deceased — 'massive stroke neurological deterioration'")
r, ms, err = search("massive stroke with neurological deterioration", {"outcome_class": "deceased"})
if err: print("  ERROR:", err)
else:
    show(r); print(f"  took={ms}ms  PASS={'yes' if all(c['outcome_class']=='deceased' for c in r['results']) else 'NO'}")

print("=" * 70)
print("T3  FILTER combo: female + age>=60 + improved — 'heart failure with dyspnea'")
r, ms, err = search("heart failure with shortness of breath", {"sex": "female", "age_min": 60, "outcome_class": "improved"})
if err: print("  ERROR:", err)
else:
    ok = all(c["sex"] == "female" and (c["age"] is None or c["age"] >= 60) and c["outcome_class"] == "improved" for c in r["results"])
    show(r); print(f"  took={ms}ms  PASS={'yes' if ok else 'NO'}")

print("=" * 70)
print("T4  ABBREVIATIONS: 'pt c/o SOB and CP, hx of HTN and T2DM'")
r, ms, err = search("pt c/o SOB and CP, hx of HTN and T2DM")
if err: print("  ERROR:", err)
else: show(r); print(f"  took={ms}ms")

print("=" * 70)
print("T5  ABBREV+LABS: '55M w/ DKA, K+ 6.2, Cr 2.1'")
r, ms, err = search("55M w/ DKA, K+ 6.2, Cr 2.1")
if err: print("  ERROR:", err)
else: show(r); print(f"  took={ms}ms")

print("=" * 70)
print("T6  SAFETY: nonsense query 'purple elephant quantum bicycle'")
r, ms, err = search("purple elephant quantum bicycle")
if err: print("  ERROR:", err)
else: print(f"  returned {len(r['results'])} results without crashing, top score={r['results'][0]['score']:.3f}  took={ms}ms")

print("=" * 70)
print("T7  SAFETY: minimal query 'pain'")
r, ms, err = search("pain")
if err: print("  ERROR:", err)
else: print(f"  returned {len(r['results'])} results, took={ms}ms")

print("=" * 70)
print("T8  MULTILINGUAL: 'douleur thoracique chez un homme âgé' (French)")
r, ms, err = search("douleur thoracique chez un homme âgé")
if err: print("  ERROR:", err)
else: show(r); print(f"  took={ms}ms")

print("=" * 70)
print("T9  EDGE CORPUS: 'Takotsubo cardiomyopathy after emotional stress'")
r, ms, err = search("Takotsubo cardiomyopathy after emotional stress", k=3)
if err: print("  ERROR:", err)
else: show(r); print(f"  took={ms}ms")

print("=" * 70)
print("T10 EDGE CORPUS: 'newborn with jaundice and poor feeding'")
r, ms, err = search("newborn with jaundice and poor feeding", k=3)
if err: print("  ERROR:", err)
else: show(r); print(f"  took={ms}ms")

print("=" * 70)
print("T11 RERANK SHOWCASE: paraphrased real case, thorough mode")
# acn-16444 paraphrased in different words (warfarin/INR/factor IX case):
para = ("middle aged woman anticoagulated with coumadin, extremely elevated INR above 13, "
        "nosebleeds and blood in stool, later given a prothrombin complex product and "
        "suffered a heart attack and cardiac arrest")
r, ms, err = search(para, rerank=True, k=5)
if err: print("  ERROR:", err)
else:
    show(r)
    rank = next((i + 1 for i, c in enumerate(r["results"]) if c["case_id"] == "acn-16444"), None)
    print(f"  took={ms}ms reranked={r['reranked']}  source case acn-16444 rank: {rank}  PASS={'yes' if rank == 1 else ('partial' if rank else 'NO')}")

print("=" * 70)
print("T12 EXPLAIN (RAG): warfarin+melena query, top case acn-16444")
t0 = time.time()
r, ms, err = post("/explain", {"query": "patient on warfarin presenting with melena", "case_ids": ["acn-16444"]}, timeout=300)
if err: print("  ERROR:", err)
else:
    print(f"  model={r.get('model_used')} degraded={r.get('degraded')} cached={r.get('cached')} took={ms}ms")
    for ce in r.get("cases", []):
        print(f"  case {ce['case_id']} confidence={ce.get('confidence')}")
        for f in ce.get("similarity_factors", [])[:3]:
            print(f"    + {f['factor']}: {f['detail'][:100]}  cites={f['citations']}")
        for d in ce.get("differences", [])[:2]:
            print(f"    - {d['detail'][:100]}  cites={d['citations']}")
    # citation check
    all_cits = []
    for ce in r.get("cases", []):
        for f in ce.get("similarity_factors", []) + ce.get("differences", []) + ce.get("treatments_observed", []):
            all_cits.extend(f.get("citations", []))
    valid = all(c.startswith("acn-") for c in all_cits) if all_cits else False
    print(f"  citations found: {len(all_cits)}, all valid case IDs: {valid}")
    print(f"  disclaimer present: {bool(r.get('disclaimer'))}")

print("=" * 70)
print("T13 EXPLAIN CACHE: same request again")
r2, ms2, err = post("/explain", {"query": "patient on warfarin presenting with melena", "case_ids": ["acn-16444"]}, timeout=300)
if err: print("  ERROR:", err)
else: print(f"  cached={r2.get('cached')} took={ms2}ms (first call was {ms}ms)  PASS={'yes' if r2.get('cached') and ms2 < ms else 'check'}")

print("=" * 70)
print("T14 EXPLAIN MULTI-CASE: 3 cases comparison")
r, ms, err = search("diabetic patient with foot ulcer and fever", k=3)
ids = [c["case_id"] for c in r["results"]] if r else []
r3, ms3, err = post("/explain", {"query": "diabetic patient with foot ulcer and fever", "case_ids": ids}, timeout=300)
if err: print("  ERROR:", err)
else:
    print(f"  explained {len(r3.get('cases', []))} cases, degraded={r3.get('degraded')} took={ms3}ms")
    if r3.get("cohort_summary"): print(f"  cohort summary: {r3['cohort_summary'][:200]}")

print("=" * 70)
print("T15 NEGATION: 'abdominal pain and vomiting, no fever, no diarrhea'")
r, ms, err = search("abdominal pain and vomiting, no fever, no diarrhea", k=5)
if err: print("  ERROR:", err)
else:
    for c in r["results"]:
        flags = ",".join(c["quality_flags"]) or "-"
        print(f"  {c['case_id']}  score={c['score']:.3f} flags={flags}")
        print(f"    | {c['snippet'][:100]}")
    print(f"  took={ms}ms  (cases asserting fever/diarrhea should carry negation_conflict + sit lower)")

print("=" * 70)
print("SUITE COMPLETE")
