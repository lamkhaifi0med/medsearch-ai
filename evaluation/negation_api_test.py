"""Targeted negation tests against the live API."""
import json
import time
import urllib.request

API = "http://localhost:8000/api/v1"


def search(q, k=8, rerank=False):
    req = urllib.request.Request(
        f"{API}/search",
        data=json.dumps({"query": q, "k": k, "rerank": rerank, "filters": {}}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r), int((time.time() - t0) * 1000)


print("N1  Query with explicit negation: 'abdominal pain and vomiting, no fever, no diarrhea'")
r, ms = search("abdominal pain and vomiting, no fever, no diarrhea")
for c in r["results"]:
    flags = ",".join(c["quality_flags"]) or "-"
    print(f"  {c['case_id']}  score={c['score']:.3f}  flags={flags}")
print(f"  took={ms}ms")

print("=" * 70)
print("N2  Same clinical picture WITHOUT the negations: 'abdominal pain and vomiting'")
r2, ms2 = search("abdominal pain and vomiting")
ids1 = [c["case_id"] for c in r["results"]]
ids2 = [c["case_id"] for c in r2["results"]]
print(f"  N1 top-8: {ids1}")
print(f"  N2 top-8: {ids2}")
print(f"  overlap: {len(set(ids1) & set(ids2))}/8 — negation query should reorder/replace some")

print("=" * 70)
print("N3  'denies chest pain' trap: query 'chest pain with palpitations'")
r3, ms3 = search("chest pain with palpitations", k=10)
flagged = [c for c in r3["results"] if "negation_conflict" in c["quality_flags"]]
print(f"  {len(r3['results'])} results, {len(flagged)} carry negation_conflict (penalized cases that only NEGATE chest pain)")
for c in flagged:
    print(f"    penalized: {c['case_id']} score={c['score']:.3f}")
    print(f"      | {c['snippet'][:110]}")

print("=" * 70)
print("N4  Thorough mode + negation together")
r4, ms4 = search("abdominal pain and vomiting, no fever", k=5, rerank=True)
print(f"  reranked={r4['reranked']} took={ms4}ms")
for c in r4["results"]:
    flags = ",".join(c["quality_flags"]) or "-"
    print(f"  {c['case_id']}  score={c['score']:.3f}  flags={flags}")
