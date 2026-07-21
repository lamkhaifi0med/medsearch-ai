import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, Search, Sparkles, X } from "lucide-react";
import {
  explain,
  getHealth,
  search,
  type ExplainResponse,
  type HealthResponse,
  type SearchFilters,
  type SearchResponse,
} from "./api";
import CaseCard from "./components/CaseCard";
import CaseDrawer from "./components/CaseDrawer";
import CohortStrip from "./components/CohortStrip";
import ExplanationPanel from "./components/ExplanationPanel";
import FilterRail from "./components/FilterRail";

const EXAMPLES = [
  "67-year-old man with heart failure, dyspnea and leg edema",
  "child with recurrent seizures and developmental delay",
  "young woman with joint pain, rash and fatigue",
];

export default function App() {
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<SearchFilters>({});
  const [k, setK] = useState(10);
  const [thorough, setThorough] = useState(false);

  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [explainData, setExplainData] = useState<ExplainResponse | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);

  const [openCase, setOpenCase] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // live system status in the header
  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  // "/" focuses search from anywhere
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const runSearch = useCallback(
    async (q?: string) => {
      const text = (q ?? query).trim();
      if (text.length < 3) return;
      if (q) setQuery(q);
      setSearching(true);
      setSearchError(null);
      setSelected(new Set());
      setPanelOpen(false);
      setExplainData(null);
      try {
        setResponse(await search(text, k, filters, thorough));
      } catch (e) {
        setSearchError(String(e));
        setResponse(null);
      } finally {
        setSearching(false);
      }
    },
    [query, k, filters, thorough],
  );

  const runExplain = useCallback(
    async (ids: string[]) => {
      if (!response || ids.length === 0) return;
      setPanelOpen(true);
      setExplaining(true);
      setExplainError(null);
      setExplainData(null);
      try {
        setExplainData(await explain(response.query, ids.slice(0, 5)));
      } catch (e) {
        setExplainError(String(e));
      } finally {
        setExplaining(false);
      }
    },
    [response],
  );

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < 5) next.add(id);
      return next;
    });
  };

  return (
    <div className="min-h-screen">
      {/* top bar */}
      <header className="border-b border-line bg-chart">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-6 py-3">
          <Activity className="size-5 text-teal" strokeWidth={2.2} />
          <span className="text-[15px] font-semibold tracking-tight">
            MedSearch AI
          </span>
          <span className="rounded-sm bg-teal-wash px-1.5 py-0.5 font-mono text-[10.5px] font-medium text-teal-deep">
            evidence retrieval
          </span>
          <span className="ml-auto flex items-center gap-2 font-mono text-[11px] text-ink-faint">
            <span
              className={`size-1.5 rounded-full ${
                health ? "bg-conf-high" : "bg-conf-mod pulse-soft"
              }`}
              title={
                health
                  ? `online · qdrant ${health.qdrant ? "ok" : "down"} · redis ${health.redis ? "ok" : "down"}`
                  : "connecting…"
              }
            />
            {response
              ? `${response.results.length} of ${(health?.points_indexed ?? 24348).toLocaleString("en-US")} cases · ${response.took_ms} ms${response.reranked ? " · reranked" : ""}`
              : `${(health?.points_indexed ?? 24348).toLocaleString("en-US")} indexed cases`}
          </span>
        </div>
      </header>

      {/* search bar */}
      <div className="border-b border-line bg-chart">
        <div className="mx-auto max-w-7xl px-6 py-5">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              runSearch();
            }}
            className="flex gap-2"
          >
            <div className="relative flex-1">
              <Search className="absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-ink-faint" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Describe the patient — presentation, history, findings…"
                className="w-full rounded-md border border-line bg-paper py-2.5 pl-10 pr-10 text-[14px] placeholder:text-ink-faint focus:border-teal"
              />
              <kbd className="pointer-events-none absolute right-3.5 top-1/2 -translate-y-1/2 rounded border border-line bg-chart px-1.5 font-mono text-[11px] text-ink-faint">
                /
              </kbd>
            </div>
            <button
              type="submit"
              disabled={searching || query.trim().length < 3}
              className="rounded-md bg-teal px-5 py-2.5 text-[14px] font-medium text-white transition-colors hover:bg-teal-deep disabled:opacity-40"
            >
              {searching ? "Searching…" : "Find similar cases"}
            </button>
          </form>

          <label
            className="mt-2.5 inline-flex cursor-pointer select-none items-center gap-2"
            title="Runs an extra AI cross-check on the top 50 candidates (measured nDCG 0.94 vs 0.66). Adds ~0.5s."
          >
            <input
              type="checkbox"
              checked={thorough}
              onChange={(e) => setThorough(e.target.checked)}
              className="peer sr-only"
            />
            <span className="relative h-4 w-7 rounded-full bg-line transition-colors peer-checked:bg-teal after:absolute after:left-0.5 after:top-0.5 after:size-3 after:rounded-full after:bg-white after:transition-transform peer-checked:after:translate-x-3" />
            <span className="flex items-center gap-1.5 text-[12.5px] text-ink-soft">
              <Sparkles className="size-3.5 text-teal" />
              Thorough mode
              <span className="font-mono text-[10.5px] text-ink-faint">
                cross-encoder rerank · +0.5s
              </span>
            </span>
          </label>

          {!response && !searching && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="text-[12px] text-ink-faint">Try:</span>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => runSearch(ex)}
                  className="rounded-full border border-line bg-paper px-3 py-1 text-[12.5px] text-ink-soft hover:border-teal hover:text-teal-deep"
                >
                  {ex}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* body */}
      <main className="mx-auto flex max-w-7xl flex-col gap-6 px-6 py-6 lg:flex-row lg:gap-8">
        <FilterRail
          filters={filters}
          onChange={setFilters}
          k={k}
          onKChange={setK}
          disabled={searching}
        />

        <div className="min-w-0 flex-1">
          {searchError && (
            <div className="rounded border border-conf-weak/30 bg-conf-weak-wash px-4 py-3 text-[13px] text-conf-weak">
              Search failed: {searchError}. Check that the API is running on
              port 8000.
            </div>
          )}

          {searching && (
            <div className="space-y-3">
              {[0, 1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="pulse-soft h-28 rounded-md bg-line-soft"
                  style={{ animationDelay: `${i * 150}ms` }}
                />
              ))}
            </div>
          )}

          {!response && !searching && !searchError && (
            <div className="flex flex-col items-center py-20 text-center">
              {/* ambient ECG — quiet, alive, on brand */}
              <svg
                width="420"
                height="56"
                viewBox="0 0 420 56"
                aria-hidden="true"
                className="max-w-full"
              >
                <line
                  x1="0"
                  y1="28"
                  x2="420"
                  y2="28"
                  stroke="var(--color-line)"
                  strokeWidth="1"
                />
                <path
                  className="ecg-ambient"
                  d="M 0 28 L 96 28 L 104 22 L 112 34 L 120 8 L 128 42 L 136 28 L 168 28 L 176 24 L 184 28 L 300 28 L 308 22 L 316 34 L 324 8 L 332 42 L 340 28 L 372 28 L 380 24 L 388 28 L 420 28"
                  fill="none"
                  stroke="var(--color-teal)"
                  strokeWidth="1.6"
                  strokeLinejoin="round"
                />
              </svg>
              <p className="mt-6 max-w-md text-[14px] leading-relaxed text-ink-soft">
                Describe a patient above and retrieve the most clinically
                similar historical cases — each with an explainable, cited
                rationale.
              </p>
              <p className="mt-2 font-mono text-[11.5px] text-ink-faint">
                press{" "}
                <kbd className="rounded border border-line bg-chart px-1">
                  /
                </kbd>{" "}
                to start typing
              </p>
            </div>
          )}

          {response && !searching && (
            <>
              <div className="mb-3 flex items-center justify-between">
                <h1 className="text-[13px] font-semibold uppercase tracking-[0.08em] text-ink-faint">
                  Retrieved evidence
                </h1>
                <span className="font-mono text-[11px] text-ink-faint">
                  select up to 5 cases to compare rationales
                </span>
              </div>

              <div className="space-y-3">
                {response.results.length > 0 && (
                  <CohortStrip results={response.results} />
                )}
                {response.results.map((r, i) => (
                  <CaseCard
                    key={r.case_id}
                    result={r}
                    rank={i + 1}
                    index={i}
                    selected={selected.has(r.case_id)}
                    onToggleSelect={() => toggleSelect(r.case_id)}
                    onOpen={() => setOpenCase(r.case_id)}
                    onExplain={() => runExplain([r.case_id])}
                  />
                ))}
                {response.results.length === 0 && (
                  <p className="py-12 text-center text-[13.5px] text-ink-soft">
                    No cases match these filters. Widen the age range or clear
                    the outcome filter.
                  </p>
                )}
              </div>
            </>
          )}
        </div>

        {/* explanation panel */}
        {panelOpen && (
          <div className="w-[26rem] shrink-0">
            <div className="sticky top-6 h-[calc(100vh-6rem)] overflow-hidden rounded-md border border-line">
              <ExplanationPanel
                data={explainData}
                loading={explaining}
                error={explainError}
                onClose={() => setPanelOpen(false)}
                onCiteClick={setOpenCase}
              />
            </div>
          </div>
        )}
      </main>

      {/* sticky selection action bar */}
      {selected.size > 0 && !panelOpen && (
        <div className="rise-in fixed bottom-6 left-1/2 z-40 flex items-center gap-3 rounded-full border border-line bg-chart py-2 pl-5 pr-2 shadow-[0_6px_24px_rgba(26,46,53,0.14)]">
          <span className="font-mono text-[12.5px] tabular-nums text-ink-soft">
            {selected.size} {selected.size === 1 ? "case" : "cases"} selected
          </span>
          <button
            type="button"
            onClick={() => runExplain([...selected])}
            className="flex items-center gap-1.5 rounded-full bg-teal px-4 py-1.5 text-[13px] font-medium text-white hover:bg-teal-deep"
          >
            <Sparkles className="size-3.5" />
            Explain matches
          </button>
          <button
            type="button"
            onClick={() => setSelected(new Set())}
            aria-label="clear selection"
            className="rounded-full p-1.5 text-ink-faint hover:bg-line-soft hover:text-ink"
          >
            <X className="size-4" />
          </button>
        </div>
      )}

      {/* footer disclaimer — always visible */}
      <footer className="border-t border-line bg-chart">
        <div className="mx-auto max-w-7xl px-6 py-3">
          <p className="text-[11px] text-ink-faint">
            MedSearch AI retrieves historical evidence for clinical decision
            support. It never diagnoses and never recommends treatment. All
            medical decisions remain with the treating physician.
          </p>
        </div>
      </footer>

      <CaseDrawer caseId={openCase} onClose={() => setOpenCase(null)} />
    </div>
  );
}
