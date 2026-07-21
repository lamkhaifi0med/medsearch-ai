import { AlertTriangle, ArrowRightLeft, Check, Minus, Pill, X } from "lucide-react";
import type { ExplainResponse } from "../api";

interface Props {
  data: ExplainResponse | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onCiteClick: (caseId: string) => void;
}

const CONF_STYLE: Record<string, string> = {
  high: "bg-conf-high-wash text-conf-high",
  moderate: "bg-conf-mod-wash text-conf-mod",
  weak: "bg-conf-weak-wash text-conf-weak",
};

function Citation({ id, onClick }: { id: string; onClick: (id: string) => void }) {
  return (
    <button
      type="button"
      onClick={() => onClick(id)}
      className="font-mono text-[11px] text-teal-deep underline-offset-2 hover:underline"
    >
      [{id}]
    </button>
  );
}

export default function ExplanationPanel({ data, loading, error, onClose, onCiteClick }: Props) {
  return (
    <section className="flex h-full flex-col border-l border-line bg-chart">
      <header className="flex items-center justify-between border-b border-line px-5 py-3">
        <div>
          <h2 className="text-[15px] font-semibold">Why these cases matched</h2>
          {data && !loading && (
            <p className="mt-0.5 font-mono text-[11px] text-ink-faint">
              {data.model_used}
              {data.cached && " · cached"}
              {data.took_ms > 0 && ` · ${(data.took_ms / 1000).toFixed(1)}s`}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1.5 text-ink-faint hover:bg-line-soft hover:text-ink"
          aria-label="close explanation panel"
        >
          <X className="size-4" />
        </button>
      </header>

      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-4">
        {loading && (
          <div className="space-y-3">
            <p className="text-[13px] text-ink-soft">
              Reading the retrieved cases and writing a cited comparison…
            </p>
            {[0, 1, 2].map((i) => (
              <div key={i} className="pulse-soft h-16 rounded bg-line-soft" style={{ animationDelay: `${i * 200}ms` }} />
            ))}
          </div>
        )}

        {error && (
          <div className="rounded border border-conf-weak/30 bg-conf-weak-wash px-3 py-2.5 text-[13px] text-conf-weak">
            Explanation failed: {error}. The retrieval results above are unaffected.
          </div>
        )}

        {data?.degraded && (
          <div className="flex items-start gap-2 rounded border border-conf-mod/30 bg-conf-mod-wash px-3 py-2.5 text-[13px] text-conf-mod">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            The language model could not produce a validated explanation. Retrieval scores remain reliable.
          </div>
        )}

        {data && !loading &&
          data.explanations.map((exp) => (
            <article key={exp.case_id} className="rounded-md border border-line">
              <header className="flex items-center justify-between border-b border-line-soft px-4 py-2.5">
                <button
                  type="button"
                  onClick={() => onCiteClick(exp.case_id)}
                  className="font-mono text-[13px] font-semibold text-teal-deep underline-offset-2 hover:underline"
                >
                  {exp.case_id}
                </button>
                <span className={`rounded-sm px-2 py-0.5 font-mono text-[11px] font-medium ${CONF_STYLE[exp.confidence]}`}>
                  {exp.confidence} confidence
                </span>
              </header>

              <div className="space-y-3.5 px-4 py-3">
                {exp.similarity_factors.length > 0 && (
                  <div>
                    <h3 className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-conf-high">
                      <Check className="size-3.5" /> Shared findings
                    </h3>
                    <ul className="space-y-1.5">
                      {exp.similarity_factors.map((f, i) => (
                        <li key={i} className="text-[13px] leading-relaxed">
                          <span className="font-medium">{f.factor}</span>
                          <span className="text-ink-soft"> — {f.detail} </span>
                          {f.citations.map((c) => (
                            <Citation key={c} id={c} onClick={onCiteClick} />
                          ))}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {exp.differences.length > 0 && (
                  <div>
                    <h3 className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-faint">
                      <Minus className="size-3.5" /> Differences
                    </h3>
                    <ul className="space-y-1.5">
                      {exp.differences.map((d, i) => (
                        <li key={i} className="text-[13px] leading-relaxed text-ink-soft">
                          {d.detail}{" "}
                          {d.citations.map((c) => (
                            <Citation key={c} id={c} onClick={onCiteClick} />
                          ))}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {exp.treatments_observed.length > 0 && (
                  <div>
                    <h3 className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-teal-deep">
                      <Pill className="size-3.5" /> Treatments documented in this case
                    </h3>
                    <ul className="space-y-1.5">
                      {exp.treatments_observed.map((t, i) => (
                        <li key={i} className="text-[13px] leading-relaxed">
                          <span className="font-medium">{t.treatment}</span>
                          <span className="text-ink-soft"> → {t.outcome_note} </span>
                          {t.citations.map((c) => (
                            <Citation key={c} id={c} onClick={onCiteClick} />
                          ))}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </article>
          ))}

        {data && !loading && data.cohort_observation && (
          <div className="rounded-md border border-teal/25 bg-teal-wash px-4 py-3">
            <h3 className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-teal-deep">
              <ArrowRightLeft className="size-3.5" /> Across the retrieved cohort
            </h3>
            <p className="text-[13px] leading-relaxed text-ink">{data.cohort_observation}</p>
          </div>
        )}
      </div>

      {data && !loading && (
        <footer className="border-t border-line px-5 py-2.5">
          <p className="text-[11px] leading-relaxed text-ink-faint">{data.disclaimer}</p>
        </footer>
      )}
    </section>
  );
}
