/* CohortStrip — outcome composition of the retrieved cohort.
   A segmented bar that encodes real information: how the retrieved historical
   patients actually fared. Reads like a lab result, not a chart widget. */

import type { CaseResult } from "../api";

interface Props {
  results: CaseResult[];
}

const SEGMENTS = [
  { key: "improved", label: "improved", color: "var(--color-conf-high)" },
  {
    key: "deteriorated",
    label: "deteriorated",
    color: "var(--color-conf-mod)",
  },
  { key: "deceased", label: "deceased", color: "var(--color-conf-weak)" },
  { key: "unknown", label: "n/d", color: "var(--color-ink-faint)" },
] as const;

export default function CohortStrip({ results }: Props) {
  const total = results.length;
  if (total === 0) return null;

  const counts = SEGMENTS.map((s) => ({
    ...s,
    n: results.filter((r) => r.outcome_class === s.key).length,
  })).filter((s) => s.n > 0);

  return (
    <div className="rounded-md border border-line bg-chart px-4 py-3">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-faint">
          Documented outcomes in this cohort
        </span>
        <span className="font-mono text-[11px] tabular-nums text-ink-faint">
          n = {total}
        </span>
      </div>

      <div className="bar-grow flex h-2 overflow-hidden rounded-full">
        {counts.map((s) => (
          <div
            key={s.key}
            style={{ width: `${(s.n / total) * 100}%`, background: s.color }}
            title={`${s.label}: ${s.n}/${total}`}
          />
        ))}
      </div>

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {counts.map((s) => (
          <span
            key={s.key}
            className="flex items-center gap-1.5 font-mono text-[11.5px] text-ink-soft"
          >
            <span
              className="size-2 rounded-full"
              style={{ background: s.color }}
            />
            {s.label}
            <span className="tabular-nums text-ink-faint">{s.n}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
