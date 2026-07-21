import { FileText, Sparkles } from "lucide-react";
import type { CaseResult } from "../api";
import SimilarityTrace from "./SimilarityTrace";

interface Props {
  result: CaseResult;
  rank: number;
  selected: boolean;
  onToggleSelect: () => void;
  onOpen: () => void;
  onExplain: () => void;
  index: number;
}

const OUTCOME_STYLE: Record<string, { label: string; cls: string }> = {
  improved: { label: "improved", cls: "bg-conf-high-wash text-conf-high" },
  deteriorated: {
    label: "deteriorated",
    cls: "bg-conf-mod-wash text-conf-mod",
  },
  deceased: { label: "deceased", cls: "bg-conf-weak-wash text-conf-weak" },
  unknown: { label: "outcome n/d", cls: "bg-line-soft text-ink-faint" },
};

export default function CaseCard({
  result,
  rank,
  selected,
  onToggleSelect,
  onOpen,
  onExplain,
  index,
}: Props) {
  const outcome = OUTCOME_STYLE[result.outcome_class] ?? OUTCOME_STYLE.unknown;

  return (
    <article
      className={`card-in group relative overflow-hidden rounded-md border bg-chart transition-shadow hover:shadow-[0_2px_10px_rgba(26,46,53,0.07)] ${
        selected ? "border-teal" : "border-line"
      }`}
      style={{ animationDelay: `${index * 45}ms` }}
    >
      {/* left accent — appears on hover / selection */}
      <span
        aria-hidden="true"
        className={`absolute inset-y-0 left-0 w-[3px] bg-teal transition-opacity ${
          selected ? "opacity-100" : "opacity-0 group-hover:opacity-60"
        }`}
      />
      <div className="flex items-start gap-4 px-4 pt-3.5">
        {/* rank + selection */}
        <label className="flex cursor-pointer flex-col items-center gap-1.5 pt-0.5">
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggleSelect}
            className="size-4 accent-(--color-teal)"
            aria-label={`select case ${result.case_id} for explanation`}
          />
          <span className="font-mono text-[11px] text-ink-faint">#{rank}</span>
        </label>

        <div className="min-w-0 flex-1">
          {/* header row */}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <button
              type="button"
              onClick={onOpen}
              className="font-mono text-[13.5px] font-semibold text-teal-deep underline-offset-2 hover:underline"
            >
              {result.case_id}
            </button>
            <span className="font-mono text-[12.5px] text-ink-soft">
              {result.sex}, {result.age ?? "—"} y
            </span>
            <span
              className={`rounded-sm px-1.5 py-0.5 font-mono text-[11px] font-medium ${outcome.cls}`}
            >
              {outcome.label}
            </span>
            {result.quality_flags.map((f) => (
              <span
                key={f}
                className="rounded-sm bg-line-soft px-1.5 py-0.5 font-mono text-[11px] text-ink-faint"
              >
                {f}
              </span>
            ))}
            <div className="ml-auto">
              <SimilarityTrace score={result.score} />
            </div>
          </div>

          {/* snippet */}
          <p className="mt-2 line-clamp-2 text-[13.5px] leading-relaxed text-ink-soft">
            {result.snippet}
          </p>
        </div>
      </div>

      {/* actions */}
      <div className="mt-3 flex items-center gap-1 border-t border-line-soft px-4 py-2">
        <button
          type="button"
          onClick={onOpen}
          className="flex items-center gap-1.5 rounded px-2 py-1 text-[12.5px] font-medium text-ink-soft hover:bg-line-soft"
        >
          <FileText className="size-3.5" />
          Read full case
        </button>
        <button
          type="button"
          onClick={onExplain}
          className="flex items-center gap-1.5 rounded px-2 py-1 text-[12.5px] font-medium text-teal-deep hover:bg-teal-wash"
        >
          <Sparkles className="size-3.5" />
          Explain this match
        </button>
      </div>
    </article>
  );
}
