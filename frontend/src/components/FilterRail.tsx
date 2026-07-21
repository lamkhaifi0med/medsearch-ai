import type { SearchFilters } from "../api";

interface Props {
  filters: SearchFilters;
  onChange: (f: SearchFilters) => void;
  k: number;
  onKChange: (k: number) => void;
  disabled?: boolean;
}

const OUTCOMES = [
  { value: "improved", label: "Improved" },
  { value: "deteriorated", label: "Deteriorated" },
  { value: "deceased", label: "Deceased" },
  { value: "unknown", label: "Not documented" },
] as const;

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-faint">
        {label}
      </div>
      {children}
    </div>
  );
}

export default function FilterRail({
  filters,
  onChange,
  k,
  onKChange,
  disabled,
}: Props) {
  const set = (patch: Partial<SearchFilters>) =>
    onChange({ ...filters, ...patch });

  const chip = (active: boolean) =>
    `rounded border px-2.5 py-1 text-[13px] transition-colors ${
      active
        ? "border-teal bg-teal-wash font-medium text-teal-deep"
        : "border-line bg-chart text-ink-soft hover:border-ink-faint"
    } ${disabled ? "pointer-events-none opacity-50" : ""}`;

  return (
    <aside className="grid w-full shrink-0 grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4 lg:block lg:w-56 lg:space-y-6">
      <Field label="Sex">
        <div className="flex gap-1.5">
          {(["male", "female"] as const).map((s) => (
            <button
              key={s}
              type="button"
              className={chip(filters.sex === s)}
              onClick={() => set({ sex: filters.sex === s ? null : s })}
            >
              {s === "male" ? "Male" : "Female"}
            </button>
          ))}
        </div>
      </Field>

      <Field label="Age range">
        <div className="flex items-center gap-2">
          <input
            type="number"
            min={0}
            max={120}
            placeholder="min"
            value={filters.age_min ?? ""}
            disabled={disabled}
            onChange={(e) =>
              set({
                age_min: e.target.value === "" ? null : Number(e.target.value),
              })
            }
            className="w-16 rounded border border-line bg-chart px-2 py-1 font-mono text-[13px] tabular-nums placeholder:text-ink-faint"
          />
          <span className="text-ink-faint">–</span>
          <input
            type="number"
            min={0}
            max={120}
            placeholder="max"
            value={filters.age_max ?? ""}
            disabled={disabled}
            onChange={(e) =>
              set({
                age_max: e.target.value === "" ? null : Number(e.target.value),
              })
            }
            className="w-16 rounded border border-line bg-chart px-2 py-1 font-mono text-[13px] tabular-nums placeholder:text-ink-faint"
          />
        </div>
      </Field>

      <Field label="Documented outcome">
        <div className="flex flex-wrap gap-1.5">
          {OUTCOMES.map((o) => (
            <button
              key={o.value}
              type="button"
              className={chip(filters.outcome_class === o.value)}
              onClick={() =>
                set({
                  outcome_class:
                    filters.outcome_class === o.value ? null : o.value,
                })
              }
            >
              {o.label}
            </button>
          ))}
        </div>
      </Field>

      <Field label="Cases to retrieve">
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={3}
            max={25}
            value={k}
            disabled={disabled}
            onChange={(e) => onKChange(Number(e.target.value))}
            className="w-full accent-(--color-teal)"
          />
          <span className="w-7 text-right font-mono text-[13px] font-medium tabular-nums">
            {k}
          </span>
        </div>
      </Field>
    </aside>
  );
}
