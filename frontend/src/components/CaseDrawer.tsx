import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { getCase, type CaseDetail } from "../api";

interface Props {
  caseId: string | null;
  onClose: () => void;
}

export default function CaseDrawer({ caseId, onClose }: Props) {
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!caseId) return;
    setDetail(null);
    setError(null);
    getCase(caseId).then(setDetail).catch((e) => setError(String(e)));
  }, [caseId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!caseId) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-ink/25" onClick={onClose} role="dialog" aria-modal="true">
      <div
        className="h-full w-full max-w-2xl overflow-y-auto bg-chart shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="sticky top-0 flex items-center justify-between border-b border-line bg-chart px-6 py-4">
          <div>
            <h2 className="font-mono text-[15px] font-semibold text-teal-deep">{caseId}</h2>
            {detail && (
              <p className="mt-0.5 font-mono text-[12px] text-ink-soft">
                {detail.sex}, {detail.age ?? "—"} y · outcome: {detail.outcome_class}
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1.5 text-ink-faint hover:bg-line-soft hover:text-ink"
            aria-label="close case"
          >
            <X className="size-4" />
          </button>
        </header>

        <div className="px-6 py-5">
          {error && <p className="text-[13px] text-conf-weak">{error}</p>}
          {!detail && !error && <div className="pulse-soft h-40 rounded bg-line-soft" />}
          {detail && (
            <>
              <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-faint">
                Full case record
              </h3>
              <p className="whitespace-pre-wrap text-[13.5px] leading-relaxed text-ink">{detail.document}</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
