/* SimilarityTrace — the signature element.
   Retrieval score rendered as an ECG-like trace: amplitude and color encode
   match strength. Evokes the clinical world without being decorative — the
   waveform IS the score. */

interface Props {
  score: number; // 0..1
}

export default function SimilarityTrace({ score }: Props) {
  const s = Math.max(0, Math.min(1, score));
  // amplitude grows with score; weak matches are nearly flatline
  const amp = 3 + s * 15;
  const mid = 14;
  const color = s >= 0.5 ? "var(--color-conf-high)" : s >= 0.34 ? "var(--color-conf-mod)" : "var(--color-conf-weak)";

  // one QRS-like complex positioned proportionally to score along the strip
  const beat = 18 + s * 60;
  const d = [
    `M 0 ${mid}`,
    `L ${beat - 12} ${mid}`,
    `L ${beat - 8} ${mid - amp * 0.25}`,
    `L ${beat - 4} ${mid + amp * 0.35}`,
    `L ${beat} ${mid - amp}`,
    `L ${beat + 4} ${mid + amp * 0.55}`,
    `L ${beat + 8} ${mid}`,
    `L ${beat + 22} ${mid}`,
    `L ${beat + 26} ${mid - amp * 0.3}`,
    `L ${beat + 30} ${mid}`,
    `L 120 ${mid}`,
  ].join(" ");

  return (
    <div className="flex items-center gap-2" title={`similarity ${s.toFixed(3)}`}>
      <svg width="120" height="28" viewBox="0 0 120 28" aria-hidden="true" className="shrink-0">
        <line x1="0" y1={mid} x2="120" y2={mid} stroke="var(--color-line)" strokeWidth="1" />
        <path className="trace-path" d={d} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" />
      </svg>
      <span className="font-mono text-[13px] font-medium tabular-nums" style={{ color }}>
        {s.toFixed(3)}
      </span>
    </div>
  );
}
