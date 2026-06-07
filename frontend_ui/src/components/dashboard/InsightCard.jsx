import React from 'react';
import { Lightbulb } from 'lucide-react';

const ACCENTS = [
  'border-emerald-500/30 bg-emerald-500/5',
  'border-sky-500/30 bg-sky-500/5',
  'border-violet-500/30 bg-violet-500/5',
];

/**
 * Single insight strip for the insights row.
 */
export function InsightCard({ title, body, index = 0 }) {
  const ring = ACCENTS[index % ACCENTS.length];
  return (
    <div className={`rounded-xl border p-5 ${ring}`}>
      <div className="mb-2 flex items-center gap-2 text-emerald-400/90">
        <Lightbulb className="h-4 w-4 shrink-0" strokeWidth={1.75} />
        <span className="text-[11px] font-bold uppercase tracking-wider text-zinc-500">Insight</span>
      </div>
      <h4 className="text-sm font-semibold text-zinc-100">{title}</h4>
      <p className="mt-2 text-sm leading-relaxed text-zinc-400">{body}</p>
    </div>
  );
}
