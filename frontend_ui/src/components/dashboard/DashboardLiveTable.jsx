import React from 'react';
import { Radio, Globe } from 'lucide-react';

function fmtTime(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch {
    return iso;
  }
}

/**
 * Recent posts from API `recent_activity` (monitoring-style table).
 */
export function DashboardLiveTable({ rows }) {
  const list = rows || [];

  return (
    <div className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/40">
      <div className="grid grid-cols-12 gap-2 border-b border-zinc-800 px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
        <div className="col-span-2">Platform</div>
        <div className="col-span-4">Preview</div>
        <div className="col-span-2">Label</div>
        <div className="col-span-2">Confidence</div>
        <div className="col-span-2 text-right">Time</div>
      </div>
      <div className="max-h-[340px] divide-y divide-zinc-800/80 overflow-y-auto">
        {list.length === 0 && (
          <div className="px-4 py-10 text-center text-sm text-zinc-500">No stored posts yet. Run Analyze or Live Feed with persistence.</div>
        )}
        {list.map((r, i) => {
          const pred = r.prediction;
          const isRel = pred !== null && pred !== undefined && Number(pred) === 0;
          const label = pred == null ? '—' : isRel ? 'Reliable' : 'Unreliable';
          const plat = (r.platform || '').trim();
          const platLower = plat.toLowerCase();
          const Icon = platLower === 'mastodon' ? Globe : Radio;
          const platDisplay = plat
            ? plat.charAt(0).toUpperCase() + plat.slice(1).toLowerCase()
            : '—';
          return (
            <div
              key={r.post_id != null ? `p-${r.post_id}` : `row-${i}`}
              className="grid grid-cols-12 items-start gap-2 px-4 py-3 text-sm hover:bg-zinc-800/40"
            >
              <div className="col-span-2 flex items-center gap-2 text-zinc-300">
                <Icon className="h-4 w-4 shrink-0 text-zinc-500" />
                <span className="truncate text-xs font-medium">{platDisplay}</span>
              </div>
              <div className="col-span-4 text-xs leading-snug text-zinc-400">{r.text_preview || '—'}</div>
              <div className="col-span-2">
                <span
                  className={`inline-flex rounded-md px-2 py-0.5 text-xs font-semibold ${
                    pred == null
                      ? 'bg-zinc-800 text-zinc-500'
                      : isRel
                        ? 'bg-emerald-500/15 text-emerald-400'
                        : 'bg-rose-500/15 text-rose-400'
                  }`}
                >
                  {label}
                </span>
              </div>
              <div className="col-span-2 font-mono text-xs text-zinc-400">
                {r.confidence != null ? `${(r.confidence * 100).toFixed(1)}%` : '—'}
              </div>
              <div className="col-span-2 text-right text-xs text-zinc-500">{fmtTime(r.created_at)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
