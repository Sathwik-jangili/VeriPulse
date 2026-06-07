import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';

/**
 * User study / questionnaire summary. Uses `questionnaire` prop or falls back to sample from benchmarks.
 */
export function UserEvaluationPanel({ questionnaire, sample }) {
  const q = questionnaire || sample;
  if (!q || q.n === 0) {
    return (
      <div className="rounded-xl border border-dashed border-zinc-700 bg-zinc-900/30 px-6 py-12 text-center">
        <p className="text-sm font-medium text-zinc-400">No questionnaire data</p>
        <p className="mt-2 text-xs text-zinc-600">Add a user study or import scores to show clarity, usefulness, and trust.</p>
      </div>
    );
  }

  const radarData = [
    { metric: 'Clarity', value: q.clarity, full: 5 },
    { metric: 'Usefulness', value: q.usefulness, full: 5 },
    { metric: 'Trust', value: q.trust, full: 5 },
  ];

  const tooltipStyle = {
    backgroundColor: '#18181b',
    border: '1px solid #3f3f46',
    borderRadius: '8px',
    fontSize: '12px',
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2 lg:items-center">
      <div className="h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
            <PolarGrid stroke="#3f3f46" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: '#a1a1aa', fontSize: 11 }} />
            <PolarRadiusAxis angle={30} domain={[0, 5]} tick={{ fill: '#71717a', fontSize: 10 }} />
            <Radar name="Score" dataKey="value" stroke="#34d399" fill="#34d399" fillOpacity={0.35} strokeWidth={2} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => [v.toFixed(2), 'Avg.']} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <div className="space-y-4">
        <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Participants (n)</p>
        <p className="text-3xl font-bold text-zinc-100">{q.n}</p>
        <div className="space-y-3 border-t border-zinc-800 pt-4">
          {[
            ['Clarity of explanations', q.clarity],
            ['Usefulness for triage', q.usefulness],
            ['Trust in the score', q.trust],
          ].map(([k, v]) => (
            <div key={k} className="flex items-center justify-between text-sm">
              <span className="text-zinc-400">{k}</span>
              <span className="font-mono font-semibold text-emerald-400">{v.toFixed(2)} / 5</span>
            </div>
          ))}
        </div>
        {q.comments && <p className="text-xs leading-relaxed text-zinc-500">{q.comments}</p>}
      </div>
    </div>
  );
}
