import React from 'react';

export function ConfusionMatrixTable({ config }) {
  if (!config?.matrix?.length) return null;
  const { title, labels, matrix, note } = config;
  const [lr, ur] = labels;

  return (
    <div className="overflow-x-auto">
      <p className="mb-3 text-center text-xs font-semibold text-zinc-400">{title}</p>
      <table className="w-full min-w-[280px] border-collapse text-center text-sm">
        <thead>
          <tr>
            <th className="border border-zinc-700 bg-zinc-800/80 p-2 text-xs font-medium text-zinc-500" />
            <th className="border border-zinc-700 bg-zinc-800/80 p-2 text-xs font-medium text-emerald-400/90">{lr}</th>
            <th className="border border-zinc-700 bg-zinc-800/80 p-2 text-xs font-medium text-rose-400/90">{ur}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th className="border border-zinc-700 bg-zinc-800/50 p-2 text-left text-xs font-medium text-emerald-400/90">True: Reliable</th>
            <td className="border border-zinc-700 p-2 font-mono text-zinc-200">{matrix[0][0]}</td>
            <td className="border border-zinc-700 p-2 font-mono text-zinc-200">{matrix[0][1]}</td>
          </tr>
          <tr>
            <th className="border border-zinc-700 bg-zinc-800/50 p-2 text-left text-xs font-medium text-rose-400/90">True: Unreliable</th>
            <td className="border border-zinc-700 p-2 font-mono text-zinc-200">{matrix[1][0]}</td>
            <td className="border border-zinc-700 p-2 font-mono text-zinc-200">{matrix[1][1]}</td>
          </tr>
        </tbody>
      </table>
      {note && <p className="mt-2 text-center text-[11px] leading-snug text-zinc-600">{note}</p>}
    </div>
  );
}
