import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { getChartTooltipProps } from './rechartsTooltipProps.js';

/**
 * Reliable vs unreliable distribution (donut).
 */
export function PredictionDonut({ data, darkMode = true }) {
  const filtered = (data || []).filter((d) => d.value > 0);
  const empty = filtered.length === 0;

  return (
    <div className="h-[280px] w-full">
      {empty ? (
        <div className="flex h-full items-center justify-center text-sm text-zinc-500">No prediction data yet.</div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={filtered}
              cx="50%"
              cy="50%"
              innerRadius={68}
              outerRadius={96}
              paddingAngle={2}
              dataKey="value"
              nameKey="name"
              stroke="#18181b"
              strokeWidth={2}
            >
              {filtered.map((entry, i) => (
                <Cell key={i} fill={entry.color || (entry.name === 'Reliable' ? '#38bdf8' : '#f472b6')} />
              ))}
            </Pie>
            <Tooltip
              {...getChartTooltipProps(darkMode)}
              formatter={(v, n) => [v, n]}
              labelFormatter={(label) => String(label)}
            />
          </PieChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
