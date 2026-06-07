import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { getChartTooltipProps } from './rechartsTooltipProps.js';

/**
 * Confidence bucket counts from API `confidence_histogram`.
 */
export function ConfidenceHistogram({ data, darkMode = true }) {
  const chartData = (data || []).map((d) => ({
    range: d.range,
    count: d.count,
    fill: d.color || '#52525b',
  }));

  const total = chartData.reduce((s, d) => s + (d.count || 0), 0);
  const empty = total === 0;

  return (
    <div className="h-[300px] w-full">
      {empty ? (
        <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-zinc-700 bg-zinc-950/40 text-sm text-zinc-500">
          No confidence scores stored yet.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 12, right: 12, left: 8, bottom: 28 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
            <XAxis
              dataKey="range"
              tick={{ fill: '#a1a1aa', fontSize: 11 }}
              label={{
                value: 'Confidence band (max class probability, %)',
                position: 'bottom',
                offset: 4,
                fill: '#71717a',
                fontSize: 11,
              }}
            />
            <YAxis
              allowDecimals={false}
              tick={{ fill: '#71717a', fontSize: 11 }}
              label={{
                value: 'Number of predictions',
                angle: -90,
                position: 'insideLeft',
                fill: '#71717a',
                fontSize: 11,
              }}
            />
            <Tooltip {...getChartTooltipProps(darkMode)} cursor={{ fill: 'rgba(63,63,70,0.35)' }} />
            <Bar dataKey="count" radius={[6, 6, 0, 0]} maxBarSize={48}>
              {chartData.map((e, i) => (
                <Cell key={i} fill={e.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
