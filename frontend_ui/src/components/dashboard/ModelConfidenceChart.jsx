import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { getChartTooltipProps } from './rechartsTooltipProps.js';

const BAR_COLORS = ['#38bdf8', '#34d399', '#a78bfa', '#f472b6'];

/**
 * Mean confidence (0–1) per stored model route — data from `/dashboard/summary` `model_confidence_routes`.
 */
export function ModelConfidenceChart({ rows, darkMode = true }) {
  const data = (rows || []).map((r, i) => ({
    name: r.label || r.key,
    avgPct: Math.round((r.avg_confidence ?? 0) * 1000) / 10,
    count: r.count ?? 0,
    key: r.key,
    fill: BAR_COLORS[i % BAR_COLORS.length],
  }));

  const empty = data.length === 0;

  return (
    <div className="h-[320px] w-full min-w-0" style={{ minWidth: 280 }}>
      {empty ? (
        <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-zinc-700 bg-zinc-950/40 px-4 text-center text-sm text-zinc-500">
          No route data returned from the API. Check that the backend is running and <code className="text-zinc-400">/dashboard/summary</code> returns JSON.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%" minWidth={280} minHeight={280}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 8, right: 24, left: 4, bottom: 8 }}
            barCategoryGap={16}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 100]}
              tick={{ fill: '#a1a1aa', fontSize: 11 }}
              tickFormatter={(v) => `${v}%`}
              label={{
                value: 'Mean confidence (%)',
                position: 'insideBottom',
                offset: -4,
                fill: '#71717a',
                fontSize: 11,
              }}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={128}
              tick={{ fill: '#d4d4d8', fontSize: 11 }}
              label={{
                value: 'Model route',
                angle: -90,
                position: 'insideLeft',
                fill: '#71717a',
                fontSize: 11,
              }}
            />
            <Tooltip
              {...getChartTooltipProps(darkMode)}
              formatter={(value, _name, props) => [
                `${value}% mean · ${props.payload.count} scores`,
                'Confidence',
              ]}
            />
            <Bar dataKey="avgPct" radius={[0, 6, 6, 0]} maxBarSize={28}>
              {data.map((e, i) => (
                <Cell key={e.key || i} fill={e.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
