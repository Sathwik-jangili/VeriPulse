import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { getChartTooltipProps } from './rechartsTooltipProps.js';

/**
 * Stored prediction activity over recent days (from `daily_activity`).
 */
export function ActivityLineChart({ data, darkMode = true }) {
  const chartData = (data || []).map((d) => ({
    day: d.day?.slice(5) || d.day,
    predictions: d.predictions ?? 0,
    posts: d.posts ?? 0,
  }));

  return (
    <div className="h-[320px] w-full">
      {chartData.length === 0 ? (
        <div className="flex h-full items-center justify-center text-sm text-zinc-500">No time-series data yet.</div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 28 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
            <XAxis
              dataKey="day"
              tick={{ fill: '#a1a1aa', fontSize: 11 }}
              label={{
                value: 'Calendar date (local)',
                position: 'insideBottom',
                offset: -2,
                fill: '#71717a',
                fontSize: 11,
              }}
            />
            <YAxis
              allowDecimals={false}
              tick={{ fill: '#71717a', fontSize: 11 }}
              label={{
                value: 'Count',
                angle: -90,
                position: 'insideLeft',
                fill: '#71717a',
                fontSize: 11,
              }}
            />
            <Tooltip {...getChartTooltipProps(darkMode)} />
            <Legend
              wrapperStyle={{
                fontSize: '12px',
                paddingTop: 8,
                color: darkMode !== false ? '#d4d4d8' : '#52525b',
              }}
            />
            <Line
              type="monotone"
              dataKey="predictions"
              name="Prediction rows written"
              stroke="#22d3ee"
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line
              type="monotone"
              dataKey="posts"
              name="Distinct posts"
              stroke="#a78bfa"
              strokeWidth={1.5}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
