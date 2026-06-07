import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const COLORS = {
  accuracy: '#34d399',
  precision: '#38bdf8',
  recall: '#fbbf24',
  f1: '#f472b6',
};

/**
 * Grouped bars: Accuracy, Precision, Recall, F1 per model (dissertation-style).
 */
export function ModelComparisonChart({ rows, darkMode = true }) {
  const data = rows.map((r) => ({
    name: r.label.length > 14 ? r.label.slice(0, 12) + '…' : r.label,
    fullName: `${r.label} (${r.corpus})`,
    Accuracy: Number(r.accuracy.toFixed(4)),
    Precision: Number(r.precision.toFixed(4)),
    Recall: Number(r.recall.toFixed(4)),
    F1: Number(r.f1.toFixed(4)),
  }));

  const tooltipStyle = {
    backgroundColor: darkMode ? '#18181b' : '#fafafa',
    border: '1px solid #3f3f46',
    borderRadius: '10px',
    fontSize: '12px',
  };

  return (
    <div className="h-[380px] w-full min-w-0">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: 4, bottom: 64 }} barGap={2} barCategoryGap={18}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fill: '#a1a1aa', fontSize: 10 }}
            interval={0}
            angle={-35}
            textAnchor="end"
            height={70}
          />
          <YAxis domain={[0, 1]} tick={{ fill: '#71717a', fontSize: 11 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value, name) => [`${(value * 100).toFixed(2)}%`, name]}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.fullName || ''}
          />
          <Legend wrapperStyle={{ fontSize: '11px', paddingTop: 8 }} />
          <Bar dataKey="Accuracy" fill={COLORS.accuracy} radius={[4, 4, 0, 0]} maxBarSize={18} />
          <Bar dataKey="Precision" fill={COLORS.precision} radius={[4, 4, 0, 0]} maxBarSize={18} />
          <Bar dataKey="Recall" fill={COLORS.recall} radius={[4, 4, 0, 0]} maxBarSize={18} />
          <Bar dataKey="F1" fill={COLORS.f1} radius={[4, 4, 0, 0]} maxBarSize={18} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
