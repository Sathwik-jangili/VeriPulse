import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import { TrendingUp, Activity, Target, Cpu, RefreshCw, AlertCircle } from 'lucide-react';
import { getApiBase } from '../config/api.js';

const Dashboard = ({ darkMode }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const res = await fetch(`${getApiBase()}/dashboard/summary`);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
      setData(json);
    } catch (e) {
      setErr(e.message || 'Failed to load dashboard');
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const gridStroke = darkMode ? '#374151' : '#e5e7eb';
  const axisFill = darkMode ? '#9ca3af' : '#6b7280';
  const tooltipStyle = {
    backgroundColor: darkMode ? '#1f2937' : '#ffffff',
    border: 'none',
    borderRadius: '8px',
    color: darkMode ? '#f3f4f6' : '#111827',
  };

  const StatCard = ({ icon: Icon, title, value, subtitle, color }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`rounded-2xl p-6 border ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}`}
    >
      <div className="flex items-center justify-between mb-4">
        <div
          className={`w-12 h-12 rounded-xl flex items-center justify-center ${
            color === 'green'
              ? 'bg-green-100 text-green-600'
              : color === 'red'
                ? 'bg-red-100 text-red-600'
                : color === 'blue'
                  ? 'bg-blue-100 text-blue-600'
                  : 'bg-purple-100 text-purple-600'
          }`}
        >
          <Icon className="w-6 h-6" />
        </div>
      </div>
      <div>
        <p className={`text-xs uppercase tracking-wide ${darkMode ? 'text-neutral-500' : 'text-neutral-500'}`}>{title}</p>
        <p className={`text-2xl font-bold mt-1 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>{value}</p>
        <p className={`text-sm mt-1 ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>{subtitle}</p>
      </div>
    </motion.div>
  );

  const totals = data?.totals || { posts: 0, predictions: 0 };
  const headline = data?.headline || {};
  const pieData = (data?.reliability_pie || []).filter((d) => d.value > 0);
  const hist = data?.confidence_histogram || [];
  const daily = data?.daily_activity || [];
  const byCombo = data?.by_model_combo || [];
  const byPlatform = data?.by_platform || [];
  const insights = data?.insights || [];

  const comboChart = byCombo.map((c) => ({
    name: c.combo.replace('/', '\n'),
    reliable: c.reliable,
    unreliable: c.unreliable,
  }));

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center mb-8 flex flex-col sm:flex-row items-center justify-center gap-4"
      >
        <div>
          <h1 className={`text-3xl font-bold mb-2 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>Analytics Dashboard</h1>
          <p className={`text-lg ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
            Live data from saved Reddit, Mastodon, and Analyze runs
          </p>
          {data?.updated_at && (
            <p className={`text-xs mt-1 ${darkMode ? 'text-neutral-500' : 'text-neutral-500'}`}>
              Updated {new Date(data.updated_at).toLocaleString()}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={load}
          disabled={loading}
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium border ${
            darkMode
              ? 'border-neutral-600 text-white hover:bg-neutral-700'
              : 'border-neutral-300 text-neutral-800 hover:bg-neutral-50'
          }`}
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </motion.div>

      {err && (
        <div
          className={`flex items-center gap-2 p-4 rounded-xl border ${
            darkMode ? 'bg-red-900/20 border-red-800 text-red-200' : 'bg-red-50 border-red-200 text-red-800'
          }`}
        >
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{err} — start the Flask API (python backend/app.py) and ensure Vite proxies /api.</span>
        </div>
      )}

      {loading && !data && (
        <p className={`text-center ${darkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>Loading analytics…</p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          icon={Activity}
          title="Posts stored"
          value={totals.posts.toLocaleString()}
          subtitle="Unique texts captured"
          color="blue"
        />
        <StatCard
          icon={Cpu}
          title="Model scores"
          value={totals.predictions.toLocaleString()}
          subtitle="All model runs (4 per post when full)"
          color="purple"
        />
        <StatCard
          icon={Target}
          title="Unreliable rate"
          value={`${((headline.unreliable_rate_fakeddit_hybrid || 0) * 100).toFixed(1)}%`}
          subtitle="Primary fusion model"
          color="red"
        />
        <StatCard
          icon={TrendingUp}
          title="Avg confidence"
          value={`${((headline.avg_confidence_fakeddit_hybrid || 0) * 100).toFixed(1)}%`}
          subtitle="Primary fusion model"
          color="green"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className={`rounded-2xl p-6 border ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}`}
        >
          <h3 className={`text-lg font-semibold mb-6 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            Confidence distribution (all models)
          </h3>
          {totals.predictions === 0 ? (
            <p className={`text-sm ${darkMode ? 'text-neutral-500' : 'text-neutral-500'}`}>No predictions yet.</p>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={hist}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                <XAxis dataKey="range" stroke={axisFill} tick={{ fill: axisFill }} />
                <YAxis stroke={axisFill} tick={{ fill: axisFill }} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                  {hist.map((entry, i) => (
                    <Cell key={i} fill={entry.color || '#3b82f6'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.15 }}
          className={`rounded-2xl p-6 border ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}`}
        >
          <h3 className={`text-lg font-semibold mb-6 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            Reliable vs unreliable (primary view)
          </h3>
          {pieData.length === 0 ? (
            <p className={`text-sm ${darkMode ? 'text-neutral-500' : 'text-neutral-500'}`}>No data yet.</p>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={56}
                  outerRadius={96}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={index} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          )}
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className={`rounded-2xl p-6 border ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}`}
      >
        <h3 className={`text-lg font-semibold mb-6 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
          By model combination
        </h3>
        {comboChart.length === 0 ? (
          <p className={`text-sm ${darkMode ? 'text-neutral-500' : 'text-neutral-500'}`}>No breakdown yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={comboChart} layout="vertical" margin={{ left: 8, right: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis type="number" stroke={axisFill} tick={{ fill: axisFill }} allowDecimals={false} />
              <YAxis type="category" dataKey="name" width={100} stroke={axisFill} tick={{ fill: axisFill, fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              <Bar dataKey="reliable" stackId="a" fill="#10b981" name="Reliable" />
              <Bar dataKey="unreliable" stackId="a" fill="#ef4444" name="Unreliable" />
            </BarChart>
          </ResponsiveContainer>
        )}
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.25 }}
          className={`rounded-2xl p-6 border ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}`}
        >
          <h3 className={`text-lg font-semibold mb-6 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>
            Activity (last 14 days)
          </h3>
          {daily.length === 0 ? (
            <p className={`text-sm ${darkMode ? 'text-neutral-500' : 'text-neutral-500'}`}>
              No dated activity yet — fetch Live Feed or run Analyze with persist.
            </p>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={daily}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                <XAxis dataKey="day" stroke={axisFill} tick={{ fill: axisFill }} />
                <YAxis stroke={axisFill} tick={{ fill: axisFill }} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend />
                <Bar dataKey="posts" fill="#6366f1" name="Unique posts" radius={[4, 4, 0, 0]} />
                <Bar dataKey="predictions" fill="#8b5cf6" name="Predictions" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className={`rounded-2xl p-6 border ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}`}
        >
          <h3 className={`text-lg font-semibold mb-6 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>By platform</h3>
          {byPlatform.length === 0 ? (
            <p className={`text-sm ${darkMode ? 'text-neutral-500' : 'text-neutral-500'}`}>No platforms yet.</p>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={byPlatform} cx="50%" cy="50%" outerRadius={90} dataKey="count" nameKey="platform" label>
                  {byPlatform.map((entry, i) => (
                    <Cell key={i} fill={['#f97316', '#a855f7', '#3b82f6', '#14b8a6'][i % 4]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          )}
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.35 }}
        className={`rounded-2xl p-6 border ${darkMode ? 'bg-neutral-800 border-neutral-700' : 'bg-white border-neutral-200'}`}
      >
        <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-neutral-900'}`}>Insights</h3>
        <div className="space-y-3">
          {insights.map((line, i) => (
            <div key={i} className={`p-4 rounded-lg ${darkMode ? 'bg-neutral-700' : 'bg-neutral-100'}`}>
              <p className={`text-sm ${darkMode ? 'text-neutral-300' : 'text-neutral-700'}`}>{line}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
