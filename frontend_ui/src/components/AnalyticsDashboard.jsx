import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Database,
  CheckCircle2,
  XCircle,
  Gauge,
  Calendar,
  Award,
  RefreshCw,
  Layers,
} from 'lucide-react';
import { fetchDashboardSummary } from '../config/api.js';

/** Map internal route keys to neutral UI labels (no corpus codenames). */
function formatRouteCombo(combo) {
  if (!combo || typeof combo !== 'string') return combo || '';
  const map = {
    'fakeddit/distilbert': 'Social-style · encoder',
    'fakeddit/hybrid': 'Social-style · fusion',
    'liar/distilbert': 'News-style · encoder',
    'liar/hybrid': 'News-style · fusion',
  };
  return map[combo] || combo.replace(/\//g, ' · ');
}
import {
  StatCard,
  ModelConfidenceChart,
  PredictionDonut,
  ConfidenceHistogram,
  ActivityLineChart,
  DashboardLiveTable,
  InsightCard,
} from './dashboard/index.js';

function buildInsightCards(api) {
  const k = api?.kpis || {};
  const combos = api?.by_model_combo || [];
  const insights = api?.insights || [];
  const routes = api?.model_confidence_routes || [];
  const cards = [];

  if (routes.length) {
    const best = [...routes].filter((r) => r.count > 0).sort((a, b) => b.avg_confidence - a.avg_confidence)[0];
    if (best) {
      cards.push({
        title: 'Highest mean confidence',
        body: `${best.label} averages ${(best.avg_confidence * 100).toFixed(1)}% confidence across ${best.count} stored scores.`,
      });
    }
  }

  if (combos.length >= 1) {
    const sorted = [...combos].sort((a, b) => b.avg_confidence - a.avg_confidence);
    const top = sorted[0];
    cards.push({
      title: 'Volume-weighted mix',
      body: `${formatRouteCombo(top.combo)} has the largest score volume (${top.reliable + top.unreliable} rows) with ${(top.avg_confidence * 100).toFixed(1)}% mean confidence.`,
    });
  }

  const rel = k.reliable_predictions ?? 0;
  const unr = k.unreliable_predictions ?? 0;
  if (rel + unr > 0) {
    cards.push({
      title: 'Label balance',
      body: `Across all routes, ${rel} reliable and ${unr} unreliable predictions are stored. Compare with the donut slice (primary route) for thesis discussion.`,
    });
  }

  if (cards.length < 3 && insights[0]) {
    cards.push({ title: 'System', body: insights[0] });
  }

  const pad = [
    {
      title: 'Populate data',
      body: 'Use Live Feed (Reddit/Mastodon) or Analyze with persistence so all four model routes persist to the database.',
    },
    {
      title: 'Calibration',
      body: 'Lower confidence on short social text versus offline benchmarks is expected under domain shift; cite the histogram in your evaluation.',
    },
  ];
  let p = 0;
  while (cards.length < 3 && p < pad.length) {
    cards.push(pad[p++]);
  }

  return cards.slice(0, 3);
}

/**
 * Analytics dashboard: KPIs and charts from SQLite via `/dashboard/summary`.
 */
export default function AnalyticsDashboard({ darkMode }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const json = await fetchDashboardSummary();
      setData(json);
    } catch (e) {
      setErr(e.message || 'Failed to load dashboard');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const totals = data?.totals || { posts: 0, predictions: 0, posts_with_predictions: 0 };
  const kpis = data?.kpis || {};
  const pie = data?.reliability_pie || [];
  const hist = data?.confidence_histogram || [];
  const daily = data?.daily_activity || [];
  const recent = data?.recent_activity || [];
  const modelRoutes = data?.model_confidence_routes || [];

  const insightCards = useMemo(() => buildInsightCards(data), [data]);

  const postsScored = kpis.posts_with_predictions ?? totals.posts_with_predictions ?? 0;
  const predToday = kpis.predictions_today ?? 0;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 -m-6 p-6 md:p-8 lg:p-10">
      <div className="mx-auto max-w-[1400px] space-y-10">
        <header className="flex flex-col gap-4 border-b border-zinc-800/80 pb-8 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-500/90">Evaluation panel</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-zinc-50 md:text-4xl">VeriPulse analytics</h1>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-zinc-500">
              Live statistics from the VeriPulse SQLite store: posts, predictions per model route, and confidence distributions.
            </p>
            {data?.updated_at && (
              <p className="mt-2 text-xs text-zinc-600">Data snapshot {new Date(data.updated_at).toLocaleString()}</p>
            )}
          </div>
          <button
            type="button"
            onClick={load}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-2.5 text-sm font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </header>

        {err && (
          <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-sm text-amber-200">
            <strong className="font-semibold">Could not refresh data.</strong> {err} — ensure the API is running:{' '}
            <code className="rounded bg-zinc-800 px-1">python backend/app.py</code>
            {data && ' · Showing last successful load below.'}
          </div>
        )}

        <section>
          <h2 className="mb-4 text-xs font-bold uppercase tracking-wider text-zinc-500">Summary indicators</h2>
          <p className="mb-4 max-w-3xl text-sm text-zinc-500">
            Values are computed from the <code className="text-zinc-400">predictions</code> and <code className="text-zinc-400">posts</code> tables. If counts stay at zero, the
            database is empty or the request failed (see banner above).
          </p>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
            <StatCard
              title="Posts with scores"
              value={loading && !data ? '…' : Number(postsScored).toLocaleString()}
              subtitle="Distinct posts with ≥1 stored prediction"
              icon={Database}
              colorKey="sky"
              delay={0}
            />
            <StatCard
              title="Reliable (stored)"
              value={loading && !data ? '…' : (kpis.reliable_predictions ?? 0).toLocaleString()}
              subtitle="All model routes"
              icon={CheckCircle2}
              colorKey="emerald"
              delay={0.05}
            />
            <StatCard
              title="Unreliable (stored)"
              value={loading && !data ? '…' : (kpis.unreliable_predictions ?? 0).toLocaleString()}
              subtitle="All model routes"
              icon={XCircle}
              colorKey="rose"
              delay={0.1}
            />
            <StatCard
              title="Avg. confidence"
              value={loading && !data ? '…' : `${((kpis.avg_confidence_all ?? 0) * 100).toFixed(1)}%`}
              subtitle="Mean over all prediction rows"
              icon={Gauge}
              colorKey="amber"
              delay={0.15}
            />
            <StatCard
              title="New posts today"
              value={loading && !data ? '…' : (kpis.posts_today ?? 0).toLocaleString()}
              subtitle={`Predictions logged today: ${loading && !data ? '…' : predToday}`}
              icon={Calendar}
              colorKey="violet"
              delay={0.2}
            />
            <StatCard
              title="Most-used route"
              value={loading && !data ? '…' : kpis.best_model_display || '—'}
              subtitle="By prediction count in DB"
              icon={Award}
              colorKey="zinc"
              delay={0.25}
            />
          </div>
        </section>

        <section className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 shadow-xl backdrop-blur-sm md:p-8">
          <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-zinc-100">Mean confidence by model route</h2>
              <p className="mt-1 max-w-3xl text-sm text-zinc-500">
                Each bar is the average stored <strong className="font-medium text-zinc-400">confidence</strong> (max class probability) for that corpus + architecture pair:
                social-style vs news-style text, each with an encoder or a fusion head. Tooltip shows how many scores contributed.
              </p>
            </div>
            <span className="rounded-lg bg-cyan-950/50 px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-cyan-500/90">
              From database
            </span>
          </div>
          <ModelConfidenceChart rows={modelRoutes} darkMode={darkMode} />
        </section>

        <div className="grid grid-cols-1 gap-8 xl:grid-cols-2">
          <section className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 md:p-8">
            <h2 className="mb-1 text-lg font-semibold text-zinc-100">Stored label distribution</h2>
            <p className="mb-3 text-sm leading-relaxed text-zinc-500">
              <strong className="text-zinc-400">What this shows:</strong> how many stored predictions are classed as reliable vs unreliable for the{' '}
              <em>primary</em> scoring slice (social-style fusion when available; otherwise all stored rows).
            </p>
            <p className="mb-6 text-xs text-zinc-600">Reliable = predicted label 0 (trustworthy); unreliable = 1.</p>
            <PredictionDonut data={pie} darkMode={darkMode} />
          </section>
          <section className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 md:p-8">
            <h2 className="mb-1 text-lg font-semibold text-zinc-100">Confidence buckets</h2>
            <p className="mb-3 text-sm leading-relaxed text-zinc-500">
              <strong className="text-zinc-400">Horizontal axis:</strong> bands of model confidence (max class probability) as a percentage.{' '}
              <strong className="text-zinc-400">Vertical axis:</strong> how many prediction rows fall in each band. Sharp spikes near 50% often indicate ambiguous or short
              text; high counts near 90–100% suggest decisive scores.
            </p>
            <p className="mb-6 text-xs text-zinc-600">Includes every row in the predictions table.</p>
            <ConfidenceHistogram data={hist} darkMode={darkMode} />
          </section>
        </div>

        <section className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 md:p-8">
          <h2 className="mb-1 text-lg font-semibold text-zinc-100">Activity (last 14 days)</h2>
          <p className="mb-3 text-sm leading-relaxed text-zinc-500">
            <strong className="text-zinc-400">X-axis:</strong> calendar date (local) when prediction rows were written. <strong className="text-zinc-400">Y-axis:</strong>{' '}
            counts per day — <span className="text-cyan-400">cyan</span> line is total prediction rows; <span className="text-violet-400">violet</span> is distinct posts
            touched that day.
          </p>
          <p className="mb-6 text-xs text-zinc-600">Useful for correlating ingestion bursts with evaluation windows.</p>
          <ActivityLineChart data={daily} darkMode={darkMode} />
        </section>

        <section className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 md:p-8">
          <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 className="mb-1 text-lg font-semibold text-zinc-100">Recent activity</h2>
              <p className="max-w-3xl text-sm text-zinc-500">
                Latest rows from <code className="text-zinc-500">posts</code>, with the preferred full-fusion score when present (social-style route first). Requires a running
                API and a non-empty database.
              </p>
            </div>
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/20 bg-cyan-500/5 px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-cyan-500/90">
              <Layers className="h-3 w-3" />
              Live store
            </span>
          </div>
          <DashboardLiveTable rows={recent} />
        </section>

        <section>
          <h2 className="mb-4 text-xs font-bold uppercase tracking-wider text-zinc-500">Insights</h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {insightCards.map((c, i) => (
              <InsightCard key={i} title={c.title} body={c.body} index={i} />
            ))}
          </div>
        </section>

        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="border-t border-zinc-800 pt-8 text-center text-xs text-zinc-600"
        >
          VeriPulse · Misinformation detection evaluation dashboard
        </motion.footer>
      </div>
    </div>
  );
}
