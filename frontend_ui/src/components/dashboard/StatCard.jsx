import React from 'react';
import { motion } from 'framer-motion';

const accent = {
  emerald: 'from-emerald-500/20 to-emerald-600/5 border-emerald-500/30 text-emerald-400',
  sky: 'from-sky-500/20 to-sky-600/5 border-sky-500/30 text-sky-400',
  amber: 'from-amber-500/20 to-amber-600/5 border-amber-500/30 text-amber-400',
  rose: 'from-rose-500/20 to-rose-600/5 border-rose-500/30 text-rose-400',
  violet: 'from-violet-500/20 to-violet-600/5 border-violet-500/30 text-violet-400',
  zinc: 'from-zinc-500/20 to-zinc-600/5 border-zinc-500/30 text-zinc-300',
};

/**
 * KPI card for analytics header row.
 */
export function StatCard({ title, value, subtitle, icon: Icon, colorKey = 'zinc', delay = 0 }) {
  const ring = accent[colorKey] || accent.zinc;
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay }}
      className={`relative overflow-hidden rounded-2xl border bg-gradient-to-br p-5 shadow-lg ${ring}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">{title}</p>
          <p className="mt-2 text-2xl font-bold tabular-nums tracking-tight text-zinc-50">{value}</p>
          {subtitle && <p className="mt-1 text-xs leading-snug text-zinc-500">{subtitle}</p>}
        </div>
        {Icon && (
          <div className="rounded-xl bg-zinc-800/80 p-2.5 text-zinc-400">
            <Icon className="h-5 w-5" strokeWidth={1.75} />
          </div>
        )}
      </div>
    </motion.div>
  );
}
