/**
 * Flask backend base URL.
 *
 * - `npm run dev`: leave VITE_API_BASE unset → uses `/api` and Vite `server.proxy` → Flask :5000.
 * - `npm run preview`: leave unset → uses `/api` on port 4173/4174 with `preview.proxy` (see vite.config.js).
 * - Deployed: set VITE_API_BASE to your API origin only, e.g. https://api.example.com
 *   (no trailing path; do NOT use .../api — Flask serves /predict, /health, not /api/predict).
 */
export function getApiBase() {
  let env = import.meta.env.VITE_API_BASE?.trim();
  if (env) {
    env = env.replace(/\/$/, '');
    if (env.endsWith('/api')) {
      env = env.slice(0, -4);
    }
    return env;
  }
  if (import.meta.env.DEV) return '/api';
  if (typeof window !== 'undefined') {
    const p = window.location.port;
    if (p === '4173' || p === '4174') return '/api';
  }
  return 'http://127.0.0.1:5000';
}

/** Join base URL (absolute or `/api`) with an API path like `/dashboard/summary`. */
export function joinApiUrl(base, path) {
  const p = path.startsWith('/') ? path : `/${path}`;
  const b = base.replace(/\/$/, '');
  if (b.startsWith('http')) {
    return `${b}${p}`;
  }
  return `${b}${p}`;
}

const DIRECT_FLASK_URLS = ['http://127.0.0.1:5000', 'http://localhost:5000'];

/**
 * Fill optional dashboard fields if the client talks to an older API, and always
 * expose four model routes when `by_model_combo` is present.
 */
export function normalizeDashboardPayload(json) {
  if (!json || typeof json !== 'object') return json;

  const combos = json.by_model_combo || [];
  const comboMap = new Map();
  for (const c of combos) {
    if (c && c.combo) comboMap.set(c.combo, c);
  }

  const ROUTES = [
    ['fakeddit', 'distilbert', 'Social-style · encoder'],
    ['fakeddit', 'hybrid', 'Social-style · fusion'],
    ['liar', 'distilbert', 'News-style · encoder'],
    ['liar', 'hybrid', 'News-style · fusion'],
  ];

  if (!Array.isArray(json.model_confidence_routes) || json.model_confidence_routes.length === 0) {
    json.model_confidence_routes = ROUTES.map(([ds, ar, label]) => {
      const row = comboMap.get(`${ds}/${ar}`);
      const n = row ? (Number(row.reliable) || 0) + (Number(row.unreliable) || 0) : 0;
      const avg = row && row.avg_confidence != null ? Number(row.avg_confidence) : 0;
      return {
        key: `${ds}:${ar}`,
        dataset: ds,
        arch: ar,
        label,
        avg_confidence: avg,
        count: n,
      };
    });
  }

  json.totals = json.totals || {};
  json.kpis = json.kpis || {};

  if (json.kpis.posts_with_predictions == null && json.totals.posts_with_predictions != null) {
    json.kpis.posts_with_predictions = json.totals.posts_with_predictions;
  }

  const totalPredRows = Number(json.totals?.predictions) || 0;
  let pwp = Number(json.kpis.posts_with_predictions);
  if (!Number.isFinite(pwp)) pwp = Number(json.totals?.posts_with_predictions) || 0;
  if ((pwp === 0 || !Number.isFinite(pwp)) && totalPredRows > 0 && combos.length > 0) {
    const maxRoute = Math.max(
      ...combos.map((c) => (Number(c.reliable) || 0) + (Number(c.unreliable) || 0))
    );
    if (maxRoute > 0) {
      json.kpis.posts_with_predictions = maxRoute;
      json.totals.posts_with_predictions = maxRoute;
    }
  }

  if (!json.kpis.best_model_display && json.kpis.best_model_combo) {
    json.kpis.best_model_display = json.kpis.best_model_combo.replace('/', ' · ');
  }

  if (
    (json.kpis.reliable_predictions == null || json.kpis.unreliable_predictions == null) &&
    combos.length > 0
  ) {
    let rel = 0;
    let unr = 0;
    for (const c of combos) {
      rel += Number(c.reliable) || 0;
      unr += Number(c.unreliable) || 0;
    }
    if (json.kpis.reliable_predictions == null) json.kpis.reliable_predictions = rel;
    if (json.kpis.unreliable_predictions == null) json.kpis.unreliable_predictions = unr;
  }

  if (json.kpis.avg_confidence_all == null && json.totals.predictions > 0 && combos.length > 0) {
    let sum = 0;
    let n = 0;
    for (const c of combos) {
      const cnt = (Number(c.reliable) || 0) + (Number(c.unreliable) || 0);
      if (cnt > 0 && c.avg_confidence != null) {
        sum += Number(c.avg_confidence) * cnt;
        n += cnt;
      }
    }
    if (n > 0) json.kpis.avg_confidence_all = sum / n;
  }

  json.recent_activity = Array.isArray(json.recent_activity) ? json.recent_activity : [];

  return json;
}

/**
 * Load `/dashboard/summary` with fallbacks: Vite `/api` proxy first, then direct Flask
 * URLs (fixes Windows setups where the dev proxy fails or Flask is only reachable directly).
 */
export async function fetchDashboardSummary() {
  const path = '/dashboard/summary';
  const primary = joinApiUrl(getApiBase(), path);

  const urls = [primary];
  const base = getApiBase();
  if (base === '/api' || base.endsWith('/api')) {
    for (const h of DIRECT_FLASK_URLS) {
      const u = joinApiUrl(h, path);
      if (!urls.includes(u)) urls.push(u);
    }
  } else if (base.startsWith('http') && !urls.includes(joinApiUrl('http://127.0.0.1:5000', path))) {
    urls.push(joinApiUrl('http://127.0.0.1:5000', path));
  }

  let lastErr = new Error('Dashboard unreachable');
  for (const url of urls) {
    try {
      const res = await fetch(url, {
        headers: { Accept: 'application/json' },
        mode: 'cors',
      });
      const text = await res.text();
      let body;
      try {
        body = JSON.parse(text);
      } catch {
        lastErr = new Error(
          `Expected JSON from dashboard API. Got non-JSON (is Flask running?). Tried: ${url.slice(0, 48)}…`
        );
        continue;
      }
      if (!res.ok) {
        lastErr = new Error(body.error || `HTTP ${res.status} from ${url}`);
        continue;
      }
      return normalizeDashboardPayload(body);
    } catch (e) {
      lastErr = e instanceof Error ? e : new Error(String(e));
    }
  }
  throw lastErr;
}
