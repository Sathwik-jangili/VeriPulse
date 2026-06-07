"""
SQLite analytics: captured live posts + multi-model predictions for the dashboard.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(_ROOT, "backend", "veripulse_analytics.db")


def _db_path() -> str:
    return os.environ.get("VERIPULSE_DB_PATH", DEFAULT_DB_PATH)


def _connect() -> sqlite3.Connection:
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_connection():
    conn = _connect()
    try:
        init_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            source_detail TEXT,
            body TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at);
        CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(platform);

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            dataset TEXT NOT NULL,
            arch TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            confidence REAL NOT NULL,
            prob_reliable REAL,
            prob_unreliable REAL,
            model_loaded INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (post_id) REFERENCES posts(id),
            UNIQUE(post_id, dataset, arch)
        );
        CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);
        CREATE INDEX IF NOT EXISTS idx_pred_combo ON predictions(dataset, arch);
        """
    )


def _content_hash(body: str) -> str:
    normalized = " ".join((body or "").split())
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def upsert_post(
    conn: sqlite3.Connection,
    platform: str,
    body: str,
    source_detail: Optional[str] = None,
) -> int:
    h = _content_hash(body)
    cur = conn.execute("SELECT id FROM posts WHERE content_hash = ?", (h,))
    row = cur.fetchone()
    if row:
        return int(row[0])
    conn.execute(
        """
        INSERT INTO posts (platform, source_detail, body, content_hash)
        VALUES (?, ?, ?, ?)
        """,
        (platform, source_detail or "", body, h),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def upsert_prediction(
    conn: sqlite3.Connection,
    post_id: int,
    dataset: str,
    arch: str,
    prediction: int,
    confidence: float,
    probabilities: Optional[List[float]] = None,
    model_loaded: bool = True,
) -> None:
    pr = pu = None
    if probabilities and len(probabilities) >= 2:
        pr = float(probabilities[0])
        pu = float(probabilities[1])
    conn.execute(
        """
        INSERT INTO predictions (post_id, dataset, arch, prediction, confidence, prob_reliable, prob_unreliable, model_loaded)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(post_id, dataset, arch) DO UPDATE SET
            prediction = excluded.prediction,
            confidence = excluded.confidence,
            prob_reliable = excluded.prob_reliable,
            prob_unreliable = excluded.prob_unreliable,
            model_loaded = excluded.model_loaded,
            created_at = datetime('now')
        """,
        (
            post_id,
            dataset,
            arch,
            int(prediction),
            float(confidence),
            pr,
            pu,
            1 if model_loaded else 0,
        ),
    )


def persist_live_post_with_models(
    platform: str,
    body: str,
    source_detail: Optional[str],
    model_outputs: Dict[str, Dict[str, Any]],
) -> int:
    """model_outputs keys: 'fakeddit:distilbert', etc."""
    with get_connection() as conn:
        post_id = upsert_post(conn, platform, body, source_detail)
        for key, out in model_outputs.items():
            if out.get("error"):
                continue
            if "prediction" not in out:
                continue
            parts = key.split(":", 1)
            if len(parts) != 2:
                continue
            ds, ar = parts
            upsert_prediction(
                conn,
                post_id,
                ds,
                ar,
                int(out["prediction"]),
                float(out.get("confidence", 0)),
                out.get("probabilities"),
                bool(out.get("model_loaded", True)),
            )
        return post_id


def _combo_label_public(combo: str) -> str:
    """Human-readable route label for UI (avoid internal corpus codenames)."""
    m = {
        "fakeddit/distilbert": "Social-style · encoder",
        "fakeddit/hybrid": "Social-style · fusion",
        "liar/distilbert": "News-style · encoder",
        "liar/hybrid": "News-style · fusion",
    }
    return m.get(combo, combo.replace("/", " · "))


def dashboard_summary(conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    close = False
    if conn is None:
        conn = _connect()
        init_schema(conn)
        close = True
    try:
        total_posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        total_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        row_dwp = conn.execute(
            "SELECT COUNT(DISTINCT post_id) FROM predictions"
        ).fetchone()
        posts_with_predictions = int(row_dwp[0]) if row_dwp and row_dwp[0] is not None else 0

        by_platform = [
            {"platform": r[0], "count": r[1]}
            for r in conn.execute(
                "SELECT platform, COUNT(*) AS c FROM posts GROUP BY platform ORDER BY c DESC"
            )
        ]

        combo_rows = conn.execute(
            """
            SELECT dataset || '/' || arch AS combo,
                   SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) AS reliable,
                   SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) AS unreliable,
                   AVG(confidence) AS avg_conf
            FROM predictions
            GROUP BY dataset, arch
            """
        ).fetchall()
        by_combo = [
            {
                "combo": r[0],
                "reliable": int(r[1] or 0),
                "unreliable": int(r[2] or 0),
                "avg_confidence": round(float(r[3] or 0), 4),
            }
            for r in combo_rows
        ]

        # If DISTINCT post_id is 0 but rows exist (legacy / odd DB), use busiest route count ≈ distinct posts when 4 routes/post
        if posts_with_predictions == 0 and total_preds and by_combo:
            try:
                posts_with_predictions = max(
                    int(c["reliable"]) + int(c["unreliable"]) for c in by_combo
                )
            except (TypeError, ValueError):
                pass

        # Fixed four routes (matches predict_all_models) for dashboard bar chart
        route_meta = [
            ("fakeddit", "distilbert", "Social-style · encoder"),
            ("fakeddit", "hybrid", "Social-style · fusion"),
            ("liar", "distilbert", "News-style · encoder"),
            ("liar", "hybrid", "News-style · fusion"),
        ]
        combo_avgs: Dict[Tuple[str, str], Tuple[float, int]] = {}
        for r in conn.execute(
            """
            SELECT dataset, arch, AVG(confidence), COUNT(*)
            FROM predictions
            GROUP BY dataset, arch
            """
        ):
            combo_avgs[(str(r[0]), str(r[1]))] = (float(r[2] or 0), int(r[3] or 0))
        model_confidence_routes: List[Dict[str, Any]] = []
        for ds, ar, short_label in route_meta:
            avg_c, n = combo_avgs.get((ds, ar), (0.0, 0))
            model_confidence_routes.append(
                {
                    "key": f"{ds}:{ar}",
                    "dataset": ds,
                    "arch": ar,
                    "label": short_label,
                    "avg_confidence": round(avg_c, 4),
                    "count": n,
                }
            )

        # Confidence buckets (all predictions)
        buckets = {"50-59": 0, "60-69": 0, "70-79": 0, "80-89": 0, "90-100": 0}
        for (conf,) in conn.execute("SELECT confidence FROM predictions"):
            c = float(conf) * 100
            if c < 60:
                buckets["50-59"] += 1
            elif c < 70:
                buckets["60-69"] += 1
            elif c < 80:
                buckets["70-79"] += 1
            elif c < 90:
                buckets["80-89"] += 1
            else:
                buckets["90-100"] += 1
        order = [
            ("90-100", "#10b981"),
            ("80-89", "#22c55e"),
            ("70-79", "#eab308"),
            ("60-69", "#f59e0b"),
            ("50-59", "#ef4444"),
        ]
        confidence_histogram = [
            {"range": rng, "count": buckets[rng], "color": col} for rng, col in order
        ]

        daily = conn.execute(
            """
            SELECT date(created_at, 'localtime') AS d,
                   COUNT(DISTINCT post_id) AS posts,
                   COUNT(*) AS predictions
            FROM predictions
            WHERE datetime(created_at, 'localtime') >= datetime('now', 'localtime', '-14 days')
            GROUP BY date(created_at, 'localtime')
            ORDER BY d
            """
        ).fetchall()
        daily_activity = [
            {"day": r[0], "posts": int(r[1] or 0), "predictions": int(r[2] or 0)} for r in daily
        ]

        fh_row = conn.execute(
            """
            SELECT AVG(CASE WHEN prediction = 1 THEN 1.0 ELSE 0.0 END), AVG(confidence)
            FROM predictions
            WHERE dataset = 'fakeddit' AND arch = 'hybrid'
            """
        ).fetchone()
        unreliable_fh = float(fh_row[0] or 0) if fh_row else 0.0
        avg_conf_fh = float(fh_row[1] or 0) if fh_row else 0.0

        # Primary pie: fakeddit hybrid if any, else all preds
        pie_r = conn.execute(
            """
            SELECT SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END)
            FROM predictions
            WHERE dataset = 'fakeddit' AND arch = 'hybrid'
            """
        ).fetchone()
        if not pie_r or (pie_r[0] or 0) + (pie_r[1] or 0) == 0:
            pie_r = conn.execute(
                """
                SELECT SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END)
                FROM predictions
                """
            ).fetchone()
        rel = int(pie_r[0] or 0) if pie_r else 0
        unrel = int(pie_r[1] or 0) if pie_r else 0

        insights: List[str] = []
        if total_preds > 0:
            insights.append(
                f"Stored {total_preds} model scores across {total_posts} unique posts. "
                f"Charts use live fetches and Analyze tab runs when configured to persist."
            )
        else:
            insights.append(
                "No analytics yet. Use Live Feed (fetch posts) or run ingestion to populate the database."
            )
        if by_combo:
            best = max(by_combo, key=lambda x: x["reliable"] + x["unreliable"])
            insights.append(
                f"Most scored route: {_combo_label_public(best['combo'])} "
                f"({best['reliable'] + best['unreliable']} predictions)."
            )

        # Aggregate KPIs across all stored predictions
        agg = conn.execute(
            """
            SELECT SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END),
                   AVG(confidence)
            FROM predictions
            """
        ).fetchone()
        reliable_all = int(agg[0] or 0) if agg else 0
        unreliable_all = int(agg[1] or 0) if agg else 0
        avg_conf_all = round(float(agg[2] or 0), 4) if agg else 0.0

        # "Today": compare stored date to local today OR UTC today (stored strings vary)
        posts_today = int(
            conn.execute(
                """
                SELECT COUNT(DISTINCT id) FROM posts
                WHERE strftime('%Y-%m-%d', created_at) = strftime('%Y-%m-%d', 'now', 'localtime')
                   OR strftime('%Y-%m-%d', created_at) = strftime('%Y-%m-%d', 'now')
                """
            ).fetchone()[0]
            or 0
        )
        predictions_today = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM predictions
                WHERE strftime('%Y-%m-%d', created_at) = strftime('%Y-%m-%d', 'now', 'localtime')
                   OR strftime('%Y-%m-%d', created_at) = strftime('%Y-%m-%d', 'now')
                """
            ).fetchone()[0]
            or 0
        )

        best_model_combo = ""
        best_model_display = ""
        if by_combo:
            best_row = max(by_combo, key=lambda x: x["reliable"] + x["unreliable"])
            best_model_combo = best_row["combo"] or ""
            ds, _, ar = best_model_combo.partition("/")
            if ds == "fakeddit":
                best_model_display = "Full fusion (social-style)" if ar == "hybrid" else "Quick scan (social-style)"
            elif ds == "liar":
                best_model_display = "Full fusion (news-style)" if ar == "hybrid" else "Quick scan (news-style)"
            else:
                best_model_display = best_model_combo or "—"
        if not best_model_display and best_model_combo:
            best_model_display = best_model_combo

        recent_activity: List[Dict[str, Any]] = []

        def _append_activity_row(
            pid: int,
            platform: str,
            source_detail: str,
            body: str,
            created_at: Any,
            pred_row: Optional[Tuple[Any, ...]],
        ) -> None:
            if pred_row:
                p0, conf, ds, ar = pred_row
                recent_activity.append(
                    {
                        "post_id": pid,
                        "platform": platform,
                        "source_detail": (source_detail or "")[:120],
                        "text_preview": (body or "")[:220],
                        "created_at": created_at,
                        "prediction": int(p0),
                        "confidence": round(float(conf), 4),
                        "dataset": ds,
                        "arch": ar,
                    }
                )
            else:
                recent_activity.append(
                    {
                        "post_id": pid,
                        "platform": platform,
                        "source_detail": (source_detail or "")[:120],
                        "text_preview": (body or "")[:220],
                        "created_at": created_at,
                        "prediction": None,
                        "confidence": None,
                        "dataset": None,
                        "arch": None,
                    }
                )

        post_rows = conn.execute(
            """
            SELECT id, platform, source_detail, body, created_at
            FROM posts
            ORDER BY datetime(created_at) DESC
            LIMIT 18
            """
        ).fetchall()

        for pr in post_rows:
            pid = int(pr[0])
            pred_row = conn.execute(
                """
                SELECT prediction, confidence, dataset, arch FROM predictions
                WHERE post_id = ?
                ORDER BY CASE WHEN dataset = 'fakeddit' AND arch = 'hybrid' THEN 0 ELSE 1 END,
                         dataset, arch
                LIMIT 1
                """,
                (pid,),
            ).fetchone()
            _append_activity_row(
                pid,
                pr[1],
                pr[2] or "",
                pr[3] or "",
                pr[4],
                pred_row,
            )

        # Fallback: posts table empty (or failed) but predictions exist — join and show latest scores
        if not recent_activity and total_preds > 0:
            pred_only = conn.execute(
                """
                SELECT pr.post_id, pr.prediction, pr.confidence, pr.dataset, pr.arch, pr.created_at,
                       p.platform, p.source_detail, p.body, p.created_at
                FROM predictions pr
                LEFT JOIN posts p ON p.id = pr.post_id
                ORDER BY datetime(COALESCE(pr.created_at, p.created_at)) DESC, pr.id DESC
                LIMIT 36
                """
            ).fetchall()
            seen_pid: set = set()
            for row in pred_only:
                pid = int(row[0])
                if pid in seen_pid:
                    continue
                seen_pid.add(pid)
                p0, conf, ds, ar = int(row[1]), float(row[2]), row[3], row[4]
                pr_at = row[5]
                plat = row[6] or "—"
                src = row[7] or ""
                body = row[8] or ""
                post_at = row[9]
                recent_activity.append(
                    {
                        "post_id": pid,
                        "platform": plat,
                        "source_detail": src[:120],
                        "text_preview": (body or f"(post id {pid})")[:220],
                        "created_at": post_at or pr_at,
                        "prediction": p0,
                        "confidence": round(float(conf), 4),
                        "dataset": ds,
                        "arch": ar,
                    }
                )
                if len(recent_activity) >= 18:
                    break

        return {
            "totals": {
                "posts": total_posts,
                "predictions": total_preds,
                "posts_with_predictions": int(posts_with_predictions or 0),
            },
            "by_platform": by_platform,
            "by_model_combo": by_combo,
            "confidence_histogram": confidence_histogram,
            "daily_activity": daily_activity,
            "reliability_pie": [
                {"name": "Reliable", "value": rel, "color": "#38bdf8"},
                {"name": "Unreliable", "value": unrel, "color": "#f472b6"},
            ],
            "headline": {
                "unreliable_rate_fakeddit_hybrid": round(unreliable_fh, 4),
                "avg_confidence_fakeddit_hybrid": round(avg_conf_fh, 4),
            },
            "kpis": {
                "reliable_predictions": reliable_all,
                "unreliable_predictions": unreliable_all,
                "avg_confidence_all": avg_conf_all,
                "posts_today": posts_today,
                "predictions_today": predictions_today,
                "posts_with_predictions": int(posts_with_predictions or 0),
                "best_model_combo": best_model_combo,
                "best_model_display": best_model_display,
            },
            "model_confidence_routes": model_confidence_routes,
            "recent_activity": recent_activity,
            "insights": insights,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        if close:
            conn.close()
