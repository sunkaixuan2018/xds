from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from detector import DetectorConfig, build_time_index, detect_anomalies, load_dataframe, validate_and_prepare


APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _get_float(form: Any, key: str, default: float) -> float:
    v = form.get(key, None)
    if v is None:
        return default
    if isinstance(v, str) and v.strip() == "":
        return default
    return float(v)


def _get_int(form: Any, key: str, default: int) -> int:
    v = form.get(key, None)
    if v is None:
        return default
    if isinstance(v, str) and v.strip() == "":
        return default
    return int(float(v))


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-dev-secret")

    # In-memory session cache: session_id -> dict(result, meta, created_at)
    app.config["SESSIONS"] = {}
    # 池子分析：上传 result Excel 后的临时信息 upload_id -> { file_path, sheet_names, file_name }
    app.config["POOL_UPLOADS"] = {}

    @app.get("/")
    def index():
        """首页即按池子分析：上传 result Excel"""
        return render_template("pool_upload.html")

    @app.post("/detect")
    def detect():
        f = request.files.get("file")
        if not f or not f.filename:
            flash("请先选择一个本地文件（xlsx/csv）。", "danger")
            return redirect(url_for("index"))

        start_time_str = request.form.get("start_time", "").strip()
        if not start_time_str:
            flash("请填写起始时间（用于把 H0..H167 映射为 datetime 轴）。", "danger")
            return redirect(url_for("index"))

        try:
            # datetime-local returns naive local time; keep as naive and treat as local display
            start_time = datetime.fromisoformat(start_time_str)
        except Exception:
            flash("起始时间格式不正确，请使用页面控件选择。", "danger")
            return redirect(url_for("index"))

        # Read params (with fallbacks). Treat empty strings as defaults.
        form = request.form
        cfg = DetectorConfig(
            baseline_window_hours=_get_int(form, "baseline_window_hours", 24),
            min_baseline_points=_get_int(form, "min_baseline_points", 8),
            sys_robust_z=_get_float(form, "sys_robust_z", 6.0),
            sys_growth_rate_threshold=_get_float(form, "sys_growth_rate_threshold", 0.15),
            sys_ratio_threshold=_get_float(form, "sys_ratio_threshold", 1.10),
            event_min_len=_get_int(form, "event_min_len", 2),
            event_merge_gap=_get_int(form, "event_merge_gap", 1),
            seasonal_period=_get_int(form, "seasonal_period", 24),
            seasonal_lookback=_get_int(form, "seasonal_lookback", 6),
            max_events=_get_int(form, "max_events", 2),
            share_z=_get_float(form, "share_z", 3.0),
            growth_rate_threshold=_get_float(form, "growth_rate_threshold", 0.8),
            growth_window_hours=_get_int(form, "growth_window_hours", 3),
            growth_min_hits=_get_int(form, "growth_min_hits", 1),
            user_k=_get_float(form, "user_k", 2.0),
            abs_z_extreme=_get_float(form, "abs_z_extreme", 5.0),
            abs_z_peak=_get_float(form, "abs_z_peak", 3.5),
            abs_z_episode=_get_float(form, "abs_z_episode", 2.5),
            strict_rootcause=True,
            culprit_top_k=_get_int(form, "culprit_top_k", 2),
            culprit_cum_ratio=_get_float(form, "culprit_cum_ratio", 0.7),
            culprit_min_ratio=_get_float(form, "culprit_min_ratio", 0.05),
        )

        filename = secure_filename(f.filename)
        session_id = uuid.uuid4().hex
        saved_path = UPLOAD_DIR / f"{session_id}__{filename}"
        f.save(saved_path)

        try:
            df = load_dataframe(str(saved_path))
            df, h_cols = validate_and_prepare(df)
            res = detect_anomalies(cfg, df, h_cols)

            hours = len(h_cols)
            t = build_time_index(start_time=start_time, hours=hours)

            # Prepare figures
            system_fig = build_system_figure(
                t,
                res["S"],
                res["system_median"],
                res["system_ratio"],
                res["sys_anom"],
                res["sys_event_mask"],
                res["events"],
                cfg.sys_ratio_threshold,
            )
            system_fig_html = pio.to_html(system_fig, full_html=False, include_plotlyjs="inline")

            # Store minimal data for follow-up pages
            app.config["SESSIONS"][session_id] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "file_name": filename,
                "saved_path": str(saved_path),
                "start_time": start_time.isoformat(timespec="minutes"),
                "cfg": asdict(cfg),
                "t": [dt.isoformat() for dt in t],
                "h_cols": h_cols,
                # For performance, store arrays in JSON-friendly form
                "S": res["S"].tolist(),
                "system_median": np.asarray(res["system_median"], dtype=float).tolist(),
                "system_ratio": np.asarray(res["system_ratio"], dtype=float).tolist(),
                "sys_anom": res["sys_anom"].astype(bool).tolist(),
                "sys_event_mask": res["sys_event_mask"].astype(bool).tolist(),
                "events": res["events"],
                "event_reports": res["event_reports"],
                "user_ids": res["user_ids"],
                "records_csv": res["records"].to_csv(index=False, encoding="utf-8"),
                "records_json": res["records"].to_dict(orient="records"),
                "system_stats": res["system_stats"],
                # We keep these for user detail plots
                "X": res["X"].tolist(),
                "flags": res["flags"].astype(bool).tolist(),
                "growth": res["growth"].tolist(),
                "abs_z": res["abs_z"].tolist(),
                "share_z": res["share_z"].tolist(),
            }

            return render_template(
                "results.html",
                session_id=session_id,
                file_name=filename,
                start_time=start_time.isoformat(timespec="minutes"),
                end_time=t[-1].isoformat(timespec="minutes") if t else "",
                cfg=asdict(cfg),
                system_stats=res["system_stats"],
                records=res["records"].head(200).to_dict(orient="records"),
                event_reports=res["event_reports"],
                system_fig_html=system_fig_html,
            )
        except Exception as e:
            flash(f"检测失败：{e}", "danger")
            return redirect(url_for("index"))

    # ---------- 池子分析流程：上传 result Excel -> 选 sheet -> 分析该池子用户异常 ----------
    @app.get("/pool")
    def pool_index():
        return render_template("pool_upload.html")

    @app.post("/pool")
    def pool_upload():
        f = request.files.get("file")
        if not f or not f.filename:
            flash("请先选择 result 目录下生成的 Excel 文件（.xlsx）。", "danger")
            return redirect(url_for("pool_index"))
        if not (f.filename.lower().endswith(".xlsx") or f.filename.lower().endswith(".xls")):
            flash("请上传 Excel 文件（.xlsx / .xls）。", "danger")
            return redirect(url_for("pool_index"))
        filename = secure_filename(f.filename)
        upload_id = uuid.uuid4().hex
        saved_path = UPLOAD_DIR / f"pool_{upload_id}__{filename}"
        f.save(saved_path)
        try:
            xl = pd.ExcelFile(saved_path)
            sheet_names = xl.sheet_names
            xl.close()
        except Exception as e:
            flash(f"读取 Excel 失败：{e}", "danger")
            return redirect(url_for("pool_index"))
        if not sheet_names:
            flash("该 Excel 中没有任何 sheet。", "danger")
            return redirect(url_for("pool_index"))
        app.config["POOL_UPLOADS"][upload_id] = {
            "file_path": str(saved_path),
            "sheet_names": sheet_names,
            "file_name": filename,
        }
        return redirect(url_for("pool_select", upload_id=upload_id))

    @app.get("/pool/<upload_id>/select")
    def pool_select(upload_id: str):
        info = app.config["POOL_UPLOADS"].get(upload_id)
        if not info:
            flash("上传已过期或无效，请重新上传 Excel。", "warning")
            return redirect(url_for("pool_index"))
        return render_template(
            "pool_select.html",
            upload_id=upload_id,
            file_name=info["file_name"],
            sheet_names=info["sheet_names"],
        )

    @app.post("/pool/<upload_id>/analyze")
    def pool_analyze(upload_id: str):
        info = app.config["POOL_UPLOADS"].get(upload_id)
        if not info:
            flash("上传已过期或无效，请重新上传 Excel。", "warning")
            return redirect(url_for("pool_index"))
        sheet_name = request.form.get("sheet_name", "").strip()
        if not sheet_name or sheet_name not in info["sheet_names"]:
            flash("请选择一个有效的池子（sheet）。", "danger")
            return redirect(url_for("pool_select", upload_id=upload_id))
        try:
            # 第一列（用户标识，如 domain_id）强制按字符串读，避免「用户360」等被读成数字导致显示 0.0
            header_df = pd.read_excel(info["file_path"], sheet_name=sheet_name, nrows=0)
            first_col_name = str(header_df.columns[0]) if len(header_df.columns) else None
            df = pd.read_excel(
                info["file_path"],
                sheet_name=sheet_name,
                dtype={first_col_name: str} if first_col_name else None,
            )
            df.columns = [str(c) for c in df.columns]
            # 按列位置再强制一次：首列必须为字符串（兼容列名异常或 dtype 未生效）
            df.iloc[:, 0] = df.iloc[:, 0].astype(str)
            if df.shape[0] == 0 or df.shape[1] < 25:
                raise ValueError("该 sheet 行数或列数不足（需要至少一列用户标识 + 24 列时间）。")
            df, h_cols = validate_and_prepare(df)
            # 起始时间从第一个时间列名解析，如 "2026-01-18 16:00"
            try:
                start_ts = pd.to_datetime(h_cols[0], errors="coerce")
                start_time = start_ts.to_pydatetime() if hasattr(start_ts, "to_pydatetime") else datetime(2026, 1, 1, 0, 0)
            except Exception:
                start_time = datetime(2026, 1, 1, 0, 0)
            cfg = DetectorConfig()
            res = detect_anomalies(cfg, df, h_cols)
            hours = len(h_cols)
            t = build_time_index(start_time=start_time, hours=hours)
            system_fig = build_system_figure(
                t,
                res["S"],
                res["system_median"],
                res["system_ratio"],
                res["sys_anom"],
                res["sys_event_mask"],
                res["events"],
                cfg.sys_ratio_threshold,
            )
            system_fig_html = pio.to_html(system_fig, full_html=False, include_plotlyjs="inline")
            session_id = uuid.uuid4().hex
            app.config["SESSIONS"][session_id] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "file_name": f"{info['file_name']} (池子: {sheet_name})",
                "saved_path": info["file_path"],
                "start_time": start_time.isoformat(timespec="minutes"),
                "cfg": asdict(cfg),
                "t": [dt.isoformat() for dt in t],
                "h_cols": h_cols,
                "S": res["S"].tolist(),
                "system_median": np.asarray(res["system_median"], dtype=float).tolist(),
                "system_ratio": np.asarray(res["system_ratio"], dtype=float).tolist(),
                "sys_anom": res["sys_anom"].astype(bool).tolist(),
                "sys_event_mask": res["sys_event_mask"].astype(bool).tolist(),
                "events": res["events"],
                "event_reports": res["event_reports"],
                "user_ids": res["user_ids"],
                "records_csv": res["records"].to_csv(index=False, encoding="utf-8"),
                "records_json": res["records"].to_dict(orient="records"),
                "system_stats": res["system_stats"],
                "X": res["X"].tolist(),
                "flags": res["flags"].astype(bool).tolist(),
                "growth": res["growth"].tolist(),
                "abs_z": res["abs_z"].tolist(),
                "share_z": res["share_z"].tolist(),
            }
            return redirect(url_for("results", session_id=session_id))
        except Exception as e:
            flash(f"分析失败：{e}", "danger")
            return redirect(url_for("pool_select", upload_id=upload_id))

    @app.get("/results/<session_id>")
    def results(session_id: str):
        s = _get_session(app, session_id)
        if s is None:
            flash("结果已过期或不存在，请重新上传检测。", "warning")
            return redirect(url_for("index"))

        t = [datetime.fromisoformat(x) for x in s["t"]]
        system_fig = build_system_figure(
            t,
            np_array(s["S"]),
            np_array(s["system_median"]),
            np_array(s["system_ratio"]),
            np_bool(s["sys_anom"]),
            np_bool(s["sys_event_mask"]),
            s.get("events", []),
            float(s["cfg"].get("sys_ratio_threshold", 1.10)),
        )
        # 池子内全部用户数据：user_id, avg_rpm, max_rpm, p95_rpm, hit_count, reason
        user_ids = s["user_ids"]
        X = np_array(s["X"])
        flags = np_bool_matrix(s["flags"])
        records_by_uid = {r["user_id"]: r for r in s.get("records_json", [])}
        all_users = []
        for i, uid in enumerate(user_ids):
            x = X[i]
            rec = records_by_uid.get(uid, {})
            all_users.append({
                "user_id": uid,
                "avg_rpm": float(np.mean(x)),
                "max_rpm": float(np.max(x)),
                "p95_rpm": float(np.quantile(x, 0.95)),
                "hit_count": int(flags[i].sum()),
                "reason": rec.get("reason", ""),
            })
        all_users.sort(key=lambda u: (-u["hit_count"], -u["max_rpm"]))

        return render_template(
            "results.html",
            session_id=session_id,
            file_name=s["file_name"],
            start_time=s["start_time"],
            end_time=t[-1].isoformat(timespec="minutes") if t else "",
            cfg=s["cfg"],
            system_stats=s["system_stats"],
            records=s["records_json"][:200],
            event_reports=s.get("event_reports", []),
            all_users=all_users,
            system_fig_html=pio.to_html(system_fig, full_html=False, include_plotlyjs="inline"),
        )

    @app.get("/user/<session_id>/<user_id>")
    def user_detail(session_id: str, user_id: str):
        s = _get_session(app, session_id)
        if s is None:
            flash("结果已过期或不存在，请重新上传检测。", "warning")
            return redirect(url_for("index"))

        user_ids = s["user_ids"]
        if user_id not in user_ids:
            flash("用户不存在。", "danger")
            return redirect(url_for("results", session_id=session_id))

        idx = user_ids.index(user_id)
        t = [datetime.fromisoformat(x) for x in s["t"]]
        x = np_array(s["X"])[idx]
        flags = np_bool_matrix(s["flags"])[idx]
        growth = np_array(s["growth"])[idx]
        abs_z = np_array(s["abs_z"])[idx]
        share_z = np_array(s["share_z"])[idx]
        sys_anom = np_bool(s["sys_anom"])
        sys_event_mask = np_bool(s["sys_event_mask"])

        fig = build_user_figure(t, x, flags, sys_anom, sys_event_mask, growth, abs_z, share_z)
        user_fig_html = pio.to_html(fig, full_html=False, include_plotlyjs="inline")

        # Lookup summary row
        row = next((r for r in s["records_json"] if r["user_id"] == user_id), None)

        return render_template(
            "user.html",
            session_id=session_id,
            file_name=s["file_name"],
            user_id=user_id,
            row=row,
            user_fig_html=user_fig_html,
            cfg=s["cfg"],
        )

    @app.get("/event/<session_id>/<int:event_idx>")
    def event_detail(session_id: str, event_idx: int):
        s = _get_session(app, session_id)
        if s is None:
            flash("结果已过期或不存在，请重新上传检测。", "warning")
            return redirect(url_for("index"))

        reports = s.get("event_reports", [])
        if event_idx < 0 or event_idx >= len(reports):
            flash("事件不存在。", "danger")
            return redirect(url_for("results", session_id=session_id))

        ev = reports[event_idx]
        recommendations = build_recommendations(ev)

        # Human-friendly time range
        t = [datetime.fromisoformat(x) for x in s["t"]]
        a = int(ev["start_hour"])
        b = int(ev["end_hour"])
        start_dt = t[a].isoformat(timespec="minutes") if 0 <= a < len(t) else ""
        end_dt = (t[b] + timedelta(hours=1)).isoformat(timespec="minutes") if 0 <= b < len(t) else ""

        return render_template(
            "event.html",
            session_id=session_id,
            file_name=s["file_name"],
            event_idx=event_idx,
            ev=ev,
            start_dt=start_dt,
            end_dt=end_dt,
            recommendations=recommendations,
            cfg=s["cfg"],
        )

    @app.get("/export/<session_id>")
    def export(session_id: str):
        s = _get_session(app, session_id)
        if s is None:
            return Response("session not found", status=404)

        fmt = request.args.get("format", "csv").lower()
        if fmt == "json":
            return Response(json.dumps(s["records_json"], ensure_ascii=False, indent=2), mimetype="application/json")
        return Response(s["records_csv"], mimetype="text/csv; charset=utf-8")

    return app


def build_recommendations(ev: dict[str, Any]) -> dict[str, Any]:
    """
    Generate actionable mitigation suggestions based on event + culprit ratios.
    This is intentionally policy-like (what to do), not vendor-specific API calls.
    """
    culprits = ev.get("culprits", []) or []
    top_ratio = float(culprits[0]["excess_ratio"]) if culprits else 0.0

    # Suggested throttling intensity
    if top_ratio >= 0.8:
        throttle = "强"
        tenant_rate_cut = "将主因租户限流到其事件窗口峰值前的 p95 或当前的 30%~50%"
        breaker = "建议开启熔断（10~60s 冷却，半开探测恢复），并返回 429 + Retry-After"
    elif top_ratio >= 0.5:
        throttle = "中"
        tenant_rate_cut = "将主因租户限流到其近期基线的 60%~80%，并限制 burst"
        breaker = "必要时启用短熔断（5~20s）避免持续冲击"
    else:
        throttle = "轻"
        tenant_rate_cut = "轻度限流，优先用整流/爬坡控制增长斜率"
        breaker = "一般不需要熔断，除非 429/错误率持续升高"

    return {
        "summary": {
            "throttle_level": throttle,
            "why": "该事件由主因租户在窗口内的超额贡献占比驱动，优先对主因租户做强处置，避免误伤其他租户。",
        },
        "system_layer": [
            {
                "title": "全局并发保护（Inflight 上限 + 有界队列）",
                "when": "进入事件窗口时立即启用",
                "action": "对下游核心服务设置 max_inflight，上游超出后进入有界排队或快速失败（避免资源水位打满）。",
                "goal": "防止级联故障，保障整体可用性。",
            },
            {
                "title": "全局软限流（保守）",
                "when": "系统水位接近上限或 429 抬头",
                "action": "将系统入口限制在安全水位的 85%~95%，主要压力转由租户级策略承担。",
                "goal": "减少全量 429，优先精确打击 noisy neighbor。",
            },
        ],
        "tenant_layer": [
            {
                "title": "租户级限流（Token Bucket / Sliding Window）",
                "when": "主因租户进入事件窗口",
                "action": tenant_rate_cut,
                "goal": "抑制突发、避免挤占共享资源池。",
            },
            {
                "title": "租户级熔断（Circuit Breaker）",
                "when": "主因租户持续触发 429 / 错误率上升",
                "action": breaker,
                "goal": "快速止损，给系统恢复与扩容时间。",
            },
            {
                "title": "QPS 爬坡（斜率控制）",
                "when": "突发增长斜率过高 / 刚解除限流后",
                "action": "按 5~15s 周期统计错误率：高错误降速；低错误小幅探升；0 错误快步探升但需防请求不足导致无限增长。",
                "goal": "避免 RequestBurstTooFast 类突增保护触发，提升成功率。",
            },
            {
                "title": "退避重试（指数退避 + 抖动）",
                "when": "遇到 429/ServerOverloaded",
                "action": "指数退避（min 100ms~500ms, max 3s~10s）+ 随机抖动，限制最大重试次数；高频重试会更快打满限流配额。",
                "goal": "提高最终成功率，降低重试风暴。",
            },
        ],
        "traffic_engineering": [
            {
                "title": "客户端整流（削峰填谷）",
                "when": "可在客户端/SDK侧改造",
                "action": "用 MQ/本地队列承接峰值，消费端以可控斜率匀速放量（例如从 5QPS 起，每 5 秒 +2QPS，到上限封顶）。",
                "goal": "从源头把陡增变为平滑增长，显著降低 429。",
            },
            {
                "title": "跨资源池分流 / 预扩容",
                "when": "可用多模型/多集群，或可预测突发",
                "action": "跨资源池分流（同资源池多 endpoint 无效）；可预测突发提前扩容；关键业务购买保障/专属单元。",
                "goal": "把突发压力分散或提前消化，提升 SLA。",
            },
        ],
    }


def _get_session(app: Flask, session_id: str) -> dict[str, Any] | None:
    return app.config["SESSIONS"].get(session_id)


def np_array(x: Any) -> Any:
    import numpy as np

    return np.asarray(x, dtype=float)


def np_bool(x: Any) -> Any:
    import numpy as np

    return np.asarray(x, dtype=bool)


def np_bool_matrix(x: Any) -> Any:
    import numpy as np

    return np.asarray(x, dtype=bool)


def build_system_figure(
    t: list[datetime],
    S: Any,
    system_median: Any,
    system_ratio: Any,
    sys_anom: Any,
    sys_event_mask: Any,
    events: Any,
    ratio_threshold: float,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("系统总量 vs 基线（含阈值带）", "ratio = S / seasonal_median（含阈值线）"),
    )

    S = np.asarray(S, dtype=float)
    system_median = np.asarray(system_median, dtype=float)
    system_ratio = np.asarray(system_ratio, dtype=float)

    # Row 1: total + baseline median + threshold band
    fig.add_trace(go.Scatter(x=t, y=S, mode="lines", name="SystemTotalRPM"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=system_median, mode="lines", name="SeasonalMedian", line=dict(dash="dash")), row=1, col=1)

    upper = system_median * ratio_threshold
    lower = system_median * (2.0 - ratio_threshold)  # symmetric band around median in multiplicative sense
    fig.add_trace(
        go.Scatter(x=t, y=upper, mode="lines", name=f"Median*{ratio_threshold:.2f}", line=dict(width=0), showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=lower,
            mode="lines",
            name="ThresholdBand",
            fill="tonexty",
            fillcolor="rgba(13,110,253,0.10)",
            line=dict(width=0),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    anom_idx = np.where(np.asarray(sys_anom, dtype=bool))[0] if len(t) else np.array([], dtype=int)
    if anom_idx.size:
        fig.add_trace(
            go.Scatter(
                x=[t[i] for i in anom_idx],
                y=[S[i] for i in anom_idx],
                mode="markers",
                name="SystemAnomalyPoint",
                marker=dict(size=9, color="#dc3545", symbol="diamond"),
                hovertext=[f"system_anom @ {t[i].isoformat(timespec='minutes')}" for i in anom_idx],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    # Shade event windows
    if events:
        for (a, b) in events:
            a = int(a)
            b = int(b)
            if a < 0 or b >= len(t):
                continue
            fig.add_vrect(
                x0=t[a],
                x1=t[b] + timedelta(hours=1),
                fillcolor="rgba(255, 193, 7, 0.12)",
                line_width=0,
                annotation_text="EventWindow",
                annotation_position="top left",
                row=1,
                col=1,
            )
            fig.add_vrect(
                x0=t[a],
                x1=t[b] + timedelta(hours=1),
                fillcolor="rgba(255, 193, 7, 0.12)",
                line_width=0,
                row=2,
                col=1,
            )

    # Row 2: ratio series + threshold line
    fig.add_trace(go.Scatter(x=t, y=system_ratio, mode="lines", name="Ratio(S/Median)"), row=2, col=1)
    fig.add_hline(y=ratio_threshold, line_dash="dash", line_color="rgba(220,53,69,0.8)", row=2, col=1)

    fig.update_yaxes(title_text="RPM", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.update_xaxes(title_text="时间", row=2, col=1)

    fig.update_layout(
        title="系统总量与基线对比（事件窗口、异常点、阈值带）",
        template="plotly_white",
        height=600,
        margin=dict(l=30, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_user_figure(
    t: list[datetime],
    x: Any,
    flags: Any,
    sys_anom: Any,
    sys_event_mask: Any,
    growth: Any,
    abs_z: Any,
    share_z: Any,
) -> go.Figure:
    x = np.asarray(x, dtype=float)
    flags = np.asarray(flags, dtype=bool)
    sys_anom = np.asarray(sys_anom, dtype=bool)
    sys_event_mask = np.asarray(sys_event_mask, dtype=bool)
    growth = np.asarray(growth, dtype=float)
    abs_z = np.asarray(abs_z, dtype=float)
    share_z = np.asarray(share_z, dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name="UserRPM"))

    hit_idx = np.where(flags)[0]
    if hit_idx.size:
        hover = []
        for h in hit_idx:
            reason = "overload_share_and_growth" if sys_anom[h] else "abs_and_growth"
            sys_state = "event_window" if sys_event_mask[h] else ("system_anom_point" if sys_anom[h] else "normal")
            hover.append(
                "<b>异常点</b><br>"
                + f"time={t[h].isoformat(timespec='minutes')}<br>"
                + f"reason={reason}<br>"
                + f"system_state={sys_state}<br>"
                + f"growth_rate={growth[h]:.2f}<br>"
                + f"abs_z={abs_z[h]:.2f}<br>"
                + f"share_z={share_z[h]:.2f}"
            )
        fig.add_trace(
            go.Scatter(
                x=[t[h] for h in hit_idx],
                y=[x[h] for h in hit_idx],
                mode="markers",
                name="Anomaly",
                marker=dict(size=10, color="#dc3545", symbol="circle"),
                hovertext=hover,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="用户 RPM（异常点已标注）",
        xaxis_title="时间",
        yaxis_title="RPM",
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=20, t=50, b=40),
    )
    return fig


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)

