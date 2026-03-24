from __future__ import annotations



import json

import os

import uuid

from dataclasses import asdict, replace

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

PROJECT_ROOT = APP_DIR.parent

UPLOAD_DIR = PROJECT_ROOT / "_uploads"

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


def _load_dataframe_flexible(file_path: str) -> pd.DataFrame:
    """
    Load csv/xlsx and tolerate non-'rpm' sheet names for xlsx.
    """
    path = str(file_path)
    if path.lower().endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(path, sheet_name="rpm")
        except Exception:
            return pd.read_excel(path, sheet_name=0)
    return pd.read_csv(path)


def _events_from_mask(mask: np.ndarray, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    if idx.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    s = int(idx[0])
    p = int(idx[0])
    for h in idx[1:]:
        h = int(h)
        if h <= p + 1:
            p = h
            continue
        segments.append((s, p))
        s = p = h
    segments.append((s, p))
    segments = [(a, b) for (a, b) in segments if (b - a + 1) >= int(max(min_len, 1))]
    if not segments:
        return []
    merged = [segments[0]]
    for a, b in segments[1:]:
        la, lb = merged[-1]
        if a <= lb + int(max(merge_gap, 0)) + 1:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def _overlap_len(a1: int, b1: int, a2: int, b2: int) -> int:
    left = max(a1, a2)
    right = min(b1, b2)
    return max(0, right - left + 1)


def _detector_config_for_sensitivity(sensitivity: str) -> DetectorConfig:
    """根据前端选择的高/中/低灵敏度返回对应检测配置。高=更多异常，低=更少异常。"""
    sensitivity = (sensitivity or "medium").strip().lower()
    if sensitivity == "high":
        return DetectorConfig(
            share_z=2.0,
            growth_rate_threshold=0.5,
            user_k=1.5,
            abs_z_extreme=4.0,
            abs_z_peak=3.0,
            abs_z_episode=2.0,
        )
    if sensitivity == "low":
        return DetectorConfig(
            share_z=4.0,
            growth_rate_threshold=1.2,
            user_k=2.5,
            abs_z_extreme=6.0,
            abs_z_peak=4.0,
            abs_z_episode=3.0,
        )
    # medium：使用默认
    return DetectorConfig()


def _apply_pool_sensitivity(cfg: DetectorConfig, pool_sensitivity: str) -> DetectorConfig:
    """
    池子（系统）异常灵敏度：只影响系统层面的 ratio 带宽与单点极端尖峰阈值。

    - ratio_threshold: 高/中/低 = 1.10 / 1.25 / 1.40
    - sys_extreme_ratio: 高/中/低 = 1.50 / 1.70 / 1.90

    说明：这里的“高”表示更敏感（更容易报异常）。
    """
    s = (pool_sensitivity or "medium").strip().lower()
    mapping = {
        "high": (1.10, 1.50),
        "medium": (1.25, 1.70),
        "low": (1.40, 1.90),
    }
    ratio_threshold, extreme_ratio = mapping.get(s, mapping["medium"])
    return replace(cfg, sys_ratio_threshold=float(ratio_threshold), sys_extreme_ratio=float(extreme_ratio))


def _apply_major_tenant_sensitivity(cfg: DetectorConfig, tenant_sensitivity: str) -> DetectorConfig:
    """
    大体量用户识别敏感度：
    仅影响用户侧“绝对量显著偏高”相关门槛，避免系统事件内漏掉高贡献用户。

    - high: 更容易识别（阈值更低）
    - medium: 默认平衡
    - low: 更保守（阈值更高）
    """
    s = (tenant_sensitivity or "medium").strip().lower()
    mapping = {
        "high": (3.0, 1.6, 4.5),   # abs_z_peak, user_k, abs_z_extreme
        "medium": (3.5, 2.0, 5.0),
        "low": (4.0, 2.4, 5.8),
    }
    abs_z_peak, user_k, abs_z_extreme = mapping.get(s, mapping["medium"])
    return replace(
        cfg,
        abs_z_peak=float(abs_z_peak),
        user_k=float(user_k),
        abs_z_extreme=float(abs_z_extreme),
    )


# 与 aggregate_pool_hours 约定：sheet 名形如 pool_key__service_tag，全部合计为 __ALL（或 ALL / 全部）
POOL_SHEET_SEP = "__"
POOL_ALL_MARKERS = frozenset({"ALL", "全部", "_ALL_"})


def parse_pool_sheet_name(sheet_name: str) -> tuple[str, str, bool]:
    """解析池子 Excel 的 sheet 名：返回 (池子键, 服务片段, 是否为全部合计)。"""
    s = str(sheet_name).strip()
    if POOL_SHEET_SEP not in s:
        return s, "", True
    a, b = s.split(POOL_SHEET_SEP, 1)
    a, b = a.strip(), b.strip()
    is_all = b in POOL_ALL_MARKERS or b == ""
    return a, b, is_all


def build_pool_sheet_groups(sheet_names: list[str]) -> list[dict[str, Any]]:
    """供模板 optgroup 使用：每个池子一组，组内为各 service（或全部合计）。"""
    from collections import defaultdict

    buckets: dict[str, list[str]] = defaultdict(list)
    for sn in sheet_names:
        pk, _, _ = parse_pool_sheet_name(sn)
        buckets[pk].append(sn)

    groups: list[dict[str, Any]] = []
    for pool_key in sorted(buckets.keys()):
        options: list[dict[str, str]] = []
        for sn in sorted(buckets[pool_key]):
            _, svc, is_all = parse_pool_sheet_name(sn)
            if is_all:
                label = "全部合计"
            elif svc:
                label = svc
            else:
                label = "全部合计"
            options.append({"sheet_name": sn, "label": label})
        options.sort(key=lambda o: (0 if o["label"] == "全部合计" else 1, o["label"]))
        groups.append({"pool_key": pool_key, "options": options})
    return groups





def create_app() -> Flask:

    # webapp.py 移到 code/ 后，显式指定模板和静态资源目录

    app = Flask(

        __name__,

        template_folder=str(PROJECT_ROOT / "templates"),

        static_folder=str(PROJECT_ROOT / "static"),

    )

    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-dev-secret")



    # In-memory session cache: session_id -> dict(result, meta, created_at)

    app.config["SESSIONS"] = {}

    # 池子分析：上传 result Excel 后的临时信息 upload_id -> { file_path, sheet_names, file_name }

    app.config["POOL_UPLOADS"] = {}



    @app.get("/")

    def index():

        return render_template("pool_upload.html")



    @app.post("/detect")

    def detect():
        flash("首页“上传文件检测”入口已废弃，请使用按池子分析（上传 RPM+TPM 双文件）。", "warning")
        return redirect(url_for("pool_index"))

    # ---------- 池子分析流程：上传 RPM/TPM 聚合 Excel -> 选 sheet -> 联合分析 ----------



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



        filename_rpm = secure_filename(f_rpm.filename)
        filename_tpm = secure_filename(f_tpm.filename)

        session_id = uuid.uuid4().hex

        saved_path_rpm = UPLOAD_DIR / f"{session_id}__rpm__{filename_rpm}"
        saved_path_tpm = UPLOAD_DIR / f"{session_id}__tpm__{filename_tpm}"

        f_rpm.save(saved_path_rpm)
        f_tpm.save(saved_path_tpm)



        try:

            df_rpm = _load_dataframe_flexible(str(saved_path_rpm))
            df_tpm = _load_dataframe_flexible(str(saved_path_tpm))

            df_rpm, h_cols_rpm = validate_and_prepare(df_rpm)
            df_tpm, h_cols_tpm = validate_and_prepare(df_tpm)

            if h_cols_rpm != h_cols_tpm:
                raise ValueError("RPM 与 TPM 文件时间列不一致，请确保列名与顺序完全相同。")

            # 用户集合按交集对齐，避免双文件存在缺失行导致错位。
            rpm_uid = df_rpm["user_id"].astype(str)
            tpm_uid = df_tpm["user_id"].astype(str)
            shared = [u for u in rpm_uid.tolist() if u in set(tpm_uid.tolist())]
            if not shared:
                raise ValueError("RPM 与 TPM 文件没有共同用户，无法联合分析。")
            if len(shared) < len(rpm_uid) or len(shared) < len(tpm_uid):
                flash(f"RPM/TPM 用户集合不完全一致，已按交集 {len(shared)} 个用户联合分析。", "warning")
            df_rpm = df_rpm[df_rpm["user_id"].astype(str).isin(shared)].copy()
            df_tpm = df_tpm[df_tpm["user_id"].astype(str).isin(shared)].copy()
            df_rpm["__order"] = pd.Categorical(df_rpm["user_id"].astype(str), categories=shared, ordered=True)
            df_tpm["__order"] = pd.Categorical(df_tpm["user_id"].astype(str), categories=shared, ordered=True)
            df_rpm = df_rpm.sort_values("__order").drop(columns="__order").reset_index(drop=True)
            df_tpm = df_tpm.sort_values("__order").drop(columns="__order").reset_index(drop=True)

            h_cols = h_cols_rpm
            res_rpm = detect_anomalies(cfg, df_rpm, h_cols)
            res_tpm = detect_anomalies(cfg, df_tpm, h_cols)



            hours = len(h_cols)

            t = build_time_index(start_time=start_time, hours=hours)



            # Prepare figures

            system_fig = build_system_figure(

                t,

                res_rpm["S"],

                res_rpm["system_median"],

                res_rpm["system_ratio"],

                res_rpm["sys_anom"],

                res_rpm["sys_event_mask"],

                res_rpm["events"],

                cfg.sys_ratio_threshold,

            )

            system_fig_html = pio.to_html(system_fig, full_html=False, include_plotlyjs="inline")

            # 联合事件：RPM/TPM 事件掩码并集
            mask_rpm = np.asarray(res_rpm["sys_event_mask"], dtype=bool)
            mask_tpm = np.asarray(res_tpm["sys_event_mask"], dtype=bool)
            sys_event_mask_joint = mask_rpm | mask_tpm
            events_joint = _events_from_mask(sys_event_mask_joint, cfg.event_min_len, cfg.event_merge_gap)

            # 联合事件报告：按窗口分别输出 RPM 与 TPM 根因
            reports_rpm = res_rpm.get("event_reports", [])
            reports_tpm = res_tpm.get("event_reports", [])
            event_reports_joint: list[dict[str, Any]] = []
            for (a, b) in events_joint:
                rpm_active = bool(np.asarray(res_rpm["sys_anom"], dtype=bool)[a : b + 1].any())
                tpm_active = bool(np.asarray(res_tpm["sys_anom"], dtype=bool)[a : b + 1].any())
                if rpm_active and tpm_active:
                    scope = "both"
                elif rpm_active:
                    scope = "rpm_only"
                else:
                    scope = "tpm_only"

                best_rpm = None
                if rpm_active and reports_rpm:
                    best_rpm = max(
                        reports_rpm,
                        key=lambda ev: _overlap_len(a, b, int(ev.get("start_hour", -1)), int(ev.get("end_hour", -1))),
                    )
                best_tpm = None
                if tpm_active and reports_tpm:
                    best_tpm = max(
                        reports_tpm,
                        key=lambda ev: _overlap_len(a, b, int(ev.get("start_hour", -1)), int(ev.get("end_hour", -1))),
                    )

                culprits_rpm = (best_rpm or {}).get("culprits", []) if rpm_active else []
                culprits_tpm = (best_tpm or {}).get("culprits", []) if tpm_active else []
                culprits_primary = culprits_rpm if culprits_rpm else culprits_tpm
                event_reports_joint.append(
                    {
                        "start_hour": int(a),
                        "end_hour": int(b),
                        "duration_hours": int(b - a + 1),
                        "rootcause_scope": scope,
                        "system_peak_hour_rpm": int(a + int(np.argmax(np.asarray(res_rpm["S"])[a : b + 1]))),
                        "system_peak_rpm": float(np.max(np.asarray(res_rpm["S"])[a : b + 1])),
                        "system_peak_hour_tpm": int(a + int(np.argmax(np.asarray(res_tpm["S"])[a : b + 1]))),
                        "system_peak_tpm": float(np.max(np.asarray(res_tpm["S"])[a : b + 1])),
                        "culprits_rpm": culprits_rpm,
                        "culprits_tpm": culprits_tpm,
                        # 兼容旧逻辑（建议模块）
                        "culprits": culprits_primary,
                        "system_peak_hour": int(a + int(np.argmax(np.asarray(res_rpm["S"])[a : b + 1]))),
                        "system_peak_rpm_legacy": float(np.max(np.asarray(res_rpm["S"])[a : b + 1])),
                        "system_peak_rpm": float(np.max(np.asarray(res_rpm["S"])[a : b + 1])),
                    }
                )

            stats_rpm = res_rpm["system_stats"]
            stats_tpm = res_tpm["system_stats"]
            system_stats_joint = {
                "hours": int(len(h_cols)),
                "event_count": int(len(events_joint)),
                "event_hours_count": int(np.asarray(sys_event_mask_joint, dtype=bool).sum()),
                "rpm": {
                    "system_avg": float(stats_rpm.get("system_avg", 0.0)),
                    "system_p95": float(stats_rpm.get("system_p95", 0.0)),
                    "system_max": float(stats_rpm.get("system_max", 0.0)),
                    "system_anom_hours_count": int(stats_rpm.get("system_anom_hours_count", 0)),
                },
                "tpm": {
                    "system_avg": float(stats_tpm.get("system_avg", 0.0)),
                    "system_p95": float(stats_tpm.get("system_p95", 0.0)),
                    "system_max": float(stats_tpm.get("system_max", 0.0)),
                    "system_anom_hours_count": int(stats_tpm.get("system_anom_hours_count", 0)),
                },
            }

            rec_rpm_by_uid = {r["user_id"]: r for r in res_rpm["records"].to_dict(orient="records")}
            rec_tpm_by_uid = {r["user_id"]: r for r in res_tpm["records"].to_dict(orient="records")}
            user_ids = res_rpm["user_ids"]
            records_joint = []
            for i, uid in enumerate(user_ids):
                rr = rec_rpm_by_uid.get(uid, {})
                rt = rec_tpm_by_uid.get(uid, {})
                hit_rpm = int(np.asarray(res_rpm["flags"], dtype=bool)[i].sum())
                hit_tpm = int(np.asarray(res_tpm["flags"], dtype=bool)[i].sum())
                records_joint.append(
                    {
                        "user_id": uid,
                        "hit_count_rpm": hit_rpm,
                        "hit_count_tpm": hit_tpm,
                        "hit_count_any": int(hit_rpm > 0 or hit_tpm > 0),
                        "avg_rpm": float(np.mean(np.asarray(res_rpm["X"])[i])),
                        "p95_rpm": float(np.quantile(np.asarray(res_rpm["X"])[i], 0.95)),
                        "max_rpm": float(np.max(np.asarray(res_rpm["X"])[i])),
                        "avg_tpm": float(np.mean(np.asarray(res_tpm["X"])[i])),
                        "p95_tpm": float(np.quantile(np.asarray(res_tpm["X"])[i], 0.95)),
                        "max_tpm": float(np.max(np.asarray(res_tpm["X"])[i])),
                        "reason_rpm": rr.get("reason", ""),
                        "reason_tpm": rt.get("reason", ""),
                        "hit_hours_rpm": rr.get("hit_hours", ""),
                        "hit_hours_tpm": rt.get("hit_hours", ""),
                    }
                )
            records_joint.sort(key=lambda x: (-(x["hit_count_rpm"] + x["hit_count_tpm"]), -x["max_rpm"], -x["max_tpm"]))



            # Store minimal data for follow-up pages

            app.config["SESSIONS"][session_id] = {

                "created_at": datetime.now(timezone.utc).isoformat(),

                "file_name": f"RPM={filename_rpm} ; TPM={filename_tpm}",

                "saved_path": str(saved_path_rpm),
                "saved_path_rpm": str(saved_path_rpm),
                "saved_path_tpm": str(saved_path_tpm),

                "start_time": start_time.isoformat(timespec="minutes"),

                "cfg": asdict(cfg),
                "analysis_mode": "dual",

                "t": [dt.isoformat() for dt in t],

                "h_cols": h_cols,

                # For performance, store arrays in JSON-friendly form

                "S": res_rpm["S"].tolist(),
                "S_rpm": res_rpm["S"].tolist(),
                "S_tpm": res_tpm["S"].tolist(),

                "system_median": np.asarray(res_rpm["system_median"], dtype=float).tolist(),
                "system_median_rpm": np.asarray(res_rpm["system_median"], dtype=float).tolist(),
                "system_median_tpm": np.asarray(res_tpm["system_median"], dtype=float).tolist(),

                "system_ratio": np.asarray(res_rpm["system_ratio"], dtype=float).tolist(),
                "system_ratio_rpm": np.asarray(res_rpm["system_ratio"], dtype=float).tolist(),
                "system_ratio_tpm": np.asarray(res_tpm["system_ratio"], dtype=float).tolist(),

                "sys_anom": res_rpm["sys_anom"].astype(bool).tolist(),
                "sys_anom_rpm": res_rpm["sys_anom"].astype(bool).tolist(),
                "sys_anom_tpm": res_tpm["sys_anom"].astype(bool).tolist(),

                "sys_event_mask": np.asarray(sys_event_mask_joint, dtype=bool).tolist(),
                "sys_event_mask_rpm": res_rpm["sys_event_mask"].astype(bool).tolist(),
                "sys_event_mask_tpm": res_tpm["sys_event_mask"].astype(bool).tolist(),

                "events": events_joint,
                "events_rpm": res_rpm["events"],
                "events_tpm": res_tpm["events"],

                "event_reports": event_reports_joint,
                "event_reports_rpm": res_rpm["event_reports"],
                "event_reports_tpm": res_tpm["event_reports"],

                "user_ids": user_ids,

                "records_csv": pd.DataFrame.from_records(records_joint).to_csv(index=False, encoding="utf-8"),

                "records_json": records_joint,
                "records_json_rpm": res_rpm["records"].to_dict(orient="records"),
                "records_json_tpm": res_tpm["records"].to_dict(orient="records"),

                "system_stats": system_stats_joint,
                "system_stats_rpm": stats_rpm,
                "system_stats_tpm": stats_tpm,

                # We keep these for user detail plots

                "X": res_rpm["X"].tolist(),
                "X_rpm": res_rpm["X"].tolist(),
                "X_tpm": res_tpm["X"].tolist(),

                "flags": np.asarray(res_rpm["flags"], dtype=bool).tolist(),
                "flags_rpm": np.asarray(res_rpm["flags"], dtype=bool).tolist(),
                "flags_tpm": np.asarray(res_tpm["flags"], dtype=bool).tolist(),
                "flags_any": (np.asarray(res_rpm["flags"], dtype=bool) | np.asarray(res_tpm["flags"], dtype=bool)).tolist(),

                "growth": res_rpm["growth"].tolist(),
                "growth_rpm": res_rpm["growth"].tolist(),
                "growth_tpm": res_tpm["growth"].tolist(),

                "abs_z": res_rpm["abs_z"].tolist(),
                "abs_z_rpm": res_rpm["abs_z"].tolist(),
                "abs_z_tpm": res_tpm["abs_z"].tolist(),

                "share_z": res_rpm["share_z"].tolist(),
                "share_z_rpm": res_rpm["share_z"].tolist(),
                "share_z_tpm": res_tpm["share_z"].tolist(),

            }



            return render_template(

                "results.html",

                session_id=session_id,

                file_name=f"RPM={filename_rpm} ; TPM={filename_tpm}",

                start_time=start_time.isoformat(timespec="minutes"),

                end_time=t[-1].isoformat(timespec="minutes") if t else "",

                cfg=asdict(cfg),

                system_stats=system_stats_joint,

                records=records_joint[:200],

                event_reports=event_reports_joint,

                system_fig_html=system_fig_html,
                rootcause_mode_label="严格" if str(asdict(cfg).get("rootcause_mode", "strict")).lower() == "strict" else "宽松",

            )

        except Exception as e:

            flash(f"检测失败：{e}", "danger")

            return redirect(url_for("index"))



    @app.get("/pool")

    def pool_index():

        return render_template("pool_upload.html")



    @app.post("/pool")

    def pool_upload():

        f_rpm = request.files.get("file_rpm")
        f_tpm = request.files.get("file_tpm")

        if not f_rpm or not f_rpm.filename or not f_tpm or not f_tpm.filename:

            flash("请同时上传 RPM 与 TPM 两个聚合 Excel 文件。", "danger")

            return redirect(url_for("pool_index"))

        if not (f_rpm.filename.lower().endswith(".xlsx") or f_rpm.filename.lower().endswith(".xls")):
            flash("RPM 文件格式不正确，请上传 Excel（.xlsx / .xls）。", "danger")
            return redirect(url_for("pool_index"))
        if not (f_tpm.filename.lower().endswith(".xlsx") or f_tpm.filename.lower().endswith(".xls")):
            flash("TPM 文件格式不正确，请上传 Excel（.xlsx / .xls）。", "danger")
            return redirect(url_for("pool_index"))

        filename_rpm = secure_filename(f_rpm.filename)
        filename_tpm = secure_filename(f_tpm.filename)

        upload_id = uuid.uuid4().hex

        saved_path_rpm = UPLOAD_DIR / f"pool_{upload_id}__rpm__{filename_rpm}"
        saved_path_tpm = UPLOAD_DIR / f"pool_{upload_id}__tpm__{filename_tpm}"

        f_rpm.save(saved_path_rpm)
        f_tpm.save(saved_path_tpm)

        try:

            xl_rpm = pd.ExcelFile(saved_path_rpm)
            xl_tpm = pd.ExcelFile(saved_path_tpm)
            sheets_rpm = xl_rpm.sheet_names
            sheets_tpm = xl_tpm.sheet_names
            xl_rpm.close()
            xl_tpm.close()

        except Exception as e:

            flash(f"读取 Excel 失败：{e}", "danger")

            return redirect(url_for("pool_index"))

        sheet_names = sorted(set(sheets_rpm).intersection(set(sheets_tpm)))
        if not sheet_names:

            flash("RPM 与 TPM 文件没有共同 sheet，无法联合分析。", "danger")

            return redirect(url_for("pool_index"))

        app.config["POOL_UPLOADS"][upload_id] = {

            "file_path_rpm": str(saved_path_rpm),
            "file_path_tpm": str(saved_path_tpm),

            "sheet_names": sheet_names,

            "file_name_rpm": filename_rpm,
            "file_name_tpm": filename_tpm,

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

            file_name=f"RPM={info['file_name_rpm']} ; TPM={info['file_name_tpm']}",

            sheet_names=info["sheet_names"],

            sheet_groups=build_pool_sheet_groups(info["sheet_names"]),

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

            # 整表先按字符串读，避免首列(用户ID)被 Excel/引擎推断为数字导致显示 0.0；
            # 时间列后续在 validate_and_prepare 里会转成数值
            df_rpm = pd.read_excel(info["file_path_rpm"], sheet_name=sheet_name, dtype=str)
            df_tpm = pd.read_excel(info["file_path_tpm"], sheet_name=sheet_name, dtype=str)

            df_rpm.columns = [str(c) for c in df_rpm.columns]
            df_tpm.columns = [str(c) for c in df_tpm.columns]
            df_rpm.iloc[:, 0] = df_rpm.iloc[:, 0].fillna("").astype(str)
            df_tpm.iloc[:, 0] = df_tpm.iloc[:, 0].fillna("").astype(str)

            if df_rpm.shape[0] == 0 or df_rpm.shape[1] < 25 or df_tpm.shape[0] == 0 or df_tpm.shape[1] < 25:
                raise ValueError("RPM/TPM 该 sheet 行数或列数不足（至少一列用户标识 + 24 列时间）。")

            df_rpm, h_cols_rpm = validate_and_prepare(df_rpm)
            df_tpm, h_cols_tpm = validate_and_prepare(df_tpm)
            if h_cols_rpm != h_cols_tpm:
                raise ValueError("RPM 与 TPM 该 sheet 的时间列不一致。")
            h_cols = h_cols_rpm

            rpm_uid = df_rpm["user_id"].astype(str)
            tpm_uid = df_tpm["user_id"].astype(str)
            shared = [u for u in rpm_uid.tolist() if u in set(tpm_uid.tolist())]
            if not shared:
                raise ValueError("RPM 与 TPM 该 sheet 没有共同用户，无法联合分析。")
            if len(shared) < len(rpm_uid) or len(shared) < len(tpm_uid):
                flash(f"RPM/TPM 用户集合不完全一致，已按交集 {len(shared)} 个用户联合分析。", "warning")
            df_rpm = df_rpm[df_rpm["user_id"].astype(str).isin(shared)].copy()
            df_tpm = df_tpm[df_tpm["user_id"].astype(str).isin(shared)].copy()
            df_rpm["__order"] = pd.Categorical(df_rpm["user_id"].astype(str), categories=shared, ordered=True)
            df_tpm["__order"] = pd.Categorical(df_tpm["user_id"].astype(str), categories=shared, ordered=True)
            df_rpm = df_rpm.sort_values("__order").drop(columns="__order").reset_index(drop=True)
            df_tpm = df_tpm.sort_values("__order").drop(columns="__order").reset_index(drop=True)

            # 起始时间从第一个时间列名解析，如 "2026-01-18 16:00"

            try:

                start_ts = pd.to_datetime(h_cols[0], errors="coerce")

                start_time = start_ts.to_pydatetime() if hasattr(start_ts, "to_pydatetime") else datetime(2026, 1, 1, 0, 0)

            except Exception:

                start_time = datetime(2026, 1, 1, 0, 0)

            sensitivity = request.form.get("sensitivity", "medium").strip() or "medium"
            pool_sensitivity = request.form.get("pool_sensitivity", "medium").strip() or "medium"
            rootcause_mode = request.form.get("rootcause_mode", "strict").strip().lower() or "strict"
            max_events_option = request.form.get("max_events_option", "5").strip().lower()
            if rootcause_mode not in {"strict", "loose"}:
                rootcause_mode = "strict"

            cfg = _detector_config_for_sensitivity(sensitivity)
            cfg = _apply_pool_sensitivity(cfg, pool_sensitivity)
            cfg = replace(cfg, rootcause_mode=rootcause_mode)
            if max_events_option == "all":
                cfg = replace(cfg, max_events=0)
            else:
                try:
                    cfg = replace(cfg, max_events=int(max_events_option))
                except Exception:
                    cfg = replace(cfg, max_events=5)

            res_rpm = detect_anomalies(cfg, df_rpm, h_cols)
            res_tpm = detect_anomalies(cfg, df_tpm, h_cols)

            hours = len(h_cols)

            t = build_time_index(start_time=start_time, hours=hours)

            system_fig = build_system_figure(

                t,

                res_rpm["S"],

                res_rpm["system_median"],

                res_rpm["system_ratio"],

                res_rpm["sys_anom"],

                res_rpm["sys_event_mask"],

                res_rpm["events"],

                cfg.sys_ratio_threshold,

            )

            system_fig_html = pio.to_html(system_fig, full_html=False, include_plotlyjs="inline")

            # 联合事件：RPM/TPM 事件掩码并集
            mask_rpm = np.asarray(res_rpm["sys_event_mask"], dtype=bool)
            mask_tpm = np.asarray(res_tpm["sys_event_mask"], dtype=bool)
            sys_event_mask_joint = mask_rpm | mask_tpm
            events_joint = _events_from_mask(sys_event_mask_joint, cfg.event_min_len, cfg.event_merge_gap)

            reports_rpm = res_rpm.get("event_reports", [])
            reports_tpm = res_tpm.get("event_reports", [])
            event_reports_joint: list[dict[str, Any]] = []
            for (a, b) in events_joint:
                rpm_active = bool(np.asarray(res_rpm["sys_anom"], dtype=bool)[a : b + 1].any())
                tpm_active = bool(np.asarray(res_tpm["sys_anom"], dtype=bool)[a : b + 1].any())
                scope = "both" if (rpm_active and tpm_active) else ("rpm_only" if rpm_active else "tpm_only")

                best_rpm = None
                if rpm_active and reports_rpm:
                    best_rpm = max(
                        reports_rpm,
                        key=lambda ev: _overlap_len(a, b, int(ev.get("start_hour", -1)), int(ev.get("end_hour", -1))),
                    )
                best_tpm = None
                if tpm_active and reports_tpm:
                    best_tpm = max(
                        reports_tpm,
                        key=lambda ev: _overlap_len(a, b, int(ev.get("start_hour", -1)), int(ev.get("end_hour", -1))),
                    )
                culprits_rpm = (best_rpm or {}).get("culprits", []) if rpm_active else []
                culprits_tpm = (best_tpm or {}).get("culprits", []) if tpm_active else []
                culprits_primary = culprits_rpm if culprits_rpm else culprits_tpm
                event_reports_joint.append(
                    {
                        "start_hour": int(a),
                        "end_hour": int(b),
                        "duration_hours": int(b - a + 1),
                        "rootcause_scope": scope,
                        "system_peak_hour_rpm": int(a + int(np.argmax(np.asarray(res_rpm["S"])[a : b + 1]))),
                        "system_peak_rpm": float(np.max(np.asarray(res_rpm["S"])[a : b + 1])),
                        "system_peak_hour_tpm": int(a + int(np.argmax(np.asarray(res_tpm["S"])[a : b + 1]))),
                        "system_peak_tpm": float(np.max(np.asarray(res_tpm["S"])[a : b + 1])),
                        "culprits_rpm": culprits_rpm,
                        "culprits_tpm": culprits_tpm,
                        "culprits": culprits_primary,
                        "system_peak_hour": int(a + int(np.argmax(np.asarray(res_rpm["S"])[a : b + 1]))),
                    }
                )

            stats_rpm = res_rpm["system_stats"]
            stats_tpm = res_tpm["system_stats"]
            system_stats_joint = {
                "hours": int(len(h_cols)),
                "event_count": int(len(events_joint)),
                "event_hours_count": int(np.asarray(sys_event_mask_joint, dtype=bool).sum()),
                "rpm": {
                    "system_avg": float(stats_rpm.get("system_avg", 0.0)),
                    "system_p95": float(stats_rpm.get("system_p95", 0.0)),
                    "system_max": float(stats_rpm.get("system_max", 0.0)),
                    "system_anom_hours_count": int(stats_rpm.get("system_anom_hours_count", 0)),
                },
                "tpm": {
                    "system_avg": float(stats_tpm.get("system_avg", 0.0)),
                    "system_p95": float(stats_tpm.get("system_p95", 0.0)),
                    "system_max": float(stats_tpm.get("system_max", 0.0)),
                    "system_anom_hours_count": int(stats_tpm.get("system_anom_hours_count", 0)),
                },
            }

            rec_rpm_by_uid = {r["user_id"]: r for r in res_rpm["records"].to_dict(orient="records")}
            rec_tpm_by_uid = {r["user_id"]: r for r in res_tpm["records"].to_dict(orient="records")}
            user_ids = res_rpm["user_ids"]
            records_joint = []
            for i, uid in enumerate(user_ids):
                rr = rec_rpm_by_uid.get(uid, {})
                rt = rec_tpm_by_uid.get(uid, {})
                hit_rpm = int(np.asarray(res_rpm["flags"], dtype=bool)[i].sum())
                hit_tpm = int(np.asarray(res_tpm["flags"], dtype=bool)[i].sum())
                records_joint.append(
                    {
                        "user_id": uid,
                        "hit_count_rpm": hit_rpm,
                        "hit_count_tpm": hit_tpm,
                        "hit_count_any": int(hit_rpm > 0 or hit_tpm > 0),
                        "avg_rpm": float(np.mean(np.asarray(res_rpm["X"])[i])),
                        "p95_rpm": float(np.quantile(np.asarray(res_rpm["X"])[i], 0.95)),
                        "max_rpm": float(np.max(np.asarray(res_rpm["X"])[i])),
                        "avg_tpm": float(np.mean(np.asarray(res_tpm["X"])[i])),
                        "p95_tpm": float(np.quantile(np.asarray(res_tpm["X"])[i], 0.95)),
                        "max_tpm": float(np.max(np.asarray(res_tpm["X"])[i])),
                        "reason_rpm": rr.get("reason", ""),
                        "reason_tpm": rt.get("reason", ""),
                        "hit_hours_rpm": rr.get("hit_hours", ""),
                        "hit_hours_tpm": rt.get("hit_hours", ""),
                    }
                )
            records_joint.sort(key=lambda x: (-(x["hit_count_rpm"] + x["hit_count_tpm"]), -x["max_rpm"], -x["max_tpm"]))

            session_id = uuid.uuid4().hex

            app.config["SESSIONS"][session_id] = {

                "created_at": datetime.now(timezone.utc).isoformat(),

                "file_name": f"RPM={info['file_name_rpm']} ; TPM={info['file_name_tpm']} (池子: {sheet_name})",

                "saved_path": info["file_path_rpm"],
                "saved_path_rpm": info["file_path_rpm"],
                "saved_path_tpm": info["file_path_tpm"],

                "start_time": start_time.isoformat(timespec="minutes"),

                "cfg": asdict(cfg),
                "analysis_mode": "dual",

                "t": [dt.isoformat() for dt in t],

                "h_cols": h_cols,

                "S": res_rpm["S"].tolist(),
                "S_rpm": res_rpm["S"].tolist(),
                "S_tpm": res_tpm["S"].tolist(),

                "system_median": np.asarray(res_rpm["system_median"], dtype=float).tolist(),
                "system_median_rpm": np.asarray(res_rpm["system_median"], dtype=float).tolist(),
                "system_median_tpm": np.asarray(res_tpm["system_median"], dtype=float).tolist(),

                "system_ratio": np.asarray(res_rpm["system_ratio"], dtype=float).tolist(),
                "system_ratio_rpm": np.asarray(res_rpm["system_ratio"], dtype=float).tolist(),
                "system_ratio_tpm": np.asarray(res_tpm["system_ratio"], dtype=float).tolist(),

                "sys_anom": res_rpm["sys_anom"].astype(bool).tolist(),
                "sys_anom_rpm": res_rpm["sys_anom"].astype(bool).tolist(),
                "sys_anom_tpm": res_tpm["sys_anom"].astype(bool).tolist(),

                "sys_event_mask": np.asarray(sys_event_mask_joint, dtype=bool).tolist(),
                "sys_event_mask_rpm": res_rpm["sys_event_mask"].astype(bool).tolist(),
                "sys_event_mask_tpm": res_tpm["sys_event_mask"].astype(bool).tolist(),

                "events": events_joint,
                "events_rpm": res_rpm["events"],
                "events_tpm": res_tpm["events"],

                "event_reports": event_reports_joint,
                "event_reports_rpm": res_rpm["event_reports"],
                "event_reports_tpm": res_tpm["event_reports"],

                "user_ids": user_ids,

                "records_csv": pd.DataFrame.from_records(records_joint).to_csv(index=False, encoding="utf-8"),

                "records_json": records_joint,
                "records_json_rpm": res_rpm["records"].to_dict(orient="records"),
                "records_json_tpm": res_tpm["records"].to_dict(orient="records"),

                "system_stats": system_stats_joint,
                "system_stats_rpm": stats_rpm,
                "system_stats_tpm": stats_tpm,

                "X": res_rpm["X"].tolist(),
                "X_rpm": res_rpm["X"].tolist(),
                "X_tpm": res_tpm["X"].tolist(),

                "flags": np.asarray(res_rpm["flags"], dtype=bool).tolist(),
                "flags_rpm": np.asarray(res_rpm["flags"], dtype=bool).tolist(),
                "flags_tpm": np.asarray(res_tpm["flags"], dtype=bool).tolist(),
                "flags_any": (np.asarray(res_rpm["flags"], dtype=bool) | np.asarray(res_tpm["flags"], dtype=bool)).tolist(),

                "growth": res_rpm["growth"].tolist(),
                "growth_rpm": res_rpm["growth"].tolist(),
                "growth_tpm": res_tpm["growth"].tolist(),

                "abs_z": res_rpm["abs_z"].tolist(),
                "abs_z_rpm": res_rpm["abs_z"].tolist(),
                "abs_z_tpm": res_tpm["abs_z"].tolist(),

                "share_z": res_rpm["share_z"].tolist(),
                "share_z_rpm": res_rpm["share_z"].tolist(),
                "share_z_tpm": res_tpm["share_z"].tolist(),

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

        is_dual = str(s.get("analysis_mode", "")) == "dual"

        ratio_threshold = float(s["cfg"].get("sys_ratio_threshold", 1.10))
        if is_dual:
            system_fig = build_system_dual_figure(
                t,
                np_array(s.get("S_rpm", s["S"])),
                np_array(s.get("S_tpm", s["S"])),
                np_bool(s.get("sys_anom_rpm", s["sys_anom"])),
                np_bool(s.get("sys_anom_tpm", s["sys_anom"])),
                s.get("events", []),
            )
            system_fig_html = pio.to_html(system_fig, full_html=False, include_plotlyjs="inline")
            system_fig_tpm_html = ""
        else:
            system_fig = build_system_figure(
                t,
                np_array(s["S"]),
                np_array(s["system_median"]),
                np_array(s["system_ratio"]),
                np_bool(s["sys_anom"]),
                np_bool(s["sys_event_mask"]),
                s.get("events", []),
                ratio_threshold,
            )
            system_fig_html = pio.to_html(system_fig, full_html=False, include_plotlyjs="inline")
            system_fig_tpm_html = ""

        all_users: list[dict[str, Any]] = []
        if is_dual:
            all_users = list(s.get("records_json", []))
            all_users.sort(key=lambda u: (-(int(u.get("hit_count_rpm", 0)) + int(u.get("hit_count_tpm", 0))), -float(u.get("max_rpm", 0.0))))
        else:
            # 池子内全部用户数据：user_id, avg_rpm, max_rpm, p95_rpm, hit_count, reason
            user_ids = s["user_ids"]
            X = np_array(s["X"])
            flags = np_bool_matrix(s["flags"])
            records_by_uid = {r["user_id"]: r for r in s.get("records_json", [])}
            for i, uid in enumerate(user_ids):
                x = X[i]
                rec = records_by_uid.get(uid, {})
                all_users.append(
                    {
                        "user_id": uid,
                        "avg_rpm": float(np.mean(x)),
                        "max_rpm": float(np.max(x)),
                        "p95_rpm": float(np.quantile(x, 0.95)),
                        "hit_count": int(flags[i].sum()),
                        "reason": rec.get("reason", ""),
                    }
                )
            all_users.sort(key=lambda u: (-u["hit_count"], -u["max_rpm"]))

        event_reports_view = _format_event_reports_with_time(s.get("event_reports", []), t)

        rootcause_mode_label = "严格" if str(s.get("cfg", {}).get("rootcause_mode", "strict")).lower() == "strict" else "宽松"

        return render_template(

            "results.html",

            session_id=session_id,

            file_name=s["file_name"],

            start_time=s["start_time"],

            end_time=t[-1].isoformat(timespec="minutes") if t else "",

            cfg=s["cfg"],

            system_stats=s["system_stats"],

            records=s["records_json"][:200],

            event_reports=event_reports_view,
            rootcause_mode_label=rootcause_mode_label,
            is_dual=is_dual,

            all_users=all_users,

            system_fig_html=system_fig_html,
            system_fig_tpm_html=system_fig_tpm_html,

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



        is_dual = str(s.get("analysis_mode", "")) == "dual"
        idx = user_ids.index(user_id)

        t = [datetime.fromisoformat(x) for x in s["t"]]

        x = np_array(s["X"])[idx]
        flags = np_bool_matrix(s["flags"])[idx]
        growth = np_array(s["growth"])[idx]
        abs_z = np_array(s["abs_z"])[idx]
        share_z = np_array(s["share_z"])[idx]
        sys_anom = np_bool(s["sys_anom"])
        sys_event_mask = np_bool(s["sys_event_mask"])

        if is_dual:
            fig_dual = build_user_dual_figure(
                t,
                np_array(s.get("X_rpm", s["X"]))[idx],
                np_array(s.get("X_tpm", s["X"]))[idx],
                np_bool_matrix(s.get("flags_rpm", s["flags"]))[idx],
                np_bool_matrix(s.get("flags_tpm", s["flags"]))[idx],
            )
            user_fig_html = pio.to_html(fig_dual, full_html=False, include_plotlyjs="inline")
            user_fig_tpm_html = ""
        else:
            fig = build_user_figure(t, x, flags, sys_anom, sys_event_mask, growth, abs_z, share_z)
            user_fig_html = pio.to_html(fig, full_html=False, include_plotlyjs="inline")
            user_fig_tpm_html = ""



        # 池子/系统曲线（预测上下界 + 异常点），与结果页一致

        ratio_threshold = float(s["cfg"].get("sys_ratio_threshold", 1.1))
        if is_dual:
            system_fig = build_system_dual_figure(
                t,
                s.get("S_rpm", s["S"]),
                s.get("S_tpm", s["S"]),
                s.get("sys_anom_rpm", s["sys_anom"]),
                s.get("sys_anom_tpm", s["sys_anom"]),
                s.get("events", s["events"]),
            )
            system_fig_html = pio.to_html(system_fig, full_html=False, include_plotlyjs="inline")
            system_fig_tpm_html = ""
        else:
            system_fig = build_system_figure(
                t,
                s["S"],
                s["system_median"],
                s["system_ratio"],
                s["sys_anom"],
                s["sys_event_mask"],
                s["events"],
                ratio_threshold,
            )
            system_fig_html = pio.to_html(system_fig, full_html=False, include_plotlyjs="inline")
            system_fig_tpm_html = ""



        # Lookup summary row

        row = next((r for r in s["records_json"] if r["user_id"] == user_id), None)
        row_dual = None
        if is_dual:
            row_rpm = next((r for r in s.get("records_json_rpm", []) if r["user_id"] == user_id), None)
            row_tpm = next((r for r in s.get("records_json_tpm", []) if r["user_id"] == user_id), None)
            x_rpm = np_array(s.get("X_rpm", s["X"]))[idx]
            x_tpm = np_array(s.get("X_tpm", s["X"]))[idx]
            flags_rpm = np_bool_matrix(s.get("flags_rpm", s["flags"]))[idx]
            flags_tpm = np_bool_matrix(s.get("flags_tpm", s["flags"]))[idx]
            row_dual = {
                "hit_count_rpm": int(flags_rpm.sum()),
                "hit_count_tpm": int(flags_tpm.sum()),
                "avg_rpm": float(np.mean(x_rpm)),
                "p95_rpm": float(np.quantile(x_rpm, 0.95)),
                "max_rpm": float(np.max(x_rpm)),
                "avg_tpm": float(np.mean(x_tpm)),
                "p95_tpm": float(np.quantile(x_tpm, 0.95)),
                "max_tpm": float(np.max(x_tpm)),
                "reason_rpm": (row_rpm or {}).get("reason", ""),
                "reason_tpm": (row_tpm or {}).get("reason", ""),
                "hit_hours_rpm": (row_rpm or {}).get("hit_hours", ""),
                "hit_hours_tpm": (row_tpm or {}).get("hit_hours", ""),
            }



        return render_template(

            "user.html",

            session_id=session_id,

            file_name=s["file_name"],

            user_id=user_id,

            row=row,
            row_dual=row_dual,
            is_dual=is_dual,

            user_fig_html=user_fig_html,
            user_fig_tpm_html=user_fig_tpm_html,

            system_fig_html=system_fig_html,
            system_fig_tpm_html=system_fig_tpm_html,

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
        is_dual = str(s.get("analysis_mode", "")) == "dual"
        if is_dual:
            culprits_for_advice = (ev.get("culprits_rpm", []) or []) + (ev.get("culprits_tpm", []) or [])
            ev_for_advice = dict(ev)
            ev_for_advice["culprits"] = culprits_for_advice
            recommendations = build_recommendations(ev_for_advice)
        else:
            recommendations = build_recommendations(ev)



        # Human-friendly time range

        t = [datetime.fromisoformat(x) for x in s["t"]]

        a = int(ev["start_hour"])

        b = int(ev["end_hour"])

        start_dt = t[a].isoformat(timespec="minutes") if 0 <= a < len(t) else ""

        end_dt = (t[b] + timedelta(hours=1)).isoformat(timespec="minutes") if 0 <= b < len(t) else ""

        event_view = _format_event_reports_with_time([ev], t)[0]



        return render_template(

            "event.html",

            session_id=session_id,

            file_name=s["file_name"],

            event_idx=event_idx,

            ev=event_view,
            is_dual=is_dual,

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





def _format_event_reports_with_time(event_reports: list[dict[str, Any]], t: list[datetime]) -> list[dict[str, Any]]:

    out: list[dict[str, Any]] = []

    n = len(t)

    for ev in event_reports:

        e = dict(ev)

        a = int(e.get("start_hour", -1))

        b = int(e.get("end_hour", -1))

        p = int(e.get("system_peak_hour", -1))
        p_rpm = int(e.get("system_peak_hour_rpm", -1))
        p_tpm = int(e.get("system_peak_hour_tpm", -1))

        e["start_time_label"] = t[a].strftime("%Y-%m-%d %H:%M") if 0 <= a < n else ""

        e["end_time_label"] = t[b].strftime("%Y-%m-%d %H:%M") if 0 <= b < n else ""

        e["system_peak_time_label"] = t[p].strftime("%Y-%m-%d %H:%M") if 0 <= p < n else ""
        e["system_peak_time_label_rpm"] = t[p_rpm].strftime("%Y-%m-%d %H:%M") if 0 <= p_rpm < n else ""
        e["system_peak_time_label_tpm"] = t[p_tpm].strftime("%Y-%m-%d %H:%M") if 0 <= p_tpm < n else ""

        culprits = []

        for c in e.get("culprits", []) or []:

            c2 = dict(c)

            ph = int(c2.get("peak_hour", -1))

            c2["peak_time_label"] = t[ph].strftime("%Y-%m-%d %H:%M") if 0 <= ph < n else ""

            culprits.append(c2)

        e["culprits"] = culprits

        for key in ("culprits_rpm", "culprits_tpm"):
            out_culprits = []
            for c in e.get(key, []) or []:
                c2 = dict(c)
                ph = int(c2.get("peak_hour", -1))
                c2["peak_time_label"] = t[ph].strftime("%Y-%m-%d %H:%M") if 0 <= ph < n else ""
                out_culprits.append(c2)
            e[key] = out_culprits

        out.append(e)

    return out



# 用户详情页上下两图统一宽度与边距，便于时间轴对齐观察

_CHART_WIDTH = 980

_CHART_MARGIN = dict(l=52, r=32, t=60, b=44)





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

        width=_CHART_WIDTH,

        height=600,

        margin=_CHART_MARGIN,

        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),

    )

    return fig


def build_system_dual_figure(
    t: list[datetime],
    s_rpm: Any,
    s_tpm: Any,
    sys_anom_rpm: Any,
    sys_anom_tpm: Any,
    events: Any,
) -> go.Figure:
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    s_rpm = np.asarray(s_rpm, dtype=float)
    s_tpm = np.asarray(s_tpm, dtype=float)
    anom_rpm = np.asarray(sys_anom_rpm, dtype=bool)
    anom_tpm = np.asarray(sys_anom_tpm, dtype=bool)

    fig.add_trace(
        go.Scatter(x=t, y=s_rpm, mode="lines", name="SystemRPM", line=dict(color="#0d6efd")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=t, y=s_tpm, mode="lines", name="SystemTPM", line=dict(color="#20c997")),
        secondary_y=True,
    )

    idx_rpm = np.where(anom_rpm)[0] if len(t) else np.array([], dtype=int)
    if idx_rpm.size:
        fig.add_trace(
            go.Scatter(
                x=[t[i] for i in idx_rpm],
                y=[s_rpm[i] for i in idx_rpm],
                mode="markers",
                name="RPMAnomaly",
                marker=dict(size=8, color="#dc3545", symbol="diamond"),
            ),
            secondary_y=False,
        )

    idx_tpm = np.where(anom_tpm)[0] if len(t) else np.array([], dtype=int)
    if idx_tpm.size:
        fig.add_trace(
            go.Scatter(
                x=[t[i] for i in idx_tpm],
                y=[s_tpm[i] for i in idx_tpm],
                mode="markers",
                name="TPMAnomaly",
                marker=dict(size=7, color="#fd7e14", symbol="circle"),
            ),
            secondary_y=True,
        )

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
            )

    fig.update_yaxes(title_text="RPM", secondary_y=False)
    fig.update_yaxes(title_text="TPM", secondary_y=True)
    fig.update_xaxes(title_text="时间")
    fig.update_layout(
        title="系统总量时序（RPM/TPM 双轴）",
        template="plotly_white",
        width=_CHART_WIDTH,
        height=480,
        margin=_CHART_MARGIN,
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

        width=_CHART_WIDTH,

        height=420,

        margin=_CHART_MARGIN,
        # 图例与池子系统图保持一致：顶部水平放置，避免占用右侧宽度导致时间轴对不齐
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),

    )

    return fig


def build_user_dual_figure(
    t: list[datetime],
    x_rpm: Any,
    x_tpm: Any,
    flags_rpm: Any,
    flags_tpm: Any,
) -> go.Figure:
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    x_rpm = np.asarray(x_rpm, dtype=float)
    x_tpm = np.asarray(x_tpm, dtype=float)
    flags_rpm = np.asarray(flags_rpm, dtype=bool)
    flags_tpm = np.asarray(flags_tpm, dtype=bool)

    fig.add_trace(
        go.Scatter(x=t, y=x_rpm, mode="lines", name="UserRPM", line=dict(color="#0d6efd")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=t, y=x_tpm, mode="lines", name="UserTPM", line=dict(color="#20c997")),
        secondary_y=True,
    )

    idx_rpm = np.where(flags_rpm)[0]
    if idx_rpm.size:
        fig.add_trace(
            go.Scatter(
                x=[t[h] for h in idx_rpm],
                y=[x_rpm[h] for h in idx_rpm],
                mode="markers",
                name="RPMAnomaly",
                marker=dict(size=8, color="#dc3545", symbol="diamond"),
            ),
            secondary_y=False,
        )
    idx_tpm = np.where(flags_tpm)[0]
    if idx_tpm.size:
        fig.add_trace(
            go.Scatter(
                x=[t[h] for h in idx_tpm],
                y=[x_tpm[h] for h in idx_tpm],
                mode="markers",
                name="TPMAnomaly",
                marker=dict(size=7, color="#fd7e14", symbol="circle"),
            ),
            secondary_y=True,
        )

    fig.update_yaxes(title_text="RPM", secondary_y=False)
    fig.update_yaxes(title_text="TPM", secondary_y=True)
    fig.update_xaxes(title_text="时间")
    fig.update_layout(
        title="用户时序（RPM/TPM 双轴）",
        template="plotly_white",
        width=_CHART_WIDTH,
        height=460,
        margin=_CHART_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig





if __name__ == "__main__":

    app = create_app()

    app.run(host="127.0.0.1", port=5000, debug=True)



