from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LatencyDetectorConfig:
    ttft_sla: float = 25000.0
    tpot_sla: float = 45.0
    severe_ratio: float = 1.3
    mild_consecutive_windows: int = 2
    event_merge_gap: int = 0
    max_events: int = 5
    baseline_window_points: int = 24
    min_baseline_points: int = 6
    culprit_top_k: int = 3
    culprit_cum_ratio: float = 0.8
    culprit_min_ratio: float = 0.05


REQUIRED_COLUMNS = [
    "domain_id",
    "rpm",
    "tpm",
    "ttft_avg",
    "tpot_avg",
    "prompt_tokens",
    "completion_tokens",
    "collect_time_std",
]

NUMERIC_COLUMNS = [
    "rpm",
    "tpm",
    "ttft_avg",
    "tpot_avg",
    "prompt_tokens",
    "completion_tokens",
]


def validate_latency_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["domain_id"] = out["domain_id"].fillna("").astype(str).str.strip()
    out = out[out["domain_id"] != ""].copy()

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    s = out["collect_time_std"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    parsed = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    fallback_mask = parsed.isna() & s.ne("") & s.notna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(s.loc[fallback_mask], errors="coerce")
    out["collect_time_std_parsed"] = parsed
    out = out.dropna(subset=["collect_time_std_parsed"]).copy()
    if out.empty:
        raise ValueError("No valid rows remained after parsing collect_time_std.")
    return out


def _weighted_average_1d(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return 0.0
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def _weighted_average_ignore_zero_1d(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0) & (values != 0)
    if not valid.any():
        return 0.0
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def _collapse_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (domain_id, ts), g in df.groupby(["domain_id", "collect_time_std_parsed"], sort=True):
        rpm_sum = float(g["rpm"].sum())
        tpm_sum = float(g["tpm"].sum())
        rows.append(
            {
                "domain_id": domain_id,
                "collect_time_std_parsed": ts,
                "rpm": rpm_sum,
                "tpm": tpm_sum,
                "ttft_avg": _weighted_average_ignore_zero_1d(g["ttft_avg"].to_numpy(), g["rpm"].to_numpy()),
                "tpot_avg": _weighted_average_ignore_zero_1d(g["tpot_avg"].to_numpy(), g["rpm"].to_numpy()),
                "prompt_tokens": _weighted_average_1d(g["prompt_tokens"].to_numpy(), g["rpm"].to_numpy()),
                "completion_tokens": _weighted_average_1d(g["completion_tokens"].to_numpy(), g["rpm"].to_numpy()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _infer_time_freq(times: pd.Series) -> pd.Timedelta:
    ordered = pd.Series(sorted(pd.unique(times.dropna())))
    if len(ordered) < 2:
        return pd.Timedelta(hours=1)
    diffs = ordered.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(hours=1)
    return diffs.min()


def _build_metric_matrix(df: pd.DataFrame, user_ids: list[str], time_index: pd.DatetimeIndex, column: str) -> np.ndarray:
    pivot = (
        df.pivot_table(index="domain_id", columns="collect_time_std_parsed", values=column, aggfunc="first")
        .reindex(index=user_ids, columns=time_index)
        .fillna(0.0)
    )
    return pivot.to_numpy(dtype=float)


def _rolling_mean(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    s = pd.Series(arr.astype(float)).shift(1)
    return s.rolling(window=window, min_periods=min_periods).mean().to_numpy()


def _mask_to_events(mask: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    if idx.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for h in idx[1:]:
        h = int(h)
        if h <= prev + 1 + int(max(merge_gap, 0)):
            prev = h
            continue
        segments.append((start, prev))
        start = prev = h
    segments.append((start, prev))
    return segments


def _mark_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if min_run <= 1:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    start: int | None = None
    for idx, val in enumerate(mask):
        if val and start is None:
            start = idx
        if (not val) and start is not None:
            if idx - start >= min_run:
                out[start:idx] = True
            start = None
    if start is not None and (len(mask) - start) >= min_run:
        out[start:] = True
    return out


def _cap_events(
    events: list[tuple[int, int]],
    max_events: int,
    system_ttft: np.ndarray,
    system_tpot: np.ndarray,
    cfg: LatencyDetectorConfig,
) -> list[tuple[int, int]]:
    if not events or max_events <= 0 or len(events) <= max_events:
        return events
    scored: list[tuple[tuple[int, int], float]] = []
    for window in events:
        a, b = window
        ttft_ratio = float(np.nanmax(system_ttft[a : b + 1] / max(cfg.ttft_sla, 1e-9)))
        tpot_ratio = float(np.nanmax(system_tpot[a : b + 1] / max(cfg.tpot_sla, 1e-9)))
        scored.append((window, max(ttft_ratio, tpot_ratio)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [window for window, _ in scored[:max_events]]


def _scope_for_window(
    sys_anom_ttft: np.ndarray,
    sys_anom_tpot: np.ndarray,
    start_hour: int,
    end_hour: int,
) -> str:
    ttft_active = bool(np.asarray(sys_anom_ttft, dtype=bool)[start_hour : end_hour + 1].any())
    tpot_active = bool(np.asarray(sys_anom_tpot, dtype=bool)[start_hour : end_hour + 1].any())
    if ttft_active and tpot_active:
        return "both"
    if ttft_active:
        return "ttft_only"
    return "tpot_only"


def _score_weights(scope: str) -> tuple[float, float, float, float]:
    if scope == "ttft_only":
        return (0.35, 0.15, 0.40, 0.10)
    if scope == "tpot_only":
        return (0.10, 0.25, 0.15, 0.50)
    return (0.225, 0.20, 0.275, 0.30)


def _safe_ratio(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0:
        return np.zeros_like(values, dtype=float)
    return np.asarray(values, dtype=float) / total


def _combined_local_score(
    rpm_excess: np.ndarray,
    tpm_excess: np.ndarray,
    prompt_load_excess: np.ndarray,
    completion_load_excess: np.ndarray,
    weights: tuple[float, float, float, float],
) -> np.ndarray:
    w_rpm, w_tpm, w_prompt, w_completion = weights
    score = np.zeros_like(rpm_excess, dtype=float)
    totals = [
        float(np.sum(rpm_excess)),
        float(np.sum(tpm_excess)),
        float(np.sum(prompt_load_excess)),
        float(np.sum(completion_load_excess)),
    ]
    if totals[0] > 0:
        score += w_rpm * (rpm_excess / totals[0])
    if totals[1] > 0:
        score += w_tpm * (tpm_excess / totals[1])
    if totals[2] > 0:
        score += w_prompt * (prompt_load_excess / totals[2])
    if totals[3] > 0:
        score += w_completion * (completion_load_excess / totals[3])
    return score


def detect_latency_anomalies(cfg: LatencyDetectorConfig, df: pd.DataFrame) -> dict[str, Any]:
    prepared = validate_latency_input(df)
    prepared = _collapse_duplicate_rows(prepared)
    prepared = prepared.sort_values(["domain_id", "collect_time_std_parsed"]).reset_index(drop=True)

    user_ids = sorted(prepared["domain_id"].astype(str).unique().tolist())
    base_time_index = pd.Series(sorted(pd.unique(prepared["collect_time_std_parsed"])))
    freq = _infer_time_freq(base_time_index)
    time_index = pd.date_range(start=base_time_index.min(), end=base_time_index.max(), freq=freq)

    rpm_matrix = _build_metric_matrix(prepared, user_ids, time_index, "rpm")
    tpm_matrix = _build_metric_matrix(prepared, user_ids, time_index, "tpm")
    ttft_matrix = _build_metric_matrix(prepared, user_ids, time_index, "ttft_avg")
    tpot_matrix = _build_metric_matrix(prepared, user_ids, time_index, "tpot_avg")
    prompt_matrix = _build_metric_matrix(prepared, user_ids, time_index, "prompt_tokens")
    completion_matrix = _build_metric_matrix(prepared, user_ids, time_index, "completion_tokens")

    system_rpm = rpm_matrix.sum(axis=0)
    system_tpm = tpm_matrix.sum(axis=0)

    system_ttft = np.array(
        [_weighted_average_ignore_zero_1d(ttft_matrix[:, i], rpm_matrix[:, i]) for i in range(len(time_index))],
        dtype=float,
    )
    system_tpot = np.array(
        [_weighted_average_ignore_zero_1d(tpot_matrix[:, i], rpm_matrix[:, i]) for i in range(len(time_index))],
        dtype=float,
    )
    system_prompt = np.array(
        [_weighted_average_1d(prompt_matrix[:, i], rpm_matrix[:, i]) for i in range(len(time_index))],
        dtype=float,
    )
    system_completion = np.array(
        [_weighted_average_1d(completion_matrix[:, i], rpm_matrix[:, i]) for i in range(len(time_index))],
        dtype=float,
    )

    ttft_heavy = system_ttft >= (cfg.ttft_sla * cfg.severe_ratio)
    tpot_heavy = system_tpot >= (cfg.tpot_sla * cfg.severe_ratio)
    ttft_mild = system_ttft > cfg.ttft_sla
    tpot_mild = system_tpot > cfg.tpot_sla
    sys_anom_ttft = ttft_heavy | _mark_runs(ttft_mild, cfg.mild_consecutive_windows)
    sys_anom_tpot = tpot_heavy | _mark_runs(tpot_mild, cfg.mild_consecutive_windows)
    sys_anom = sys_anom_ttft | sys_anom_tpot

    events = _mask_to_events(sys_anom, cfg.event_merge_gap)
    events = _cap_events(events, cfg.max_events, system_ttft, system_tpot, cfg)
    events = sorted(events, key=lambda item: item[0])

    sys_event_mask = np.zeros_like(sys_anom, dtype=bool)
    for a, b in events:
        sys_event_mask[a : b + 1] = True

    baseline_rpm = np.vstack(
        [_rolling_mean(rpm_matrix[i], cfg.baseline_window_points, cfg.min_baseline_points) for i in range(len(user_ids))]
    )
    baseline_tpm = np.vstack(
        [_rolling_mean(tpm_matrix[i], cfg.baseline_window_points, cfg.min_baseline_points) for i in range(len(user_ids))]
    )
    baseline_prompt = np.vstack(
        [_rolling_mean(prompt_matrix[i], cfg.baseline_window_points, cfg.min_baseline_points) for i in range(len(user_ids))]
    )
    baseline_completion = np.vstack(
        [
            _rolling_mean(completion_matrix[i], cfg.baseline_window_points, cfg.min_baseline_points)
            for i in range(len(user_ids))
        ]
    )
    baseline_rpm = np.nan_to_num(baseline_rpm, nan=0.0)
    baseline_tpm = np.nan_to_num(baseline_tpm, nan=0.0)
    baseline_prompt = np.nan_to_num(baseline_prompt, nan=0.0)
    baseline_completion = np.nan_to_num(baseline_completion, nan=0.0)

    flags = np.zeros_like(rpm_matrix, dtype=bool)
    event_reports: list[dict[str, Any]] = []

    for a, b in events:
        scope = _scope_for_window(sys_anom_ttft, sys_anom_tpot, a, b)
        weights = _score_weights(scope)

        rpm_excess_window = np.clip(rpm_matrix[:, a : b + 1] - baseline_rpm[:, a : b + 1], 0.0, None)
        tpm_excess_window = np.clip(tpm_matrix[:, a : b + 1] - baseline_tpm[:, a : b + 1], 0.0, None)
        prompt_delta_window = np.clip(prompt_matrix[:, a : b + 1] - baseline_prompt[:, a : b + 1], 0.0, None)
        completion_delta_window = np.clip(completion_matrix[:, a : b + 1] - baseline_completion[:, a : b + 1], 0.0, None)
        prompt_load_window = prompt_delta_window * rpm_matrix[:, a : b + 1]
        completion_load_window = completion_delta_window * rpm_matrix[:, a : b + 1]

        rpm_excess_sum = rpm_excess_window.sum(axis=1)
        tpm_excess_sum = tpm_excess_window.sum(axis=1)
        prompt_load_sum = prompt_load_window.sum(axis=1)
        completion_load_sum = completion_load_window.sum(axis=1)

        rpm_ratio = _safe_ratio(rpm_excess_sum)
        tpm_ratio = _safe_ratio(tpm_excess_sum)
        prompt_ratio = _safe_ratio(prompt_load_sum)
        completion_ratio = _safe_ratio(completion_load_sum)

        w_rpm, w_tpm, w_prompt, w_completion = weights
        scores = (
            w_rpm * rpm_ratio
            + w_tpm * tpm_ratio
            + w_prompt * prompt_ratio
            + w_completion * completion_ratio
        )
        score_sum = float(np.sum(scores))
        score_ratio = scores / score_sum if score_sum > 0 else np.zeros_like(scores)

        order = np.argsort(scores)[::-1]
        culprits: list[dict[str, Any]] = []
        cumulative = 0.0
        for idx in order:
            if scores[idx] <= 0:
                break
            ratio = float(score_ratio[idx])
            if culprits and ratio < cfg.culprit_min_ratio:
                break

            local_score = _combined_local_score(
                rpm_excess_window[idx],
                tpm_excess_window[idx],
                prompt_load_window[idx],
                completion_load_window[idx],
                weights,
            )
            peak_offset = int(np.argmax(local_score)) if local_score.size else 0
            peak_hour = int(a + peak_offset)
            flags[idx, peak_hour] = True

            culprits.append(
                {
                    "user_id": user_ids[idx],
                    "score": float(scores[idx]),
                    "score_ratio": ratio,
                    "rpm_excess_ratio": float(rpm_ratio[idx]),
                    "tpm_excess_ratio": float(tpm_ratio[idx]),
                    "prompt_load_ratio": float(prompt_ratio[idx]),
                    "completion_load_ratio": float(completion_ratio[idx]),
                    "peak_hour": peak_hour,
                    "peak_rpm": float(rpm_matrix[idx, peak_hour]),
                    "peak_tpm": float(tpm_matrix[idx, peak_hour]),
                    "peak_prompt_tokens": float(prompt_matrix[idx, peak_hour]),
                    "peak_completion_tokens": float(completion_matrix[idx, peak_hour]),
                    "peak_ttft": float(ttft_matrix[idx, peak_hour]),
                    "peak_tpot": float(tpot_matrix[idx, peak_hour]),
                }
            )
            cumulative += ratio
            if len(culprits) >= cfg.culprit_top_k or cumulative >= cfg.culprit_cum_ratio:
                break

        system_peak_ttft_hour = int(a + np.argmax(system_ttft[a : b + 1])) if b >= a else int(a)
        system_peak_tpot_hour = int(a + np.argmax(system_tpot[a : b + 1])) if b >= a else int(a)

        event_reports.append(
            {
                "start_hour": int(a),
                "end_hour": int(b),
                "duration_hours": int(b - a + 1),
                "rootcause_scope": scope,
                "system_peak_hour_ttft": system_peak_ttft_hour,
                "system_peak_ttft": float(system_ttft[system_peak_ttft_hour]),
                "system_peak_hour_tpot": system_peak_tpot_hour,
                "system_peak_tpot": float(system_tpot[system_peak_tpot_hour]),
                "culprits": culprits,
                "totals": {
                    "rpm_excess": float(np.sum(rpm_excess_sum)),
                    "tpm_excess": float(np.sum(tpm_excess_sum)),
                    "prompt_load_excess": float(np.sum(prompt_load_sum)),
                    "completion_load_excess": float(np.sum(completion_load_sum)),
                },
            }
        )

    reason_map: dict[str, set[str]] = {uid: set() for uid in user_ids}
    for report in event_reports:
        scope = str(report.get("rootcause_scope", "both"))
        for culprit in report.get("culprits", []) or []:
            reason_map[str(culprit["user_id"])].add(scope)

    records: list[dict[str, Any]] = []
    for idx, uid in enumerate(user_ids):
        hit_idx = np.where(flags[idx])[0]
        if hit_idx.size == 0:
            continue
        reasons = sorted(reason_map.get(uid, set()))
        records.append(
            {
                "user_id": uid,
                "hit_count": int(hit_idx.size),
                "hit_hours": ",".join(map(str, hit_idx.tolist())),
                "reason": ",".join(reasons) if reasons else "latency_rootcause",
                "avg_rpm": float(np.mean(rpm_matrix[idx])),
                "p95_rpm": float(np.quantile(rpm_matrix[idx], 0.95)),
                "max_rpm": float(np.max(rpm_matrix[idx])),
                "avg_tpm": float(np.mean(tpm_matrix[idx])),
                "p95_tpm": float(np.quantile(tpm_matrix[idx], 0.95)),
                "max_tpm": float(np.max(tpm_matrix[idx])),
                "avg_ttft": float(np.mean(ttft_matrix[idx])),
                "p95_ttft": float(np.quantile(ttft_matrix[idx], 0.95)),
                "max_ttft": float(np.max(ttft_matrix[idx])),
                "avg_tpot": float(np.mean(tpot_matrix[idx])),
                "p95_tpot": float(np.quantile(tpot_matrix[idx], 0.95)),
                "max_tpot": float(np.max(tpot_matrix[idx])),
                "avg_prompt_tokens": float(np.mean(prompt_matrix[idx])),
                "avg_completion_tokens": float(np.mean(completion_matrix[idx])),
            }
        )

    records_df = pd.DataFrame.from_records(records)
    if not records_df.empty:
        records_df = records_df.sort_values(["hit_count", "max_ttft", "max_tpot", "user_id"], ascending=[False, False, False, True])

    system_stats = {
        "hours": int(len(time_index)),
        "event_count": int(len(events)),
        "event_hours_count": int(np.sum(sys_event_mask)),
        "system_anom_hours_count": int(np.sum(sys_anom)),
        "ttft": {
            "system_avg": float(np.mean(system_ttft)) if len(system_ttft) else 0.0,
            "system_p95": float(np.quantile(system_ttft, 0.95)) if len(system_ttft) else 0.0,
            "system_max": float(np.max(system_ttft)) if len(system_ttft) else 0.0,
            "sla": float(cfg.ttft_sla),
            "severe_threshold": float(cfg.ttft_sla * cfg.severe_ratio),
        },
        "tpot": {
            "system_avg": float(np.mean(system_tpot)) if len(system_tpot) else 0.0,
            "system_p95": float(np.quantile(system_tpot, 0.95)) if len(system_tpot) else 0.0,
            "system_max": float(np.max(system_tpot)) if len(system_tpot) else 0.0,
            "sla": float(cfg.tpot_sla),
            "severe_threshold": float(cfg.tpot_sla * cfg.severe_ratio),
        },
        "rpm": {
            "system_avg": float(np.mean(system_rpm)) if len(system_rpm) else 0.0,
            "system_p95": float(np.quantile(system_rpm, 0.95)) if len(system_rpm) else 0.0,
            "system_max": float(np.max(system_rpm)) if len(system_rpm) else 0.0,
        },
        "tpm": {
            "system_avg": float(np.mean(system_tpm)) if len(system_tpm) else 0.0,
            "system_p95": float(np.quantile(system_tpm, 0.95)) if len(system_tpm) else 0.0,
            "system_max": float(np.max(system_tpm)) if len(system_tpm) else 0.0,
        },
        "prompt_tokens": {
            "system_avg": float(np.mean(system_prompt)) if len(system_prompt) else 0.0,
            "system_p95": float(np.quantile(system_prompt, 0.95)) if len(system_prompt) else 0.0,
            "system_max": float(np.max(system_prompt)) if len(system_prompt) else 0.0,
        },
        "completion_tokens": {
            "system_avg": float(np.mean(system_completion)) if len(system_completion) else 0.0,
            "system_p95": float(np.quantile(system_completion, 0.95)) if len(system_completion) else 0.0,
            "system_max": float(np.max(system_completion)) if len(system_completion) else 0.0,
        },
    }

    return {
        "time_index": [ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts for ts in time_index],
        "time_step_minutes": int(freq.total_seconds() // 60) if freq.total_seconds() > 0 else 60,
        "user_ids": user_ids,
        "rpm": rpm_matrix,
        "tpm": tpm_matrix,
        "ttft": ttft_matrix,
        "tpot": tpot_matrix,
        "prompt_tokens": prompt_matrix,
        "completion_tokens": completion_matrix,
        "system_rpm": system_rpm,
        "system_tpm": system_tpm,
        "system_ttft": system_ttft,
        "system_tpot": system_tpot,
        "system_prompt_tokens": system_prompt,
        "system_completion_tokens": system_completion,
        "sys_anom": sys_anom,
        "sys_anom_ttft": sys_anom_ttft,
        "sys_anom_tpot": sys_anom_tpot,
        "sys_event_mask": sys_event_mask,
        "events": events,
        "event_reports": event_reports,
        "flags": flags,
        "records": records_df,
        "system_stats": system_stats,
        "config_echo": asdict(cfg),
    }
