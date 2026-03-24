from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectorConfig:
    days: int = 7
    hours_per_day: int = 24

    baseline_window_hours: int = 24
    min_baseline_points: int = 8
    eps: float = 1e-9

    # System overload: S(t) > mu + sys_k * sigma
    # Strict event detector (robust z + growth)
    sys_robust_z: float = 4.0
    sys_growth_rate_threshold: float = 0.15  # auxiliary; primary gate uses ratio to seasonal median
    sys_ratio_threshold: float = 1.10  # +10% over seasonal median (same hour-of-day)
    # Single-hour extreme ratio: treat very sharp spikes as standalone events
    sys_extreme_ratio: float = 1.50  # 50%+ over seasonal median, even if not consecutive

    # Warm-up: skip detection for early hours where baselines are unreliable.
    # Typical choice: 48h so seasonal same-hour median has at least 2 historical points.
    warmup_hours: int = 48
    event_min_len: int = 2
    event_merge_gap: int = 1
    seasonal_period: int = 24
    seasonal_lookback: int = 6  # days to look back for same hour-of-day baseline
    max_events: int = 2
    # Soft cap: when capping events, also keep near-tie events whose peak ratio
    # is close to the N-th event peak. Helps preserve more valid segments.
    max_events_keep_tie_ratio: float = 0.85

    # Tenant share anomaly during overload
    share_z: float = 3.0

    # Growth/burst
    growth_rate_threshold: float = 0.8  # 80%+ hourly growth
    growth_window_hours: int = 3
    growth_min_hits: int = 1

    # Absolute anomaly outside overload
    user_k: float = 2.0

    # Extreme-point fallback (avoid missing sharp peaks even if growth is small)
    abs_z_extreme: float = 5.0
    share_z_extreme: float = 5.0
    abs_z_peak: float = 3.5
    abs_z_episode: float = 2.5  # episode mask threshold for peak backfill
    # Strict mode: only label users inside system events
    strict_rootcause: bool = True
    # Culprit selection: keep top users until cumulative_excess_ratio reached
    culprit_top_k: int = 2
    culprit_cum_ratio: float = 0.7
    culprit_min_ratio: float = 0.05
    # In system event windows, trigger user anomaly when excess contribution ratio
    # is high enough, even if share/growth rules are not hit.
    event_contrib_ratio_trigger: float = 0.35


def _parse_hour_columns(df: pd.DataFrame) -> list[str]:
    """Return columns like H0..H167 sorted by hour index."""
    h_cols = []
    for c in df.columns:
        if isinstance(c, str) and c.startswith("H") and c[1:].isdigit():
            h_cols.append(c)
    h_cols.sort(key=lambda x: int(x[1:]))
    return h_cols


def _rolling_mean_std_1d(arr: np.ndarray, window: int, min_periods: int) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(arr.astype(float))
    # Use only past points as baseline (avoid current-point leakage)
    s1 = s.shift(1)
    mu = s1.rolling(window=window, min_periods=min_periods).mean().to_numpy()
    sigma = s1.rolling(window=window, min_periods=min_periods).std(ddof=0).to_numpy()
    return mu, sigma


def _consecutive_true(mask: np.ndarray, min_run: int) -> np.ndarray:
    if min_run <= 1:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= min_run:
            out[i] = True
    return out


def _growth_rate(x: np.ndarray) -> np.ndarray:
    x_prev = np.r_[x[0], x[:-1]]
    g = x / (x_prev + 1.0) - 1.0
    g[0] = 0.0
    return g


def _robust_z_score(x: np.ndarray, window: int, min_periods: int, eps: float) -> np.ndarray:
    """
    Robust z-score using rolling median and MAD on past points only (shifted by 1).
    z = (x - median) / (1.4826 * MAD)
    """
    s = pd.Series(x.astype(float)).shift(1)
    med = s.rolling(window=window, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window=window, min_periods=min_periods).median()
    denom = 1.4826 * mad + eps
    return ((pd.Series(x.astype(float)) - med) / denom).to_numpy()


def _seasonal_robust_z(x: np.ndarray, period: int, lookback: int, eps: float) -> np.ndarray:
    """
    Seasonal robust z-score: for each t, baseline uses {t-period, t-2*period, ...} up to lookback points.
    z = (x(t) - median(hist)) / (1.4826*MAD(hist))
    """
    n = len(x)
    z = np.full(n, np.nan, dtype=float)
    x = x.astype(float)
    for t in range(n):
        hist = []
        for k in range(1, lookback + 1):
            j = t - k * period
            if j < 0:
                break
            hist.append(x[j])
        if len(hist) < 2:
            continue
        hist = np.asarray(hist, dtype=float)
        med = np.median(hist)
        mad = np.median(np.abs(hist - med))
        # Avoid MAD=0 blow-up: use a floor tied to scale of median
        scale_floor = max(1.0, abs(med) * 0.01)
        denom = max(1.4826 * mad, scale_floor) + eps
        z[t] = (x[t] - med) / denom
    return z


def _seasonal_median(x: np.ndarray, period: int, lookback: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    x = x.astype(float)
    for t in range(n):
        hist = []
        for k in range(1, lookback + 1):
            j = t - k * period
            if j < 0:
                break
            hist.append(x[j])
        if len(hist) < 2:
            continue
        out[t] = float(np.median(np.asarray(hist, dtype=float)))
    return out

def _mask_to_events(mask: np.ndarray, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    """Convert a boolean mask to merged (start,end) event windows."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    segments = []
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

    # filter by min_len
    segments = [(a, b) for (a, b) in segments if (b - a + 1) >= min_len]
    if not segments:
        return []

    # merge close segments
    merged = [segments[0]]
    for a, b in segments[1:]:
        la, lb = merged[-1]
        if a <= lb + merge_gap + 1:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def load_dataframe(file_path: str) -> pd.DataFrame:
    if file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path, sheet_name="rpm")
    else:
        df = pd.read_csv(file_path)
    return df


def validate_and_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    数据列自适应规则：
    - 默认**第一列**为用户 ID（不再强制列名为 user_id），内部会统一重命名为 user_id；
    - 其余列视为时间序列列，支持：
      - 旧格式：H0, H1, ...（小时索引）；
      - 新格式：具体时间字符串/Excel 时间（如 2026-01-01 15:00）；
    - 最后一列若为 Total/sum（大小写不敏感），则视为总量列，否则按所有时间列行求和生成 Total。
    """
    if df.shape[1] < 2:
        raise ValueError("数据列不足，至少需要一列用户标识和若干时间列。")

    # 统一把列名转成字符串，兼容 Excel datetime 类型的列名
    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    # 1) 第一列固定视为 user_id
    first_col = df.columns[0]

    # 2) 优先兼容旧格式：显式 user_id + H0..Hn
    h_cols_legacy = _parse_hour_columns(df)
    if "user_id" in df.columns and len(h_cols_legacy) >= 24:
        out = df.copy()
        out["user_id"] = out["user_id"].astype(str)
        h_cols = h_cols_legacy
    else:
        # 3) 新格式：第一列为 user_id，其余为时间列（最后一列可选为总量列）
        out = df.copy()
        # 先根据当前列确定时间列，再添加 user_id，避免 user_id 被算进 h_cols 后被 to_numeric 覆盖成 0
        candidate_cols = list(out.columns[1:])
        if not candidate_cols:
            raise ValueError("缺少时间序列列。请确保第一列为用户标识，后面至少有 24 列时间数据。")

        last_col = candidate_cols[-1]
        last_name = last_col.strip().lower()
        if last_name in {"total", "sum"}:
            time_cols = candidate_cols[:-1]
            total_col = last_col
        else:
            time_cols = candidate_cols
            total_col = None

        h_cols = time_cols
        out["user_id"] = out[first_col].astype(str)

    if len(h_cols) < 24:
        raise ValueError("未找到足够的时间列（H0, H1, ... 或具体时间列）。至少需要 24 列。")

    # 统一保证小时/时间列为数值
    out[h_cols] = out[h_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # 处理 Total 列：已有的直接清洗为数值，否则按时间列求和
    if "Total" in out.columns:
        out["Total"] = pd.to_numeric(out["Total"], errors="coerce").fillna(0)
    else:
        # 如果通过新格式检测到了 total_col，但列名不是 "Total"，沿用其值
        if "total_col" in locals() and total_col is not None and total_col in out.columns:
            out["Total"] = pd.to_numeric(out[total_col], errors="coerce").fillna(0)
        else:
            out["Total"] = out[h_cols].sum(axis=1)

    return out, h_cols


def build_time_index(start_time: datetime, hours: int) -> list[datetime]:
    return [start_time + timedelta(hours=h) for h in range(hours)]


def detect_anomalies(
    cfg: DetectorConfig,
    df: pd.DataFrame,
    h_cols: list[str],
) -> dict[str, Any]:
    user_ids = df["user_id"].to_numpy()
    X = df[h_cols].to_numpy(dtype=float)  # (users, hours)
    hours = X.shape[1]

    # System total
    S = X.sum(axis=0)
    sys_growth = _growth_rate(S)
    sys_burst = sys_growth >= cfg.sys_growth_rate_threshold
    sys_rz_seasonal = _seasonal_robust_z(S, period=cfg.seasonal_period, lookback=cfg.seasonal_lookback, eps=cfg.eps)
    sys_rz_roll = _robust_z_score(S, window=cfg.baseline_window_hours, min_periods=cfg.min_baseline_points, eps=cfg.eps)
    sys_rz = np.where(np.isfinite(sys_rz_seasonal), sys_rz_seasonal, sys_rz_roll)
    sys_level_anom = sys_rz >= cfg.sys_robust_z

    # Primary gate: ratio over seasonal median (filters out normal diurnal ramps)
    med = _seasonal_median(S, period=cfg.seasonal_period, lookback=cfg.seasonal_lookback)
    ratio = np.where(np.isfinite(med) & (med > 0), S / med, np.nan)
    sys_ratio_anom = ratio >= cfg.sys_ratio_threshold

    # System anomaly points:
    # - normal: (ratio gate) AND (robust-z OR burst)
    # - extreme spike: ratio >= sys_extreme_ratio (bypass robust-z/burst), so single-hour sharp deviation is not missed
    sys_anom_normal = sys_ratio_anom & (sys_level_anom | sys_burst)
    sys_anom_extreme = np.isfinite(ratio) & (ratio >= cfg.sys_extreme_ratio)
    sys_anom = sys_anom_normal | sys_anom_extreme

    # Warm-up: do not detect anomalies in early hours
    if cfg.warmup_hours and hours:
        w = int(min(max(cfg.warmup_hours, 0), hours))
        if w > 0:
            sys_anom_normal[:w] = False
            sys_anom_extreme[:w] = False
            sys_anom[:w] = False
    events = _mask_to_events(sys_anom, min_len=cfg.event_min_len, merge_gap=cfg.event_merge_gap)

    # 补充：对“单点极端偏离”的异常也认为是独立事件（即使不连续）
    if hours:
        covered = np.zeros_like(sys_anom, dtype=bool)
        for (a, b) in events:
            covered[a : b + 1] = True
        extra_events: list[tuple[int, int]] = []
        for h in range(hours):
            if covered[h]:
                continue
            # ratio 明显超过更高阈值（例如 50%+），视为单点事件（即使未形成连续窗口）
            if bool(sys_anom_extreme[h]):
                extra_events.append((h, h))
        if extra_events:
            events = list(events) + extra_events

    # Keep only top-N strongest events (by system peak ratio)
    # Important:
    # - cfg.max_events is meant to cap long/continuous event windows for root-cause reporting.
    # - single-hour extreme spikes (ratio >= sys_extreme_ratio) should NOT be dropped by this cap,
    #   otherwise the "3h+3h + many 1h extreme spikes" case will miss the later spikes.
    if events:
        # de-dup in case an extreme point is also part of a longer window (or added twice)
        events = sorted(set(tuple(w) for w in events))

    if events and cfg.max_events and len(events) > cfg.max_events:
        extreme_single_events = []
        normal_events = []
        for (a, b) in events:
            if a == b and bool(sys_anom_extreme[a]):
                extreme_single_events.append((a, b))
            else:
                normal_events.append((a, b))

        # Cap normal events by their peak ratio, but keep "near ties"
        # to avoid over-dropping later but still strong segments.
        if len(normal_events) > cfg.max_events:
            scored = []
            for (a, b) in normal_events:
                peak = float(np.nanmax(ratio[a : b + 1]))
                scored.append(((a, b), peak))
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[: cfg.max_events]
            cutoff_peak = float(top[-1][1]) if top else float("inf")
            keep_tie_ratio = float(max(min(cfg.max_events_keep_tie_ratio, 1.0), 0.0))
            keep_floor = cutoff_peak * keep_tie_ratio
            normal_events = [w for (w, p) in scored if p >= keep_floor]

        # Always keep all extreme single-hour events
        events = sorted(set(normal_events + extreme_single_events))

    # Build event mask for strict root-cause mode
    sys_event_mask = np.zeros_like(sys_anom, dtype=bool)
    for (a, b) in events:
        sys_event_mask[a : b + 1] = True

    # Shares
    denom = np.where(S > 0, S, 1.0)
    P = X / denom
    P_df = pd.DataFrame(P)
    muP = (
        P_df.T.shift(1).rolling(window=cfg.baseline_window_hours, min_periods=cfg.min_baseline_points)
        .mean()
        .T.to_numpy()
    )
    sdP = (
        P_df.T.shift(1).rolling(window=cfg.baseline_window_hours, min_periods=cfg.min_baseline_points)
        .std(ddof=0)
        .T.to_numpy()
    )
    zP = (P - muP) / (sdP + cfg.eps)
    share_anom = zP > cfg.share_z
    share_extreme = zP > cfg.share_z_extreme

    # Per-user abs baseline & z
    abs_z = np.zeros_like(X, dtype=float)
    abs_anom = np.zeros_like(X, dtype=bool)
    mu_abs = np.zeros_like(X, dtype=float)  # baseline mean for excess scoring
    for i in range(X.shape[0]):
        mu, sd = _rolling_mean_std_1d(X[i, :], window=cfg.baseline_window_hours, min_periods=cfg.min_baseline_points)
        mu_abs[i, :] = mu
        abs_z[i, :] = (X[i, :] - mu) / (sd + cfg.eps)
        abs_anom[i, :] = X[i, :] > (mu + cfg.user_k * (sd + cfg.eps))

    # Growth and burst window
    growth = np.zeros_like(X, dtype=float)
    growth_flag = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        g = _growth_rate(X[i, :])
        growth[i, :] = g
        growth_flag[i, :] = g >= cfg.growth_rate_threshold
    growth_hits = (
        pd.DataFrame(growth_flag.astype(int))
        .T.rolling(window=cfg.growth_window_hours, min_periods=1)
        .sum()
        .T.to_numpy()
    )
    growth_burst = growth_hits >= cfg.growth_min_hits

    # Decision logic:
    # - In strict mode, only evaluate tenants inside system events.
    # - If system event hour, prioritize share signals to find culprits.
    sys_anom_mask = (sys_event_mask if cfg.strict_rootcause else sys_anom)[np.newaxis, :]
    abs_extreme = abs_z > cfg.abs_z_extreme

    # In overload/event windows, allow either:
    # - share anomaly with burst/extreme, or
    # - absolute extreme, or
    # - strong absolute elevation (helps catch dominant tenants with stable share).
    flags_overload = sys_anom_mask & (
        (share_anom & (growth_burst | share_extreme))
        | abs_extreme
        | (abs_anom & (abs_z > cfg.abs_z_peak))
    )
    # Outside system anomaly: allow sharp peaks even if growth isn't large
    if cfg.strict_rootcause:
        flags_normal = np.zeros_like(flags_overload, dtype=bool)
    else:
        flags_normal = (~sys_anom_mask) & ((abs_anom & (growth_burst | (abs_z > cfg.abs_z_peak))) | abs_extreme)
    flags = flags_overload | flags_normal

    # Post-process: backfill peak within an anomalous episode.
    # Rationale: the sharpest peak may occur after the steepest growth hour.
    for i in range(X.shape[0]):
        episode_mask = abs_anom[i, :] | (abs_z[i, :] > cfg.abs_z_episode)
        if not episode_mask.any():
            continue
        idx = np.where(episode_mask)[0]
        # build segments of consecutive indices
        seg_start = int(idx[0])
        prev = int(idx[0])
        segments = []
        for h in idx[1:]:
            h = int(h)
            if h == prev + 1:
                prev = h
                continue
            segments.append((seg_start, prev))
            seg_start = prev = h
        segments.append((seg_start, prev))

        for a, b in segments:
            if not flags[i, a : b + 1].any():
                continue
            # mark the peak hour inside this episode
            local = X[i, a : b + 1]
            peak_h = a + int(np.argmax(local))
            flags[i, peak_h] = True

    # Culprit localization by excess contribution inside each event window（仅从未被判定为正常的用户中识别：事件窗口内至少有一个异常点）
    event_reports: list[dict[str, Any]] = []
    event_contrib_flags = np.zeros_like(flags, dtype=bool)
    for (a, b) in events:
        excess = np.clip(X[:, a : b + 1] - mu_abs[:, a : b + 1], 0, None)
        excess_sum_user = excess.sum(axis=1)
        total_excess = float(excess_sum_user.sum())

        selected: list[tuple[int, float]] = []
        if total_excess > 0:
            order = np.argsort(excess_sum_user)[::-1]
            cum = 0.0
            for idx in order[: max(cfg.culprit_top_k, 1) * 6]:
                val = float(excess_sum_user[idx])
                if val <= 0:
                    break
                if (val / total_excess) < cfg.culprit_min_ratio and selected:
                    break
                selected.append((idx, val))
                cum += val
                if len(selected) >= cfg.culprit_top_k or (cum / total_excess) >= cfg.culprit_cum_ratio:
                    break

            # Contribution-based user trigger within event window:
            # mark each major contributor's peak hour in this window as anomalous.
            for idx in order:
                val = float(excess_sum_user[idx])
                if val <= 0:
                    break
                ratio_user = val / total_excess
                if ratio_user < cfg.event_contrib_ratio_trigger:
                    continue
                local = X[idx, a : b + 1]
                peak_h = a + int(np.argmax(local))
                event_contrib_flags[idx, peak_h] = True

        culprits = []
        for (idx, val) in selected:
            uid = str(user_ids[idx])
            window_series = X[idx, a : b + 1]
            peak_local = int(np.argmax(window_series))
            peak_hour = a + peak_local
            culprits.append(
                {
                    "user_id": uid,
                    "excess_sum": float(val),
                    "excess_ratio": float(val / total_excess),
                    "peak_hour": int(peak_hour),
                    "peak_rpm": float(X[idx, peak_hour]),
                }
            )

        event_reports.append(
            {
                "start_hour": int(a),
                "end_hour": int(b),
                "duration_hours": int(b - a + 1),
                "system_peak_hour": int(a + int(np.argmax(S[a : b + 1]))),
                "system_peak_rpm": float(np.max(S[a : b + 1])),
                "total_excess": total_excess,
                "culprits": culprits,
            }
        )

    # Merge contribution-based flags (event-only) into final user flags.
    flags = flags | event_contrib_flags

    # System stats
    system_stats = {
        "hours": int(hours),
        "system_avg": float(np.mean(S)) if hours else 0.0,
        "system_p95": float(np.quantile(S, 0.95)) if hours else 0.0,
        "system_max": float(np.max(S)) if hours else 0.0,
        "event_count": int(len(events)),
        "event_hours_count": int(sys_event_mask.sum()),
        "system_anom_hours_count": int(sys_anom.sum()),
    }

    # Per-user summary + explanation
    records: list[dict[str, Any]] = []
    for i, uid in enumerate(user_ids):
        hit_idx = np.where(flags[i, :])[0]
        if hit_idx.size == 0:
            continue

        hit_hours = hit_idx.tolist()
        during_event = [h for h in hit_hours if sys_event_mask[h]]
        outside_event = [h for h in hit_hours if not sys_event_mask[h]]

        # Decide reason by which rule contributed most for this user
        if len(during_event) >= len(outside_event):
            reason = "overload_share_and_growth" if len(during_event) > 0 else "abs_and_growth"
        else:
            reason = "abs_and_growth"

        # Evidence: summarize on hit hours
        g_vals = growth[i, hit_idx]
        absz_vals = abs_z[i, hit_idx]
        sharez_vals = zP[i, hit_idx]

        rec = {
            "user_id": str(uid),
            "hit_count": int(hit_idx.size),
            "hit_hours": ",".join(map(str, hit_hours[:500])),
            "hits_during_overload": int(len(during_event)),
            "hits_outside_overload": int(len(outside_event)),
            "share_anom_hits": int((share_anom[i, :] & sys_event_mask).sum()),
            "abs_anom_hits": int((abs_anom[i, :] & (~sys_event_mask)).sum()),
            "growth_hits": int(growth_flag[i, :].sum()),
            "avg_rpm": float(np.mean(X[i, :])),
            "max_rpm": float(np.max(X[i, :])),
            "p95_rpm": float(np.quantile(X[i, :], 0.95)),
            "reason": reason,
            "evidence_growth_rate_max": float(np.max(g_vals)) if g_vals.size else 0.0,
            "evidence_abs_z_max": float(np.max(absz_vals)) if absz_vals.size else 0.0,
            "evidence_share_z_max": float(np.max(sharez_vals)) if sharez_vals.size else 0.0,
        }
        records.append(rec)

    records_df = pd.DataFrame.from_records(records)
    if not records_df.empty:
        records_df = records_df.sort_values(["hit_count", "max_rpm", "user_id"], ascending=[False, False, True])

    anomalous_users_count = int(records_df.shape[0])
    system_stats["anomalous_users_count"] = anomalous_users_count
    system_stats["anomalous_ratio"] = anomalous_users_count / max(int(X.shape[0]), 1)

    return {
        "h_cols": h_cols,
        "user_ids": user_ids.tolist(),
        "X": X,  # numeric matrix for plotting (users x hours)
        "S": S,  # system series
        "system_median": med,  # seasonal median baseline (same hour-of-day)
        "system_ratio": ratio,  # S / median
        "sys_anom": sys_anom,  # bool[hours]
        "sys_event_mask": sys_event_mask,  # bool[hours]
        "events": events,  # list[(start,end)]
        "event_reports": event_reports,
        "flags": flags,  # bool[users x hours]
        "growth": growth,  # float[users x hours]
        "abs_z": abs_z,  # float[users x hours]
        "share_z": zP,  # float[users x hours]
        "records": records_df,
        "system_stats": system_stats,
        "config_echo": cfg.__dict__,
    }

