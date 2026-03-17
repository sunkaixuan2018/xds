from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    output_xlsx: str = "fake_rpm_data.xlsx"
    output_results_csv: str = "anomaly_results.csv"

    users: int = 80
    days: int = 7
    hours_per_day: int = 24

    # Fake-data controls
    seed: int = 7
    event_count: int = 2
    # Each event duration in hours (inclusive endpoints will be chosen)
    event_min_len: int = 4
    event_max_len: int = 6
    # Avoid placing events too early to ensure baselines exist
    event_earliest_hour: int = 48
    event_latest_hour: int = 160
    # Root-cause tenants per event (you asked for 2)
    culprit_min: int = 2
    culprit_max: int = 2
    # System-level lift target during event (relative to baseline system total)
    system_lift_min: float = 0.10
    system_lift_max: float = 0.15
    # During event, make culprits more volatile rather than only higher
    culprit_extra_vol_sigma: float = 0.18
    # Ensure culprits are chosen among top baseline users
    culprit_pick_top_frac: float = 0.25
    # Non-culprit coupling during event (small)
    coupling_sigma: float = 0.02

    # Detection controls
    baseline_window_hours: int = 24
    min_baseline_points: int = 8
    eps: float = 1e-9

    # System overload threshold: S(t) > mu + k * sigma
    sys_k: float = 3.0
    sys_min_consecutive: int = 2

    # Tenant share anomaly threshold during overload
    share_z: float = 3.0
    share_min_hits: int = 2

    # RPM slope/burst detection
    # Hour-over-hour growth rate g(t)=x(t)/(x(t-1)+1)-1
    growth_rate_threshold: float = 0.8  # 80%+ hourly growth
    growth_window_hours: int = 3  # lookback window to count bursts
    growth_min_hits: int = 1

    # Non-overload absolute anomaly threshold per user: x(t) > mu + k*sigma
    user_k: float = 2.0
    user_min_hits: int = 2


def hour_columns(total_hours: int) -> list[str]:
    return [f"H{h}" for h in range(total_hours)]


def make_fake_data(cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(cfg.seed)
    total_hours = cfg.days * cfg.hours_per_day
    cols = hour_columns(total_hours)

    # User base levels (heterogeneous tenants), keep overall stable
    base_level = rng.lognormal(mean=2.6, sigma=0.45, size=cfg.users)
    base_level = np.clip(base_level, 5, 400)  # tighter distribution than before to reduce background variance

    # Diurnal pattern: stable periodicity for the whole pool
    hour_of_day = np.arange(total_hours) % cfg.hours_per_day
    diurnal = 0.75 + 0.25 * np.sin((hour_of_day - 8) / 24 * 2 * np.pi)  # mild day-night
    diurnal = np.clip(diurnal, 0.55, 1.05)

    # Build stable per-user series with low noise
    X = np.zeros((cfg.users, total_hours), dtype=float)
    for i in range(cfg.users):
        noise = rng.normal(loc=0.0, scale=0.05, size=total_hours)
        series = base_level[i] * diurnal * (1.0 + noise)
        X[i, :] = np.clip(series, 0, None)

    baseline_X = X.copy()

    # Choose event windows (non-overlapping)
    events = []
    tries = 0
    while len(events) < cfg.event_count and tries < 2000:
        tries += 1
        length = int(rng.integers(cfg.event_min_len, cfg.event_max_len + 1))
        start = int(rng.integers(cfg.event_earliest_hour, cfg.event_latest_hour - length))
        end = start + length - 1
        # no overlap (with 2h buffer)
        ok = True
        for (s, e) in events:
            if not (end + 2 < s or start - 2 > e):
                ok = False
                break
        if ok:
            events.append((start, end))
    events.sort()

    # Choose culprits per event and inject system-visible lifts
    all_culprit_ids: list[str] = []
    truth_rows = []
    # Prefer high baseline users as culprits so their spikes matter at system level
    top_n = max(1, int(math.ceil(cfg.users * cfg.culprit_pick_top_frac)))
    top_users = np.argsort(base_level)[::-1][:top_n]
    for ei, (s, e) in enumerate(events):
        k = int(rng.integers(cfg.culprit_min, min(cfg.culprit_max, cfg.users) + 1))
        culprit_idx = rng.choice(top_users, size=k, replace=False)
        culprit_ids = [f"user_{int(i)+1:03d}" for i in culprit_idx.tolist()]
        all_culprit_ids.extend(culprit_ids)

        # Shape inside window: mild ramp up/down (triangular)
        L = e - s + 1
        ramp = np.r_[np.linspace(0.4, 1.0, num=(L + 1) // 2), np.linspace(1.0, 0.6, num=L - (L + 1) // 2)]
        ramp = np.clip(ramp, 0.4, 1.0)

        # Target system lift each hour relative to baseline system total
        lift = float(rng.uniform(cfg.system_lift_min, cfg.system_lift_max))
        S0 = baseline_X[:, s : e + 1].sum(axis=0)  # baseline system
        target = S0 * (1.0 + lift * ramp)
        delta = np.clip(target - X[:, s : e + 1].sum(axis=0), 0, None)  # how much to add to reach target

        # Allocate delta mainly to culprits proportional to their baseline weights
        w = baseline_X[culprit_idx, s : e + 1]
        w_sum = np.maximum(w.sum(axis=0), 1e-9)
        alloc = (w / w_sum) * delta  # (k, L)
        X[culprit_idx, s : e + 1] += alloc

        # Make culprits more volatile within the event window (stays within a range, but swings harder)
        vol = rng.normal(loc=0.0, scale=cfg.culprit_extra_vol_sigma, size=(k, L))
        X[culprit_idx, s : e + 1] *= np.clip(1.0 + vol, 0.6, 1.8)

        # Non-culprits: small coupling noise, keep system mostly stable outside event
        coupling = 1.0 + rng.normal(loc=0.0, scale=cfg.coupling_sigma, size=(cfg.users, e - s + 1))
        X[:, s : e + 1] *= np.clip(coupling, 0.9, 1.1)

        truth_rows.append(
            {
                "event_id": f"event_{ei+1}",
                "start_hour": s,
                "end_hour": e,
                "culprit_users": ",".join(culprit_ids),
                "system_lift": lift,
            }
        )

    # Attach truth as extra sheet later (returned separately here)
    truth_df = pd.DataFrame.from_records(truth_rows)

    # Round to integers (RPM-like counts for that hour)
    X = np.rint(X).astype(int)

    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "user_id", [f"user_{i+1:03d}" for i in range(cfg.users)])
    df["Total"] = df[cols].sum(axis=1)
    # Return unique culprit ids for console display; truth_df for excel writing handled in main
    unique_culprits = sorted(set(all_culprit_ids))
    df.attrs["truth_events"] = truth_df
    return df, unique_culprits


def rolling_mean_std(arr: np.ndarray, window: int, min_periods: int) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(arr.astype(float))
    mu = s.rolling(window=window, min_periods=min_periods).mean().to_numpy()
    sigma = s.rolling(window=window, min_periods=min_periods).std(ddof=0).to_numpy()
    return mu, sigma


def consecutive_true(mask: np.ndarray, min_run: int) -> np.ndarray:
    if min_run <= 1:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= min_run:
            out[i] = True
    return out


def compute_growth_flags(x: np.ndarray, threshold: float) -> np.ndarray:
    # growth rate g(t)=x(t)/(x(t-1)+1)-1; g(0)=0
    x_prev = np.r_[x[0], x[:-1]]
    g = x / (x_prev + 1.0) - 1.0
    g[0] = 0.0
    return g >= threshold


def detect(cfg: Config, df: pd.DataFrame) -> dict:
    total_hours = cfg.days * cfg.hours_per_day
    cols = hour_columns(total_hours)

    user_ids = df["user_id"].astype(str).to_numpy()
    X = df[cols].to_numpy(dtype=float)  # shape: (users, hours)

    # System total S(t)
    S = X.sum(axis=0)
    muS, sigmaS = rolling_mean_std(S, window=cfg.baseline_window_hours, min_periods=cfg.min_baseline_points)
    sys_overload_raw = S > (muS + cfg.sys_k * (sigmaS + cfg.eps))
    sys_overload = consecutive_true(sys_overload_raw, cfg.sys_min_consecutive)

    # Tenant shares p_i(t)
    denom = np.where(S > 0, S, 1.0)
    P = X / denom  # (users, hours)

    # Rolling baseline for shares per user (vectorized with pandas rolling via DataFrame)
    P_df = pd.DataFrame(P)
    # pandas 3.x removed rolling(axis=1); do rolling on transpose then transpose back
    muP = (
        P_df.T.rolling(window=cfg.baseline_window_hours, min_periods=cfg.min_baseline_points)
        .mean()
        .T.to_numpy()
    )
    sdP = (
        P_df.T.rolling(window=cfg.baseline_window_hours, min_periods=cfg.min_baseline_points)
        .std(ddof=0)
        .T.to_numpy()
    )
    zP = (P - muP) / (sdP + cfg.eps)
    share_anom = zP > cfg.share_z

    # Growth/slope flags per user
    growth_flags = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        growth_flags[i, :] = compute_growth_flags(X[i, :], cfg.growth_rate_threshold)

    # Aggregate growth hits in a short lookback window (e.g., last 3 hours)
    growth_hits = (
        pd.DataFrame(growth_flags.astype(int))
        .T.rolling(window=cfg.growth_window_hours, min_periods=1)
        .sum()
        .T.to_numpy()
    )
    growth_burst = growth_hits >= cfg.growth_min_hits

    # User absolute anomaly when not overloaded
    user_abs_anom = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        mu, sd = rolling_mean_std(X[i, :], window=cfg.baseline_window_hours, min_periods=cfg.min_baseline_points)
        user_abs_anom[i, :] = X[i, :] > (mu + cfg.user_k * (sd + cfg.eps))

    # Decision logic
    # - During overload: flag tenant if (share_anom AND growth_burst) to focus on "culprit" tenants
    # - Outside overload: flag tenant if (user_abs_anom AND growth_burst) to catch bursts without system-wide effect
    overload_mask = sys_overload[np.newaxis, :]
    flags_overload = overload_mask & share_anom & growth_burst
    flags_normal = (~overload_mask) & user_abs_anom & growth_burst
    flags = flags_overload | flags_normal

    # Per-user summary
    per_user_hits = flags.sum(axis=1)
    anomalous_users = user_ids[per_user_hits > 0].tolist()

    # For reporting: record hours and reason counts
    records = []
    for i, uid in enumerate(user_ids):
        if per_user_hits[i] <= 0:
            continue
        hit_hours = np.where(flags[i, :])[0].tolist()
        overload_hours = [h for h in hit_hours if sys_overload[h]]
        normal_hours = [h for h in hit_hours if not sys_overload[h]]
        records.append(
            {
                "user_id": uid,
                "hit_count": int(per_user_hits[i]),
                "hit_hours": ",".join(map(str, hit_hours[:200])),
                "hits_during_overload": int(len(overload_hours)),
                "hits_outside_overload": int(len(normal_hours)),
                "share_anom_hits": int((share_anom[i, :] & sys_overload).sum()),
                "abs_anom_hits": int((user_abs_anom[i, :] & (~sys_overload)).sum()),
                "growth_hits": int(growth_flags[i, :].sum()),
            }
        )

    return {
        "system_overload_hours": np.where(sys_overload)[0].tolist(),
        "anomalous_users": anomalous_users,
        "records": pd.DataFrame.from_records(records).sort_values(["hit_count", "user_id"], ascending=[False, True]),
    }


def write_excel(path: Path, df: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="rpm")
        truth = df.attrs.get("truth_events")
        if isinstance(truth, pd.DataFrame) and not truth.empty:
            truth.to_excel(writer, index=False, sheet_name="truth")


def read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="rpm")


def main() -> int:
    cfg = Config()

    ap = argparse.ArgumentParser(description="Generate fake multi-tenant RPM excel and detect anomalies.")
    ap.add_argument("--regenerate", action="store_true", help="Force regenerate fake_rpm_data.xlsx")
    ap.add_argument("--users", type=int, default=cfg.users, help="Number of users/tenants (rows)")
    ap.add_argument("--seed", type=int, default=cfg.seed, help="Random seed")
    ap.add_argument("--growth_threshold", type=float, default=cfg.growth_rate_threshold, help="Hourly growth rate threshold")
    args = ap.parse_args()

    cfg = Config(users=args.users, seed=args.seed, growth_rate_threshold=args.growth_threshold)

    base_dir = Path(__file__).resolve().parent
    result_dir = base_dir / "result"
    result_dir.mkdir(exist_ok=True)
    xlsx_path = result_dir / cfg.output_xlsx
    results_csv = result_dir / cfg.output_results_csv

    if args.regenerate or (not xlsx_path.exists()):
        df_fake, injected_anomaly_users = make_fake_data(cfg)
        write_excel(xlsx_path, df_fake)
        print(f"[OK] generated: {xlsx_path}")
        print(f"[INFO] injected anomaly users: {', '.join(injected_anomaly_users)}")
    else:
        print(f"[OK] exists, skip generate: {xlsx_path}")

    df = read_excel(xlsx_path)
    res = detect(cfg, df)

    overload_hours = res["system_overload_hours"]
    print(f"[RESULT] system_overload_hours: {overload_hours[:50]}{'...' if len(overload_hours) > 50 else ''}")
    print(f"[RESULT] anomalous_users ({len(res['anomalous_users'])}): {', '.join(res['anomalous_users'][:50])}{'...' if len(res['anomalous_users']) > 50 else ''}")

    res_df: pd.DataFrame = res["records"]
    res_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote results: {results_csv}")
    if not res_df.empty:
        print("[TOP] first 10 anomalies:")
        print(res_df.head(10).to_string(index=False))
    else:
        print("[TOP] no anomalies detected (check thresholds).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

