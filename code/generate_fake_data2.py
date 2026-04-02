from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA2_DIR = BASE_DIR / "data2"


def ensure_dirs() -> None:
    DATA2_DIR.mkdir(exist_ok=True)


def build_time_range(start: datetime, hours: int) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(hours)]


def format_collect_time(t: datetime, mode: int) -> str:
    if mode == 0:
        return f"{t.year}/{t.month}/{t.day}  {t.hour}:{t.minute:02d}:{t.second:02d}"
    if mode == 1:
        return f"{t.year}/{t.month}/{t.day} {t.hour}:{t.minute:02d}:{t.second:02d}"
    if mode == 2:
        return f"{t.year}/{t.month}/{t.day} {t.hour}:{t.minute:02d}"
    return t.strftime("%Y-%m-%d %H:%M:%S")


def make_fake_csv(file_idx: int, start: datetime, hours: int, infer_service_id: str, service_name: str, pool_id: str) -> None:
    times = build_time_range(start, hours)
    rng = np.random.default_rng(20260402 + file_idx)
    rows: list[dict[str, object]] = []

    domains = [
        f"{pool_id}-tenant-A",
        f"{pool_id}-tenant-B",
        f"{pool_id}-tenant-C",
    ]

    for domain_idx, domain_id in enumerate(domains):
        for hour_idx, t in enumerate(times):
            if domain_idx == 1 and hour_idx in {2, 6}:
                continue

            rpm = float(rng.integers(30, 180))
            tpm = float(rng.integers(800, 5500))
            ttft_avg = float(rng.uniform(180, 950))
            tpot_avg = float(rng.uniform(25, 180))
            prompt_tokens = float(rng.integers(400, 5000))
            completion_tokens = float(rng.integers(200, 3500))
            req_count = int(rng.integers(20, 120))
            req_4xx = int(rng.integers(0, 3))
            req_5xx = int(rng.integers(0, 2))
            req_error = req_4xx + req_5xx
            total_tokens = prompt_tokens + completion_tokens

            row = {
                "service_name": service_name,
                "domain_id": domain_id,
                "card_model": "H100",
                "rpm": rpm,
                "tpm": tpm,
                "ttft_avg": ttft_avg,
                "tpot_avg": tpot_avg,
                "req_count_4xx": req_4xx,
                "req_count_5xx": req_5xx,
                "req_error_rate": float(req_error / max(req_count, 1)),
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_avg": float(ttft_avg + tpot_avg * 10),
                "req_count": req_count,
                "req_count_2xx": req_count - req_error,
                "req_count_error": req_error,
                "prompt_tokens_avg": float(prompt_tokens / max(req_count, 1)),
                "prompt_tokens_p50": float(prompt_tokens * 0.65),
                "prompt_tokens_p80": float(prompt_tokens * 0.82),
                "prompt_tokens_p90": float(prompt_tokens * 0.90),
                "prompt_tokens_p99": float(prompt_tokens * 0.99),
                "prompt_tokens_max": float(prompt_tokens * 1.10),
                "completion_tokens_avg": float(completion_tokens / max(req_count, 1)),
                "completion_tokens_p50": float(completion_tokens * 0.62),
                "completion_tokens_p80": float(completion_tokens * 0.81),
                "completion_tokens_p90": float(completion_tokens * 0.90),
                "completion_tokens_p99": float(completion_tokens * 0.99),
                "completion_tokens_max": float(completion_tokens * 1.12),
                "ttft_p50": float(ttft_avg * 0.85),
                "ttft_p80": float(ttft_avg * 1.05),
                "ttft_p90": float(ttft_avg * 1.15),
                "ttft_p99": float(ttft_avg * 1.35),
                "ttft_max": float(ttft_avg * 1.55),
                "tpot_p50": float(tpot_avg * 0.82),
                "tpot_p80": float(tpot_avg * 1.05),
                "tpot_p90": float(tpot_avg * 1.15),
                "tpot_p99": float(tpot_avg * 1.30),
                "tpot_max": float(tpot_avg * 1.48),
                "collect_time_std": format_collect_time(t, hour_idx % 4),
                "prompt_tpm": float(prompt_tokens / 60.0),
                "completion_tpm": float(completion_tokens / 60.0),
                "infer_service_id": infer_service_id,
                "pool_id": pool_id,
                "region": "cn-shanghai",
                "instance_count": int(rng.integers(2, 6)),
                "infer_engine": "demo-engine",
                "prompt_tps": float(prompt_tokens / 3600.0),
                "completion_tps": float(completion_tokens / 3600.0),
                "total_tps": float(total_tokens / 3600.0),
                "service_type": "chat",
                "stream": "true" if hour_idx % 2 == 0 else "false",
                "sub_models": "demo-submodel",
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = DATA2_DIR / f"fake_data2_{file_idx}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[ok] wrote {out_path} rows={len(df)}")


def main() -> None:
    ensure_dirs()

    start = datetime(2026, 3, 14, 4, 0, 0)
    hours = 10

    make_fake_csv(
        file_idx=1,
        start=start,
        hours=hours,
        infer_service_id="svc-inference-main-cluster-east-long-name",
        service_name="DeepSeek-R1-Reasoning-Service-Ultra-Long-Name",
        pool_id="pool-A",
    )
    make_fake_csv(
        file_idx=2,
        start=start + timedelta(hours=1),
        hours=9,
        infer_service_id="svc-inference-main-cluster-east-long-name",
        service_name="DeepSeek-V3-Chat-Service-Ultra-Long-Name",
        pool_id="pool-A",
    )
    make_fake_csv(
        file_idx=3,
        start=start,
        hours=8,
        infer_service_id="svc-vision-prod",
        service_name="Vision-Pro-Max",
        pool_id="pool-B",
    )


if __name__ == "__main__":
    main()
