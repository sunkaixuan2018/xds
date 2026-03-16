import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def build_time_range(
    start: datetime,
    hours: int,
) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(hours)]


def make_fake_csv(file_idx: int, start: datetime, hours: int, pool_id: str, domain_prefix: str) -> None:
    times = build_time_range(start, hours)
    rows = []

    # 两个用户，部分小时缺失，用于测试按小时补 0 的逻辑
    users = [
        {"domain_id": f"{domain_prefix}-A", "service_name": "DeepSeek-V3"},
        {"domain_id": f"{domain_prefix}-B", "service_name": "DeepSeek-V3"},
    ]

    rng = np.random.default_rng(42 + file_idx)

    for u in users:
        for t in times:
            # 模拟“有缺口”的情况：第二个用户少报几个小时
            if u["domain_id"].endswith("B") and (t.hour % 3 == 0):
                continue

            rpm = float(rng.integers(50, 200))
            tpm = float(rng.integers(1000, 5000))
            req_count = int(rng.integers(10, 80))

            row = {
                "id": f"{file_idx}-{u['domain_id']}-{t:%H%M}",
                "service_name": u["service_name"],
                "domain_id": u["domain_id"],
                "collect_time": "",
                "card_model": "",
                "rpm": rpm,
                "tpm": tpm,
                "req_count_4xx": 0,
                "req_count_5xx": 0,
                "req_error_rate": 0,
                "total_tokens": tpm,
                "prompt_tokens": tpm * 0.6,
                "completion_tokens": tpm * 0.4,
                "req_count": req_count,
                "req_count_2xx": req_count,
                "req_count_error": 0,
                "prompt_tokens_avg": 0,
                "prompt_tokens_p50": 0,
                "prompt_tokens_p80": 0,
                "prompt_tokens_p90": 0,
                "prompt_tokens_p99": 0,
                "prompt_tokens_max": 0,
                "completion_tokens_avg": 0,
                "completion_tokens_p50": 0,
                "completion_tokens_p80": 0,
                "completion_tokens_p90": 0,
                "completion_tokens_p99": 0,
                "completion_tokens_max": 0,
                # 关键时间字段，示例格式：2026/1/21 0:05
                "collect_time_std": t.strftime("%Y/%-m/%-d %H:%M") if os.name != "nt" else f"{t.year}/{t.month}/{t.day} {t.hour}:{t.minute:02d}",
                "data_time": t.replace(minute=0, second=0, microsecond=0).strftime("%Y/%m/%d %H:%M:%S"),
                "data_date": t.strftime("%Y/%m/%d"),
                "prompt_tpm": tpm * 0.6,
                "completion_tpm": tpm * 0.4,
                "infer_service_id": "svc-demo",
                "pool_id": pool_id,
                "region": "cn-shanghai",
                "instance_count": 3,
                "infer_engine": "fabric",
                "prompt_tps": rpm * 0.6,
                "completion_tps": rpm * 0.4,
                "total_tps": rpm,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    file_path = DATA_DIR / f"fake_pool_{pool_id}_{file_idx}.csv"
    df.to_csv(file_path, index=False, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    base_start = datetime(2026, 1, 21, 0, 5, 0)
    hours = 8

    make_fake_csv(1, base_start, hours, pool_id="pool-A", domain_prefix="cust1")
    make_fake_csv(2, base_start, hours, pool_id="pool-B", domain_prefix="cust2")


if __name__ == "__main__":
    main()

