from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "result"


def load_all_csv() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"data 目录下未找到任何 csv 文件：{DATA_DIR}")

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df["__source_file"] = f.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def parse_collect_time_std(df: pd.DataFrame) -> pd.DataFrame:
    if "collect_time_std" not in df.columns:
        raise ValueError("缺少列 collect_time_std，无法按小时聚合。")

    df = df.copy()
    df["collect_time_std"] = pd.to_datetime(df["collect_time_std"], errors="coerce")
    if df["collect_time_std"].isna().all():
        raise ValueError("collect_time_std 列解析失败，请检查时间格式。")

    # pandas 2.2+ 推荐使用小写 'h'
    df["collect_hour"] = df["collect_time_std"].dt.floor("h")
    return df


def ensure_dirs() -> None:
    RESULT_DIR.mkdir(exist_ok=True)


def build_hour_range(series: pd.Series) -> pd.DatetimeIndex:
    start = series.min()
    end = series.max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("collect_hour 为空，无法构造时间范围。")
    # pandas 2.2+ 推荐使用小写 'h'
    return pd.date_range(start=start, end=end, freq="h")


def hour_label(dt: pd.Timestamp) -> str:
    # 示例：2026/1/21 0:00:00
    return f"{dt.year}/{dt.month}/{dt.day} {dt.hour}:00:00"


def aggregate_by_pool_and_user(df: pd.DataFrame, metric: str = "rpm") -> None:
    ensure_dirs()

    # 以 infer_service_id 作为“池子”标识
    if "infer_service_id" not in df.columns:
        raise ValueError("缺少列 infer_service_id，无法按池子拆分。")

    if "domain_id" not in df.columns:
        raise ValueError("缺少列 domain_id，无法按用户聚合。")

    if metric not in df.columns:
        raise ValueError(f"缺少待聚合指标列：{metric}")

    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)

    writer_path = RESULT_DIR / "pool_hourly_summary.xlsx"

    with pd.ExcelWriter(writer_path, engine="openpyxl") as writer:
        wrote_any = False
        # 每个 infer_service_id 一个 sheet
        for pool, g in df.groupby("infer_service_id"):
            # 只保留有合法 collect_hour 的数据
            g = g[g["collect_hour"].notna()].copy()
            if g.empty:
                continue

            hour_index = build_hour_range(g["collect_hour"])

            # groupby(pool_id, domain_id, hour) 求和
            grouped = (
                g.groupby(["infer_service_id", "domain_id", "collect_hour"])[metric]
                .sum()
                .reset_index()
            )

            # 透视成：每行一个用户，每列一个小时
            pivot = grouped.pivot_table(
                index="domain_id",
                columns="collect_hour",
                values=metric,
                aggfunc="sum",
                fill_value=0.0,
            )

            # 补全全量小时范围，缺的填 0
            pivot = pivot.reindex(columns=hour_index, fill_value=0.0)

            # 将列名转成指定字符串格式
            pivot.columns = [hour_label(c) for c in pivot.columns]

            # 行索引恢复为普通列
            pivot.reset_index(inplace=True)

            sheet_name = str(pool)[:31] or "unknown_pool"
            pivot.to_excel(writer, sheet_name=sheet_name, index=False)
            wrote_any = True

        if not wrote_any:
            # 至少写一个空的占位 sheet，避免 openpyxl 报错
            empty = pd.DataFrame({"info": ["no data"]})
            empty.to_excel(writer, sheet_name="empty", index=False)


def main() -> None:
    df = load_all_csv()
    df = parse_collect_time_std(df)
    aggregate_by_pool_and_user(df, metric="rpm")


if __name__ == "__main__":
    main()

