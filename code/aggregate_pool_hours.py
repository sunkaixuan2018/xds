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
        print(f"[load] 读取文件: {f.name}")
        df = pd.read_csv(f, encoding="utf-8")
        print(f"       行数: {len(df)}")
        df["__source_file"] = f.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[load] 合并后总行数: {len(combined)}")
    return combined


def parse_collect_time_std(df: pd.DataFrame) -> pd.DataFrame:
    if "collect_time_std" not in df.columns:
        raise ValueError("缺少列 collect_time_std，无法按小时聚合。")

    df = df.copy()
    print("[time] 按格式 %Y/%m/%d %H:%M 解析 collect_time_std ...")
    df["collect_time_std"] = pd.to_datetime(
        df["collect_time_std"],
        format="%Y/%m/%d %H:%M",
        errors="coerce",
    )

    before = len(df)
    df = df.dropna(subset=["collect_time_std"])
    after = len(df)
    if after == 0:
        raise ValueError("collect_time_std 列全部解析失败，请检查时间格式是否为 2026/1/21 0:05 这类。")
    if after < before:
        print(f"[time] 有 {before - after} 行时间解析失败已被丢弃。")

    # pandas 2.2+ 推荐使用小写 'h'
    df["collect_hour"] = df["collect_time_std"].dt.floor("h")
    print(f"[time] 时间范围: {df['collect_hour'].min()} ~ {df['collect_hour'].max()}")
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
    # 与示例脚本保持一致：2026-01-21 00:00
    return dt.strftime("%Y-%m-%d %H:%M")


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

    # 归一化池子 ID：a-b-c-d-e 只看前四段 a-b-c-d，e 部分允许有轻微变化
    def normalize_pool(x: str) -> str:
        if not isinstance(x, str):
            x = str(x)
        parts = x.split("-")
        if len(parts) >= 4:
            return "-".join(parts[:4])
        return x

    df["__pool_group"] = df["infer_service_id"].astype(str).map(normalize_pool)

    writer_path = RESULT_DIR / "pool_hourly_summary.xlsx"

    with pd.ExcelWriter(writer_path, engine="openpyxl") as writer:
        wrote_any = False
        # 每个归一化后的池子 ID（前四段）一个 sheet
        for pool, g in df.groupby("__pool_group"):
            # 只保留有合法 collect_hour 的数据
            g = g[g["collect_hour"].notna()].copy()
            if g.empty:
                continue

            print(f"[pool] 处理池子: {pool}，行数: {len(g)}，用户数(domain_id): {g['domain_id'].nunique()}")

            hour_index = build_hour_range(g["collect_hour"])

            # groupby(pool_id, domain_id, hour) 求和
            grouped = (
                g.groupby(["__pool_group", "domain_id", "collect_hour"])[metric]
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

