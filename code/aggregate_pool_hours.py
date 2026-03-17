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
    # 先尝试显式格式（无秒），再对失败的部分用 pandas 自动推断（支持带秒、额外空格等）
    print("[time] 第一步：按格式 %Y/%m/%d %H:%M 解析 collect_time_std ...")
    s = df["collect_time_std"].astype(str).str.strip()
    t = pd.to_datetime(s, format="%Y/%m/%d %H:%M", errors="coerce")

    # 对还没解析成功的记录，做第二步：不指定 format，让 pandas 自动推断
    mask_step1_fail = t.isna() & s.notna() & (s.str.len() > 0)
    if mask_step1_fail.any():
        print(f"[time] 第一步失败 {mask_step1_fail.sum()} 行，开始第二步自动推断格式（支持带秒、不同空格等）...")
        t2 = pd.to_datetime(s[mask_step1_fail], errors="coerce")
        t.loc[mask_step1_fail] = t2

        # 将自动推断成功的部分按照“格式特征”分组，每类打印一个样例（原始字符串 + 解析后的标准格式）
        success_mask = mask_step1_fail & t.notna()
        if success_mask.any():
            print("[time] 自动推断成功的格式类别示例（每类 1 条，原始 -> 解析后）：")

            def fmt_key(val: str) -> tuple:
                # 按一些简单特征分组：长度、冒号数量、是否包含秒、“T”/空格分隔等
                return (
                    len(val),
                    val.count(":"),
                    "T" in val,
                    " " in val,
                    "/" in val,
                    "-" in val,
                )

            seen_keys = set()
            for raw in s[success_mask]:
                key = fmt_key(raw)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                parsed = pd.to_datetime(raw, errors="coerce")
                if pd.isna(parsed):
                    continue
                print(f"    '{raw}' -> '{parsed.strftime('%Y-%m-%d %H:%M:%S')}'")

    df["collect_time_std"] = t

    # 统计两步之后仍失败的原始字符串，便于后续针对性适配
    fail_mask = df["collect_time_std"].isna() & s.notna() & (s.str.len() > 0)
    if fail_mask.any():
        failed_values = s[fail_mask]
        vc = failed_values.value_counts()
        print(f"[time] 两步解析仍失败的行数: {fail_mask.sum()}，不同格式种类数: {len(vc)}")
        print("[time] 失败格式 Top 20（格式示例 -> 次数）:")
        for val, cnt in vc.head(20).items():
            print(f"    '{val}' -> {cnt}")

    before = len(df)
    df = df.dropna(subset=["collect_time_std"])
    after = len(df)
    if after == 0:
        raise ValueError("collect_time_std 列全部解析失败，请检查时间格式是否为 2026/1/21 0:05 / 2026/1/21 0:05:00 这类。")
    if after < before:
        print(f"[time] 共有 {before - after} 行时间解析失败已被丢弃（两步解析均失败）。")

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
            # 第一列 domain_id 强制为字符串并写入，避免在 Excel 里被当数字导致读回时变成 0.0
            pivot["domain_id"] = pivot["domain_id"].astype(str)

            sheet_name = str(pool)[:31] or "unknown_pool"
            pivot.to_excel(writer, sheet_name=sheet_name, index=False)
            # 将首列设为 Excel 文本格式，避免被 Excel 自动转成数字
            ws = writer.sheets[sheet_name]
            for row in range(2, len(pivot) + 2):
                ws.cell(row=row, column=1).number_format = "@"
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

