from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "result"

DEFAULT_OUTPUT_RPM = "pool_hourly_summary_rpm.xlsx"
DEFAULT_OUTPUT_TPM = "pool_hourly_summary_tpm.xlsx"
# 兼容旧版单文件名
LEGACY_OUTPUT = "pool_hourly_summary.xlsx"

# 与网页约定：sheet = {pool_key}__{service_tag}，全部合计用 ALL
SHEET_SEP = "__"
ALL_TAG = "ALL"


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


def normalize_pool(x: str) -> str:
    if not isinstance(x, str):
        x = str(x)
    parts = x.split("-")
    if len(parts) >= 4:
        return "-".join(parts[:4])
    return x


_INVALID_SHEET_CHARS = re.compile(r"[\[\]\:\*\?\/\\]")

# Excel 工作表名上限 31；openpyxl 若收到更长字符串会截断，导致 writer.sheets[原名] KeyError，故必须自控长度。
EXCEL_SHEET_MAX = 31


def sanitize_excel_sheet_name(name: str, max_len: int = EXCEL_SHEET_MAX) -> str:
    s = _INVALID_SHEET_CHARS.sub("_", str(name).strip())
    s = s.strip("'") or "sheet"
    if max_len < 1:
        max_len = 1
    return s[:max_len]


def build_sheet_name(pool_key: str, service_tag: str, used: set[str]) -> str:
    """
    生成唯一、长度 <= EXCEL_SHEET_MAX 的 sheet 名：优先可读「池子前缀__服务后缀」；
    过长时截断两侧；仍冲突时用短 hash 区分。
    """
    sep = SHEET_SEP
    sep_len = len(sep)

    def candidate_readable() -> str:
        # 左侧池子、右侧 service，总宽严格不超过 31
        p_budget = 14
        s_budget = EXCEL_SHEET_MAX - sep_len - p_budget
        if s_budget < 4:
            p_budget = EXCEL_SHEET_MAX - sep_len - 8
            s_budget = 8
        p = sanitize_excel_sheet_name(pool_key, p_budget)
        s = sanitize_excel_sheet_name(service_tag, s_budget)
        raw = f"{p}{sep}{s}"
        return sanitize_excel_sheet_name(raw, EXCEL_SHEET_MAX)

    def candidate_hashed(salt: bytes = b"") -> str:
        digest = hashlib.sha256(pool_key.encode("utf-8") + b"\0" + service_tag.encode("utf-8") + salt).hexdigest()[:8]
        p = sanitize_excel_sheet_name(pool_key, 10)
        # 形如 b91b1526-55__a1b2c3d4，总长 <= 31
        raw = f"{p}{sep}h{digest}"
        return sanitize_excel_sheet_name(raw, EXCEL_SHEET_MAX)

    name = candidate_readable()
    if name not in used:
        used.add(name)
        return name

    for salt in (b"", b"x", b"y", b"z"):
        cand = candidate_hashed(salt)
        if cand not in used:
            used.add(cand)
            return cand

    for i in range(2, 10000):
        suffix = f"_{i}"
        base = candidate_readable()
        cand = sanitize_excel_sheet_name(base[: EXCEL_SHEET_MAX - len(suffix)] + suffix, EXCEL_SHEET_MAX)
        if cand not in used:
            used.add(cand)
            return cand
    raise RuntimeError("无法生成唯一 sheet 名")


def _apply_domain_id_text_format_after_write(writer: pd.ExcelWriter, row_count: int) -> None:
    """to_excel 之后用工作簿实际 sheet 名取表（openpyxl 可能对标题再截断/规范化，勿用传入字符串反查）。"""
    wb = writer.book
    title = wb.sheetnames[-1]
    ws = wb[title]
    for row in range(2, row_count + 2):
        ws.cell(row=row, column=1).number_format = "@"


def _pivot_pool_users(
    g: pd.DataFrame,
    hour_index: pd.DatetimeIndex,
    metric: str,
) -> pd.DataFrame:
    grouped = (
        g.groupby(["__pool_group", "domain_id", "collect_hour"])[metric]
        .sum()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index="domain_id",
        columns="collect_hour",
        values=metric,
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot = pivot.reindex(columns=hour_index, fill_value=0.0)
    pivot.columns = [hour_label(c) for c in pivot.columns]
    pivot.reset_index(inplace=True)
    pivot["domain_id"] = pivot["domain_id"].astype(str)
    return pivot


def aggregate_by_pool_and_user(
    df: pd.DataFrame,
    metric: str = "rpm",
    output_path: Path | None = None,
) -> Path:
    ensure_dirs()

    if "infer_service_id" not in df.columns:
        raise ValueError("缺少列 infer_service_id，无法按池子拆分。")

    if "domain_id" not in df.columns:
        raise ValueError("缺少列 domain_id，无法按用户聚合。")

    if metric not in df.columns:
        raise ValueError(f"缺少待聚合指标列：{metric}")

    if output_path is None:
        output_path = RESULT_DIR / (DEFAULT_OUTPUT_RPM if metric == "rpm" else DEFAULT_OUTPUT_TPM)

    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)

    if "service_name" in df.columns:
        df["__service_name"] = df["service_name"].fillna("").astype(str).str.strip()
        df.loc[df["__service_name"] == "", "__service_name"] = "_default"
    else:
        df["__service_name"] = "_default"
        print("[warn] 缺少列 service_name，将全部归为 _default，并仅生成与「全部」等价的单分面。")

    df["__pool_group"] = df["infer_service_id"].astype(str).map(normalize_pool)

    used_sheet_names: set[str] = set()
    wrote_any = False

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        grouped_pools: list[tuple[str, pd.DataFrame, int]] = []
        for pool, g in df.groupby("__pool_group"):
            g2 = g[g["collect_hour"].notna()]
            if g2.empty:
                continue
            user_cnt = int(g2["domain_id"].nunique())
            grouped_pools.append((str(pool), g2.copy(), user_cnt))

        grouped_pools.sort(key=lambda x: x[2], reverse=True)

        for pool, g, user_cnt in grouped_pools:
            print(f"[pool] 处理池子: {pool}，行数: {len(g)}，用户数(domain_id): {user_cnt}")

            hour_index = build_hour_range(g["collect_hour"])
            services = sorted(g["__service_name"].dropna().unique().tolist())

            # 1) 各 service_name 分面
            for svc in services:
                g_svc = g[g["__service_name"] == svc]
                if g_svc.empty:
                    continue
                pivot = _pivot_pool_users(g_svc, hour_index, metric)
                sheet = build_sheet_name(pool, svc, used_sheet_names)
                pivot.to_excel(writer, sheet_name=sheet, index=False)
                _apply_domain_id_text_format_after_write(writer, len(pivot))
                wrote_any = True
                actual = writer.book.sheetnames[-1]
                print(f"       [sheet] {actual}（请求名={sheet}，service={svc}），行数: {len(pivot)}")

            # 2) 多个 service 时额外写「全部」合计（同用户同小时跨 service 求和）；仅 1 个 service 时不重复写与分面相同的数据
            if len(services) > 1:
                pivot_all = _pivot_pool_users(g, hour_index, metric)
                sheet_all = build_sheet_name(pool, ALL_TAG, used_sheet_names)
                pivot_all.to_excel(writer, sheet_name=sheet_all, index=False)
                _apply_domain_id_text_format_after_write(writer, len(pivot_all))
                wrote_any = True
                actual_all = writer.book.sheetnames[-1]
                print(f"       [sheet] {actual_all}（请求名={sheet_all}，全部合计），行数: {len(pivot_all)}")

        if not wrote_any:
            empty = pd.DataFrame({"info": ["no data"]})
            empty.to_excel(writer, sheet_name="empty", index=False)

    print(f"[ok] 已写入: {output_path}")
    return output_path


def main() -> None:
    ap = argparse.ArgumentParser(description="按池子、用户、service_name 聚合小时级 rpm/tpm，输出到 result/。")
    ap.add_argument(
        "--metrics",
        choices=("rpm", "tpm", "both"),
        default="rpm",
        help="输出指标：rpm、tpm 或两者各一个 xlsx（默认 rpm）。",
    )
    ap.add_argument(
        "--legacy-name",
        action="store_true",
        help=f"在仅生成 rpm 时额外写一份兼容旧名 {LEGACY_OUTPUT}。",
    )
    args = ap.parse_args()

    df = load_all_csv()
    df = parse_collect_time_std(df)

    metrics: list[str]
    if args.metrics == "both":
        metrics = ["rpm", "tpm"]
    else:
        metrics = [args.metrics]

    for m in metrics:
        out = RESULT_DIR / (DEFAULT_OUTPUT_RPM if m == "rpm" else DEFAULT_OUTPUT_TPM)
        aggregate_by_pool_and_user(df, metric=m, output_path=out)

    if args.legacy_name and "rpm" in metrics:
        legacy = RESULT_DIR / LEGACY_OUTPUT
        aggregate_by_pool_and_user(df, metric="rpm", output_path=legacy)
        print(f"[ok] 兼容副本: {legacy}")


if __name__ == "__main__":
    main()
