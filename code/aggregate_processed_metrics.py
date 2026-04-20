from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = BASE_DIR / "result" / "new_data_processed.xlsx"
DEFAULT_OUTPUT_PATH = BASE_DIR / "result" / "new_data_aggregated.xlsx"

OUTPUT_COLUMNS = [
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


def parse_bucket_freq(granularity: str) -> str:
    value = str(granularity).strip().lower().replace("_", "").replace(" ", "")
    mapping = {
        "hour": "1h",
        "hours": "1h",
        "1hour": "1h",
        "1hours": "1h",
        "1h": "1h",
        "h": "1h",
        "halfhour": "30min",
        "minute": "1min",
        "minutes": "1min",
        "min": "1min",
    }
    if value in mapping:
        return mapping[value]

    minute_match = re.fullmatch(r"([1-9]\d*)(?:m|min|minute|minutes)", value)
    if minute_match:
        return f"{int(minute_match.group(1))}min"

    raise ValueError(
        f"Unsupported granularity: {granularity}. "
        "Use 1h or a positive minute bucket such as 1min, 5min, 10min, 30min."
    )


def parse_time_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    print(f"[progress] parsing time column rows={len(out)}")
    s = out["collect_time_std"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    parsed = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    fallback_mask = parsed.isna() & s.ne("") & s.notna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(s.loc[fallback_mask], errors="coerce")

    out["collect_time_std_parsed"] = parsed
    out = out.dropna(subset=["collect_time_std_parsed"]).copy()
    if out.empty:
        raise ValueError("All collect_time_std values failed to parse.")
    return out


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    values_num = pd.to_numeric(values, errors="coerce")
    weights_num = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid = values_num.notna() & weights_num.notna() & (weights_num > 0)
    if not valid.any():
        return 0.0
    return float((values_num[valid] * weights_num[valid]).sum() / weights_num[valid].sum())


def weighted_average_ignore_zero(values: pd.Series, weights: pd.Series) -> float:
    values_num = pd.to_numeric(values, errors="coerce")
    weights_num = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid = values_num.notna() & weights_num.notna() & (weights_num > 0) & (values_num != 0)
    if not valid.any():
        return 0.0
    return float((values_num[valid] * weights_num[valid]).sum() / weights_num[valid].sum())


def aggregate_one_sheet(df: pd.DataFrame, bucket_freq: str) -> pd.DataFrame:
    out = parse_time_column(df)
    out["domain_id"] = out["domain_id"].fillna("").astype(str).str.strip()
    out = out[out["domain_id"] != ""].copy()

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["bucket_time"] = out["collect_time_std_parsed"].dt.floor(bucket_freq)

    grouped_rows: list[dict[str, object]] = []
    group_cols = ["domain_id", "bucket_time"]
    grouped = list(out.groupby(group_cols, sort=True))
    print(f"[progress] aggregating groups total={len(grouped)} bucket={bucket_freq}")
    for idx, ((domain_id, bucket_time), g) in enumerate(grouped, start=1):
        if idx == 1 or idx % 500 == 0 or idx == len(grouped):
            print(f"[progress] aggregated group {idx}/{len(grouped)} domain_id={domain_id} bucket_time={bucket_time}")
        rpm_sum = float(g["rpm"].sum())
        tpm_sum = float(g["tpm"].sum())
        grouped_rows.append(
            {
                "domain_id": domain_id,
                "rpm": rpm_sum,
                "tpm": tpm_sum,
                "ttft_avg": weighted_average_ignore_zero(g["ttft_avg"], g["rpm"]),
                "tpot_avg": weighted_average_ignore_zero(g["tpot_avg"], g["rpm"]),
                "prompt_tokens": weighted_average(g["prompt_tokens"], g["rpm"]),
                "completion_tokens": weighted_average(g["completion_tokens"], g["rpm"]),
                "collect_time_std": pd.Timestamp(bucket_time).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    result = pd.DataFrame(grouped_rows, columns=OUTPUT_COLUMNS)
    if result.empty:
        return result

    result = result.sort_values(["domain_id", "collect_time_std"]).reset_index(drop=True)
    return result


def load_workbook_frames(input_path: Path) -> list[tuple[str, pd.DataFrame]]:
    print(f"[progress] opening workbook {input_path}")
    xl = pd.ExcelFile(input_path)
    print(f"[progress] workbook sheets={len(xl.sheet_names)}")
    frames: list[tuple[str, pd.DataFrame]] = []
    for idx, sheet_name in enumerate(xl.sheet_names, start=1):
        print(f"[progress] reading sheet {idx}/{len(xl.sheet_names)}: {sheet_name}")
        frame = pd.read_excel(input_path, sheet_name=sheet_name)
        frames.append((sheet_name, frame))
    return frames


def write_workbook(frames: list[tuple[str, pd.DataFrame]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    print(f"[progress] writing workbook sheets={len(frames)} output={output_path}")
    with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
        for idx, (sheet_name, frame) in enumerate(frames, start=1):
            print(f"[progress] writing sheet {idx}/{len(frames)}: {sheet_name} rows={len(frame)}")
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    temp_path.replace(output_path)
    print(f"[ok] wrote {output_path}")
    return output_path


def aggregate_workbook(input_path: Path, output_path: Path, granularity: str) -> Path:
    bucket_freq = parse_bucket_freq(granularity)
    print(f"[progress] start aggregation input={input_path} output={output_path} bucket={bucket_freq}")
    source_frames = load_workbook_frames(input_path)

    out_frames: list[tuple[str, pd.DataFrame]] = []
    for idx, (sheet_name, frame) in enumerate(source_frames, start=1):
        print(f"[progress] processing sheet {idx}/{len(source_frames)}: {sheet_name} input_rows={len(frame)}")
        aggregated = aggregate_one_sheet(frame, bucket_freq)
        out_frames.append((sheet_name, aggregated))
        print(f"[sheet] {sheet_name} rows={len(aggregated)} granularity={bucket_freq}")

    return write_workbook(out_frames, output_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate processed metrics workbook by configurable time buckets.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input xlsx path.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output xlsx path.")
    ap.add_argument(
        "--granularity",
        default="1h",
        help="Bucket size: 1h or any positive minute bucket, for example 1min, 5min, 10min, 30min.",
    )
    args = ap.parse_args()

    aggregate_workbook(args.input, args.output, args.granularity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
