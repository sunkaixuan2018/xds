from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
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

MIN_GROUPS_FOR_PROGRESS = 1000
PROGRESS_STEPS = 10
CHECKPOINT_SHEETS = 10


def resolve_workers(workers: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if workers < 1:
        return min(task_count, 32, (os.cpu_count() or 1) + 4)
    return min(task_count, workers)


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


def aggregate_one_sheet(df: pd.DataFrame, bucket_freq: str, progress_label: str = "") -> pd.DataFrame:
    out = parse_time_column(df)
    out["domain_id"] = out["domain_id"].fillna("").astype(str).str.strip()
    out = out[out["domain_id"] != ""].copy()

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["bucket_time"] = out["collect_time_std_parsed"].dt.floor(bucket_freq)

    grouped_rows: list[dict[str, object]] = []
    group_cols = ["domain_id", "bucket_time"]
    grouped = list(out.groupby(group_cols, sort=True))
    total_groups = len(grouped)
    label = f" {progress_label}" if progress_label else ""
    print(f"[progress] sheet{label} groups={total_groups} bucket={bucket_freq}")
    show_group_progress = total_groups >= MIN_GROUPS_FOR_PROGRESS
    progress_interval = max(1, total_groups // PROGRESS_STEPS) if show_group_progress else 0
    for idx, ((domain_id, bucket_time), g) in enumerate(grouped, start=1):
        if show_group_progress and (idx % progress_interval == 0 or idx == total_groups):
            percent = int(idx * 100 / total_groups)
            print(f"[progress] sheet{label} groups {idx}/{total_groups} ({percent}%)")
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


def iter_workbook_batches(input_path: Path, batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    print(f"[progress] opening workbook {input_path}")
    xl = pd.ExcelFile(input_path)
    total_sheets = len(xl.sheet_names)
    print(f"[progress] workbook sheets={total_sheets}")

    batch: list[tuple[int, str, pd.DataFrame]] = []
    for idx, sheet_name in enumerate(xl.sheet_names, start=1):
        print(f"[progress] reading sheet {idx}/{total_sheets}: {sheet_name}")
        frame = pd.read_excel(xl, sheet_name=sheet_name)
        batch.append((idx, sheet_name, frame))
        if len(batch) >= batch_size:
            yield total_sheets, batch
            batch = []

    if batch:
        yield total_sheets, batch


def write_sheet_batch(
    writer: pd.ExcelWriter,
    frames: list[tuple[int, str, pd.DataFrame]],
    total_sheets: int,
    checkpoint_path: Path,
    sheets_written: int,
) -> int:
    for idx, sheet_name, frame in frames:
        print(f"[progress] writing sheet {idx}/{total_sheets}: {sheet_name} rows={len(frame)}")
        frame.to_excel(writer, sheet_name=sheet_name, index=False)
        sheets_written += 1

    writer.book.save(checkpoint_path)
    print(f"[progress] checkpoint saved sheets={sheets_written}/{total_sheets} path={checkpoint_path}")
    return sheets_written


def aggregate_sheet_task(
    idx: int,
    total: int,
    sheet_name: str,
    frame: pd.DataFrame,
    bucket_freq: str,
) -> tuple[int, str, pd.DataFrame]:
    print(f"[progress] sheet {idx}/{total} start rows={len(frame)} name={sheet_name}")
    aggregated = aggregate_one_sheet(frame, bucket_freq, f"{idx}/{total}")
    print(f"[progress] sheet {idx}/{total} done output_rows={len(aggregated)}")
    return idx, sheet_name, aggregated


def aggregate_frames(
    source_frames: list[tuple[int, str, pd.DataFrame]],
    total_sheets: int,
    bucket_freq: str,
    workers: int,
) -> list[tuple[int, str, pd.DataFrame]]:
    max_workers = resolve_workers(workers, len(source_frames))
    first_idx = source_frames[0][0]
    last_idx = source_frames[-1][0]
    print(f"[progress] aggregating sheet batch {first_idx}-{last_idx}/{total_sheets} workers={max_workers}")

    if max_workers == 1:
        frames: list[tuple[int, str, pd.DataFrame]] = []
        for idx, sheet_name, frame in source_frames:
            _, task_sheet_name, aggregated = aggregate_sheet_task(
                idx,
                total_sheets,
                sheet_name,
                frame,
                bucket_freq,
            )
            frames.append((idx, task_sheet_name, aggregated))
        return frames

    results: list[tuple[int, str, pd.DataFrame]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                aggregate_sheet_task,
                idx,
                total_sheets,
                sheet_name,
                frame,
                bucket_freq,
            )
            for idx, sheet_name, frame in source_frames
        ]
        for future in as_completed(futures):
            idx, sheet_name, aggregated = future.result()
            results.append((idx, sheet_name, aggregated))

    return sorted(results, key=lambda item: item[0])


def aggregate_workbook(input_path: Path, output_path: Path, granularity: str, workers: int = 0) -> Path:
    bucket_freq = parse_bucket_freq(granularity)
    print(f"[progress] start aggregation input={input_path} output={output_path} bucket={bucket_freq}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()

    sheets_written = 0
    print(f"[progress] writing checkpoints every {CHECKPOINT_SHEETS} sheets")
    with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
        for total_sheets, source_frames in iter_workbook_batches(input_path, CHECKPOINT_SHEETS):
            out_frames = aggregate_frames(source_frames, total_sheets, bucket_freq, workers)
            sheets_written = write_sheet_batch(writer, out_frames, total_sheets, temp_path, sheets_written)

    temp_path.replace(output_path)
    print(f"[ok] wrote {output_path} sheets={sheets_written}")
    return output_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate processed metrics workbook by configurable time buckets.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input xlsx path.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output xlsx path.")
    ap.add_argument(
        "--granularity",
        default="1h",
        help="Bucket size: 1h or any positive minute bucket, for example 1min, 5min, 10min, 30min.",
    )
    ap.add_argument("--workers", type=int, default=0, help="Thread workers. Use 0 for auto, 1 to disable threading.")
    args = ap.parse_args()

    aggregate_workbook(args.input, args.output, args.granularity, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
