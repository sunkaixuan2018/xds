from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = BASE_DIR / "data2"
DEFAULT_OUTPUT_PATH = BASE_DIR / "result" / "new_data_processed.xlsx"

REQUIRED_COLUMNS = [
    "infer_service_id",
    "service_name",
    "domain_id",
    "rpm",
    "tpm",
    "ttft_avg",
    "tpot_avg",
    "prompt_tokens",
    "completion_tokens",
    "collect_time_std",
]

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

_INVALID_SHEET_CHARS = re.compile(r"[\[\]\:\*\?\/\\]")
EXCEL_SHEET_MAX = 31


def sanitize_excel_sheet_name(name: str, max_len: int = EXCEL_SHEET_MAX) -> str:
    s = _INVALID_SHEET_CHARS.sub("_", str(name).strip())
    s = s.strip("'") or "sheet"
    if max_len < 1:
        max_len = 1
    return s[:max_len]


def build_sheet_name(infer_service_id: str, service_name: str, used: set[str]) -> str:
    sep = "__"
    sep_len = len(sep)

    def readable_name() -> str:
        left_budget = 14
        right_budget = EXCEL_SHEET_MAX - sep_len - left_budget
        if right_budget < 6:
            left_budget = 10
            right_budget = EXCEL_SHEET_MAX - sep_len - left_budget
        left = sanitize_excel_sheet_name(infer_service_id, left_budget)
        right = sanitize_excel_sheet_name(service_name, right_budget)
        return sanitize_excel_sheet_name(f"{left}{sep}{right}", EXCEL_SHEET_MAX)

    name = readable_name()
    if name not in used:
        used.add(name)
        return name

    digest = hashlib.sha256(f"{infer_service_id}\0{service_name}".encode("utf-8")).hexdigest()[:8]
    fallback = sanitize_excel_sheet_name(f"{sanitize_excel_sheet_name(infer_service_id, 10)}{sep}h{digest}")
    if fallback not in used:
        used.add(fallback)
        return fallback

    for i in range(2, 10000):
        suffix = f"_{i}"
        candidate = sanitize_excel_sheet_name(name[: EXCEL_SHEET_MAX - len(suffix)] + suffix, EXCEL_SHEET_MAX)
        if candidate not in used:
            used.add(candidate)
            return candidate

    raise RuntimeError("Unable to generate a unique sheet name.")


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
    raise RuntimeError(f"Failed to read csv: {path}") from last_error


def load_all_csv(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv files found under: {input_dir}")

    dfs: list[pd.DataFrame] = []
    for f in files:
        print(f"[load] {f.name}")
        df = _read_csv_flexible(f)
        df["__source_file"] = f.name
        dfs.append(df)
        print(f"       rows={len(df)}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[load] combined rows={len(combined)}")
    return combined


def parse_collect_time_std(df: pd.DataFrame) -> pd.DataFrame:
    if "collect_time_std" not in df.columns:
        raise ValueError("Missing required column: collect_time_std")

    out = df.copy()
    s = out["collect_time_std"].astype(str).str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)

    parsed = pd.to_datetime(s, format="%Y/%m/%d %H:%M:%S", errors="coerce")

    step2_mask = parsed.isna() & s.ne("") & s.notna()
    if step2_mask.any():
        parsed.loc[step2_mask] = pd.to_datetime(s.loc[step2_mask], format="%Y/%m/%d %H:%M", errors="coerce")

    step3_mask = parsed.isna() & s.ne("") & s.notna()
    if step3_mask.any():
        parsed.loc[step3_mask] = pd.to_datetime(s.loc[step3_mask], errors="coerce")

    out["collect_time_std_parsed"] = parsed

    failed = int(out["collect_time_std_parsed"].isna().sum())
    if failed:
        print(f"[time] failed to parse rows={failed}")

    out = out.dropna(subset=["collect_time_std_parsed"]).copy()
    if out.empty:
        raise ValueError("All collect_time_std values failed to parse.")

    out["collect_time_std"] = out["collect_time_std_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_output_frames(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    df = parse_collect_time_std(df)
    validate_columns(df)

    out = df.copy()
    out["infer_service_id"] = out["infer_service_id"].fillna("").astype(str).str.strip()
    out["service_name"] = out["service_name"].fillna("").astype(str).str.strip()
    out["domain_id"] = out["domain_id"].fillna("").astype(str).str.strip()

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out = out[out["infer_service_id"] != ""].copy()
    out = out[out["service_name"] != ""].copy()
    out = out[out["domain_id"] != ""].copy()

    out = out.sort_values(["infer_service_id", "service_name", "domain_id", "collect_time_std_parsed"]).reset_index(drop=True)

    used_sheet_names: set[str] = set()
    frames: list[tuple[str, pd.DataFrame]] = []
    for (infer_service_id, service_name), group in out.groupby(["infer_service_id", "service_name"], sort=True):
        sheet_name = build_sheet_name(infer_service_id, service_name, used_sheet_names)
        sheet_df = group[OUTPUT_COLUMNS].copy()
        sheet_df = sheet_df.sort_values(["domain_id", "collect_time_std"]).reset_index(drop=True)
        frames.append((sheet_name, sheet_df))
        print(f"[sheet] {sheet_name} rows={len(sheet_df)}")

    if not frames:
        raise ValueError("No valid rows remained after filtering.")

    return frames


def write_excel(frames: list[tuple[str, pd.DataFrame]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
        for sheet_name, frame in frames:
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    temp_path.replace(output_path)
    print(f"[ok] wrote {output_path}")
    return output_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter data2 csv files and write grouped excel sheets.")
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing source csv files.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output xlsx path.")
    args = ap.parse_args()

    df = load_all_csv(args.input_dir)
    frames = build_output_frames(df)
    write_excel(frames, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
