"""测试池子 Excel 能否正确读出用户 ID（用户360 等），而非 0.0。运行前请先执行: python code/gen_fake_pool_data.py"""
import sys
from pathlib import Path

# 从项目根运行 detector 等模块（在项目根执行: python code/test_pool_read.py）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from detector import DetectorConfig, detect_anomalies, validate_and_prepare

# 与 webapp pool_analyze 一致的读表逻辑
RESULT_DIR = Path(__file__).resolve().parents[1] / "result"
TEST_FILE = RESULT_DIR / "pool_hourly_summary_test.xlsx"
SHEET_NAME = "test_pool"


def main() -> None:
    if not TEST_FILE.exists():
        print("请先生成假数据: python code/gen_fake_pool_data.py")
        return
    # 与 webapp 完全一致的读取方式
    df = pd.read_excel(TEST_FILE, sheet_name=SHEET_NAME, dtype=str)
    df.columns = [str(c) for c in df.columns]
    df.iloc[:, 0] = df.iloc[:, 0].fillna("").astype(str)

    df, h_cols = validate_and_prepare(df)
    cfg = DetectorConfig()
    res = detect_anomalies(cfg, df, h_cols)
    user_ids = res["user_ids"]

    expected = [f"用户{360 + i}" for i in range(12)]
    ok = user_ids == expected
    print("读取到的 user_ids:", user_ids)
    print("期望的 user_ids:  ", expected)
    if ok:
        print("\n通过：用户 ID 已正确读取为「用户360」等形式。")
    else:
        print("\n失败：存在 0.0 或与期望不一致。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
