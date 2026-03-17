"""生成按池子分析用的假数据 Excel：首列 domain_id，值为「用户360」形式，供本地测试。"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE_DIR / "result"


def main() -> None:
    RESULT_DIR.mkdir(exist_ok=True)
    out_path = RESULT_DIR / "pool_hourly_summary_test.xlsx"

    # 用户 ID：字符串，如 用户360、用户361
    num_users = 12
    domain_ids = [f"用户{360 + i}" for i in range(num_users)]

    # 时间列：至少 24 列，格式与聚合脚本一致
    hours = 24
    start = pd.Timestamp("2026-01-18 00:00:00")
    time_cols = [(start + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(hours)]

    # 假数据：每行一个用户，每列一小时，随机 rpm
    np.random.seed(42)
    data = np.random.randint(80, 200, size=(num_users, hours)).astype(float)
    # 个别点做高一点模拟异常
    data[2, 10:14] += 150
    data[5, 18:20] += 200

    df = pd.DataFrame(data, columns=time_cols)
    df.insert(0, "domain_id", domain_ids)
    # 确保 domain_id 列为字符串
    df["domain_id"] = df["domain_id"].astype(str)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="test_pool", index=False)
        ws = writer.sheets["test_pool"]
        for row in range(2, len(df) + 2):
            ws.cell(row=row, column=1).number_format = "@"

    print(f"已生成: {out_path}")
    print(f"  sheet: test_pool, 用户数: {num_users}, 时间列: {hours}")
    print(f"  domain_id 示例: {domain_ids[:3]} ...")


if __name__ == "__main__":
    main()
