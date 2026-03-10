## 本地 Web：多租户 RPM 异常检测（Flask + Plotly）

### 1. 安装依赖

在 PowerShell 里执行：

```powershell
cd "c:\Users\10157\Desktop\xds"
python -m pip install -r requirements.txt
```

### 2. 启动网页（localhost）

```powershell
python webapp.py
```

浏览器访问：

- `http://127.0.0.1:5000`

### 3. 使用流程

- **上传文件**：选择本地 `xlsx/csv`。
- **起始时间**：选择 `H0` 对应的时间（网页会把 `H0..H167` 映射到真实 datetime 横轴）。
- 点击 **开始检测**，进入结果页：
  - 系统总量时序（若判定系统过载，会用浅红色区间标注）
  - Top 异常用户表（含 avg/p95/max、reason、命中小时）
  - 点“查看详情”进入单用户页：曲线 + 异常点（红点）+ tooltip 证据
  - 右上角可导出 CSV/JSON

### 4. 输入文件格式要求

支持两种：

- **Excel**：默认读取 sheet 名 `rpm`
- **CSV**：同列名

必需列：

- `user_id`
- `H0, H1, ...`（至少 24 个小时列；推荐 168 列表示 7 天）

可选列：

- `Total`：若缺失会自动计算（行内小时列求和）

### 5. “异常原因/证据”字段说明

网页中的 `reason`：

- `overload_share_and_growth`：系统过载时，占比 z-score 异常 + 增长斜率突发
- `abs_and_growth`：非过载时，绝对值异常 + 增长斜率突发

tooltip 中会显示（命中点处）：

- `growth_rate`：小时环比增长率 \(g(t)=x(t)/(x(t-1)+1)-1\)
- `abs_z`：该用户相对自身滑窗基线的 z 值
- `share_z`：该用户占系统总量的占比 z 值（系统过载时更关键）

### 6. 常见错误与处理

- **缺少 `user_id`**：请确保第一列或某列名为 `user_id`。
- **找不到 `H0..Hn`**：请确保小时列名形如 `H0`、`H1`…（大小写敏感）。
- **起始时间为空/格式不对**：请用页面的时间选择控件填写。

