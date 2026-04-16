# 本地 Web：池子时延异常检测（Flask + Plotly）

当前版本的主流程是：

1. 从原始指标 CSV 生成多 sheet Excel
2. 按时间粒度聚合
3. 在网页中上传聚合后的 Excel，选择池子并分析 TTFT / TPOT 异常

## 1. 安装依赖

在 PowerShell 中执行：

```powershell
cd "C:\Users\10157\Desktop\xds"
python -m pip install -r requirements.txt
```

## 2. 准备输入数据

如果你手里已经有符合要求的聚合结果 Excel，可以直接跳到第 3 步。

如果你手里是原始 CSV，默认处理链路如下：

```powershell
python code/process_data2_to_excel.py
python code/aggregate_processed_metrics.py --granularity 1h
```

默认输出文件：

- 原始 CSV 分组结果：`result/new_data_processed.xlsx`
- 按时间粒度聚合结果：`result/new_data_aggregated.xlsx`

如果需要本地造一份演示数据，可以先执行：

```powershell
python code/generate_fake_data2.py
```

再运行上面的两个处理脚本。

## 3. 启动网页

```powershell
python code/webapp.py
```

浏览器访问：

- `http://127.0.0.1:5000`

## 4. 网页使用流程

1. 上传聚合后的 Excel 文件
2. 选择要分析的 sheet（按池子 / 服务分组展示）
3. 选择灵敏度和保留事件数
4. 进入结果页查看系统异常、事件根因和用户明细

结果页包含：

- 系统 TTFT / TPOT 与 RPM / TPM 时序图
- 异常事件列表
- 根因用户列表
- 单用户详情页
- 事件详情页
- CSV / JSON 导出

## 5. Excel 输入格式要求

上传到网页的文件应为单个 Excel，包含一个或多个 sheet。

每个 sheet 至少需要这些列：

- `domain_id`
- `rpm`
- `tpm`
- `ttft_avg`
- `tpot_avg`
- `prompt_tokens`
- `completion_tokens`
- `collect_time_std`

说明：

- `collect_time_std` 会在后端自动解析并补齐时间轴
- 同一 `domain_id + collect_time_std` 的重复记录会先合并
- 系统层指标会按 `rpm` 加权计算 TTFT / TPOT / token 长度均值

## 6. 当前保留的核心脚本

- `code/webapp.py`：Web 入口
- `code/latency_detector.py`：延迟异常检测与根因定位
- `code/process_data2_to_excel.py`：把 `data2/*.csv` 汇总为多 sheet Excel
- `code/aggregate_processed_metrics.py`：按 `1h / 30min / 10min` 聚合
