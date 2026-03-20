## 异常检测逻辑说明（系统事件 + 用户根因）

### 总体思路（两层：系统事件 -> 用户异常）
本项目在每个“池子（pool）”内对多租户用户流量做异常检测，整体分为两步：

1. 系统层先找出“系统过载/异常事件”的小时窗口（system event windows）。
2. 用户层只在这些系统事件小时窗口内进行严格根因打标，判断哪些 `user_id` 才是真正造成系统异常的主贡献者（culprit）。

这种“先系统、再用户”的设计能抑制误报：当系统因为某个客户突然增量过载时，很多正常客户可能会随之出现波动；严格模式下只评估系统事件窗口内的用户异常，从而减少把被动波动误判为根因。

简化流程图：
```text
+------------------------------+
| 用户时序矩阵 X(user, hour)   |
+---------------+--------------+
                |
                v
+------------------------------+
| 系统总量 S(hour) = sum_user |
+---------------+--------------+
                |
                v
+------------------------------+
| 生成系统事件窗口            |
| events + sys_event_mask     |
+---------------+--------------+
                |
                v
+------------------------------+
| 严格模式用户打标            |
| flags(user, hour)           |
+---------------+--------------+
                |
                v
+------------------------------+
| 事件内定位主因 culprits     |
| 输出 reason/evidence        |
+------------------------------+
```

---

### 第一章：数据准备（构造可检测矩阵）
入口函数 `detect_anomalies(cfg, df, h_cols)` 的输入语义是：
- `df`：一个池子内的用户表（每行一个用户）
- `h_cols`：按小时顺序的时间列名
- 在内部会构造 `X = df[h_cols].to_numpy(dtype=float)`，得到矩阵形状 `(users, hours)`

系统总量：
- `S[t] = X[:, t].sum()`，即该小时所有用户的总量。

---

### 第二章：系统层异常事件窗口生成（events）
系统层在 `S[t]` 上生成系统异常点 `sys_anom[t]`，再把点合并成事件窗口 `events=[(start,end), ...]`。

#### 2.1 主门控：ratio 门（过滤日周期正常爬坡）
首先构造“同小时-of-day 基线”的季节中位数：
- `med[t] = seasonal_median(S[t], same hour-of-day history)`
- `ratio[t] = S[t] / med[t]`
- `sys_ratio_anom[t] = ratio[t] >= sys_ratio_threshold (默认 1.10)`

ratio 门含义：要求系统总量相对“同小时历史中位数”至少上升 10%，避免把正常的日常波动当异常。

#### 2.2 两个确认信号：robust z 或 growth burst
在通过 ratio 门的候选时刻上，再做确认，确认信号来源于两条“或”逻辑：

1) robust z（稳健水平异常）：`sys_level_anom[t]`
- 计算 `sys_rz_seasonal[t] = _seasonal_robust_z(S, period=24, lookback=6)`
  - 对每个 `t`，历史只取 `t-24, t-48, ...`（同小时-of-day）
  - 用 `median(hist)` 与 `MAD(hist)` 计算稳健 z-score
- 如果 `sys_rz_seasonal[t]` 不有限，则回退到 rolling 版本：
  - `_robust_z_score(S, window=24, min_periods=8)`
  - 基线用 `shift(1)` 后的 rolling median+MAD（past-only）

最终：
- `sys_level_anom[t] = sys_rz[t] >= sys_robust_z(默认 4.0)`

2) growth burst（增长爆发）：`sys_burst[t]`
- 系统增长率：
  - `sys_growth[t] = S[t]/(S[t-1]+1) - 1`（`_growth_rate`）
- 触发阈值：
  - `sys_burst[t] = sys_growth[t] >= sys_growth_rate_threshold(默认 0.15)`

在实现上：
- `sys_anom_normal = sys_ratio_anom & (sys_level_anom | sys_burst)`

#### 2.3 单点极端异常旁路（避免漏掉尖峰）
即使没有 robust z 或增长确认，只要 ratio 足够极端也认为异常：
- `sys_anom_extreme[t] = ratio[t] >= sys_extreme_ratio(默认 1.50)`

最终系统异常点：
- `sys_anom[t] = sys_anom_normal OR sys_anom_extreme[t]`

#### 2.4 从异常点到事件窗口 events
把 `sys_anom[t]` 的布尔序列用 `_mask_to_events(mask, min_len=2, merge_gap=1)` 转为合并事件窗口 `(a,b)`：
- 连续异常点段形成候选段
- 过滤掉长度 < 2 的段
- 距离很近的段按 `merge_gap` 合并

warm-up（前期不检测）：
- `warmup_hours=48`：把前 48 小时的系统异常清空，避免基线不足导致的异常判断失真

额外一步：保留极端单点
- 即使某些极端单点没有落入连续窗口，也会被强制加入 `events`，避免只捕捉连续段而漏掉孤立尖峰。

裁剪事件数：
- `max_events=2`：对“正常窗口段”按峰值 ratio 进行裁剪，但**始终保留所有极端单点事件**。

最终得到：
- `events`：系统异常事件窗口列表
- `sys_event_mask[t]`：事件覆盖的小时掩码，用于后续严格根因打标。

系统层流程图：
```text
+------------------------------+
| 系统总量 S(hour)             |
+---------------+--------------+
                |
                v
+------------------------------+
| ratio 门：                    |
| S/seasonal_median >= 1.10     |
+---------------+--------------+
      |通过                      |不通过
      v                          v
+------------------------------+  (但若)
| 候选点验证：                  |  ratio >= 1.50 ?
| (robust z >= 4.0) OR          +----> sys_anom_extreme
| (growth burst >= 0.15)        |
+---------------+--------------+
                |
                v
+------------------------------+
| sys_anom_normal = ratio_gate |
| AND (robust_z OR growth_burst)
+---------------+--------------+
                |
                v
+------------------------------+
| sys_anom = sys_anom_normal  |
| OR sys_anom_extreme          |
+---------------+--------------+
                |
                v
+------------------------------+
| mask_to_events(min_len=2,  |
| merge_gap=1)                |
+---------------+--------------+
                |
                v
+------------------------------+
| events + sys_event_mask     |
+------------------------------+
```

---

### 第三章：用户层异常判定（严格根因打标 flags）
用户侧不在系统之外做自由异常检测；默认 `strict_rootcause=True`，因此用户异常只在 `sys_event_mask` 为真的小时评估。

#### 3.1 用户份额异常（share z）
用户份额：
- `P[user,t] = X[user,t] / max(S[t], 1.0)`

用 `P` 的 rolling baseline（shift(1) 后 rolling mean/std）得到 `zP`，判定：
- `share_anom = zP > share_z(默认 3.0)`
- `share_extreme = zP > share_z_extreme(默认 5.0)`

#### 3.2 用户绝对异常（abs）
每个用户单独做 rolling mean/std baseline（past-only）：
- `abs_anom = X[user,t] > mu + user_k*(sd)`，`user_k=2.0`
- 并计算稳健 `abs_z`，用于极端/回填相关阈值：
  - `abs_z_extreme = 5.0`
  - `abs_z_peak = 3.5`
  - `abs_z_episode = 2.5`

#### 3.3 用户增长证据（growth burst）
用户增长率定义与系统一致（只是输入换成用户序列 `X[user,:]`）：
- `growth_flag = growth >= growth_rate_threshold(默认 0.8)`
- `growth_burst`：过去 `growth_window_hours=3` 小时内增长命中次数达到 `growth_min_hits=1`

#### 3.4 严格根因最终判定 flags_overload
严格模式下：
- 系统事件之外 `flags_normal` 全为 False
- 用户异常主要由事件内的过载证据产生：

```text
flags_overload(user,t) =
  sys_event_mask(t) AND
  ( (share_anom AND (growth_burst OR share_extreme)) OR abs_z_extreme )
```

最终用户异常：
- `flags = flags_overload`（严格模式下不加系统外正常路径）

用户侧流程图：
```text
+----------------------------------------------+
| 仅当 sys_event_mask(t)==True 才评估用户异常 |
+------------------------------+---------------+
                               |
       +-----------------------+-----------------------+
       |                       |                       |
       v                       v                       v
+------------------+   +-------------------+   +------------------+
| share 证据        |   | growth 证据       |   | abs 证据          |
| share_anom 或     |   | growth_burst      |   | abs_z_extreme    |
| share_extreme(zP) |   +-------------------+   +------------------+
+---------+--------+              |                       |
          |                        |                       |
          v                        v                       v
+--------------------------------------------------------------+
| flags_overload = (share_anom AND (growth_burst OR share_extreme)) |
|                   OR abs_z_extreme                            |
+------------------------------+--------------------------------+
                               |
                               v
+------------------------------+
| flags(user,t)               |
+------------------------------+
```

---

### 第四章：事件内主因定位、episode 回填与 reason 生成
这一部分主要用于解释性输出（用于网页展示和导出）。

#### 4.1 事件内主因用户 culprits（excess 排序）
对每个事件窗口 `(a,b)`：
- 用每个用户的 baseline mean（`mu_abs`）计算“超额贡献”：
  - `excess = clip(X[user,a:b]-mu_abs[user,a:b], 0, None)`
- 窗口内对用户求和：`excess_sum_user`
- 按 `excess_sum_user` 从大到小选择主因：
  - `culprit_top_k=2`
  - 或累计占比 `>=0.7`（`culprit_cum_ratio`）
  - 并要求每个被选用户至少贡献 `>=0.05` 的份额（`culprit_min_ratio`）

输出每个主因用户的：
- `peak_hour`（窗口内峰值发生时间）
- `peak_rpm`
- `excess_sum / excess_ratio`

#### 4.2 episode 回填：把“episode 内峰值”强制置 True
为了处理“增长发生时序提前但峰值滞后”的错位问题：
- 对每个用户，将以下条件视为 episode：
  - `episode_mask = abs_anom OR (abs_z > abs_z_episode(2.5))`
- 对 episode 的每个连续段，如果该段内原本已有异常命中，则把该段内部 rpm 最大的小时强制设为异常：
  - `flags[user,peak_h] = True`

#### 4.3 reason：用 during_event vs outside_event 给解释标签
对每个用户，统计其命中小时中：
- `during_event`：落在 `sys_event_mask` 内的命中数
- `outside_event`：落在系统事件外的命中数

reason 规则：
- 若 `during_event >= outside_event`
  - 且 during_event > 0：`reason="overload_share_and_growth"`
  - 否则：`reason="abs_and_growth"`
- 否则：
  - `reason="abs_and_growth"`

---

## 你可以用它来回答的关键问题
1. “为什么误报会被抑制？”  
   因为默认严格模式下，用户异常只在 `sys_event_mask` 覆盖的系统异常事件小时里评估。
2. “系统事件窗口是怎么来的？”  
   ratio 门先筛候选，再用 robust z 或 growth burst 确认，最后用极端单点旁路补漏，并合并成窗口。
3. “主因怎么解释出来？”  
   用每个事件窗口内用户相对 baseline 的 `excess` 贡献排序，再做 culprits 选择与 episode 峰值回填。

