# 🎮 游戏运营数据分析：《Cookie Cats》门禁位置 A/B 实验

基于**真实线上实验数据**的游戏运营分析项目：Tactile Entertainment 旗下三消手游
*Cookie Cats* 把关卡之间的"门禁"从第 30 关后移到第 40 关，对 **90,189 名随机分流玩家**
做了 14 天观察。项目从数据质量审计（SRM/异常值）出发，完成留存漏斗、参与度分层、
频率派假设检验、bootstrap、贝叶斯后验、功效复盘的完整推断链，最终给出业务决策。
一键可复现：静态报告、已执行的 EDA notebook、Streamlit 交互看板齐备。

> 📊 数据来源：[Kaggle - Mobile Games A/B Testing with Cookie Cats](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)
> （Tactile Entertainment 公开发布的真实实验数据，非模拟）。原始 CSV 随仓库附在
> [`data/raw/cookie_cats.csv`](data/raw/cookie_cats.csv)，带 sha256 校验，
> 也可用 `python -m src.download_data` 重新下载。

---

## 📌 一句话结论

**门禁后移到第 40 关使 7 日留存从 19.02% 显著下降到 18.20%（−0.82pp，p=0.0016），
z 检验 / 1 万次 bootstrap / 贝叶斯后验三套方法结论一致（后验 P≈99.9%），
建议门禁维持在第 30 关**；次日留存方向同为负但不显著（−0.59pp，p=0.074），
游玩局数差异被极端账号支配、不稳健，不能作为决策依据。

## 🧩 分析模块全景

| 模块 | 内容 |
|---|---|
| 数据质量 | schema/缺失/主键重复校验、SRM 分流比例检验（p<0.005 或偏差>1pp 双门槛分级） |
| 大盘画像 | 分组描述统计、留存漏斗（安装→D1→D7）、参与度洛伦兹曲线/Gini、五层玩家分层 |
| 频率派检验 | 两比例 z 检验（合并 SE）、Wald 95% CI、与卡方检验交叉验证、Bonferroni 多重比较 |
| Bootstrap | 10,000 次组内重采样，百分位 CI 与 P(实验组更低)；局数均值差分块重采样 |
| 贝叶斯 | Jeffreys Beta(½,½) 先验 + 20 万次蒙特卡洛，95% 可信区间与后验概率 |
| 重尾指标 | 游玩局数 Mann-Whitney U 秩检验 + rank-biserial 效应量；p99.9 截尾敏感性分析 |
| 功效分析 | 两比例样本量公式、MDE 灵敏度曲线（二分法反解）、"不显著≠无影响"复盘 |
| 分层分析 | 参与度五层 × 两组的留存对比（明确标注处理后变量，只作描述、不作因果） |
| 交付物 | 10 张色盲友好图表、[分析报告](reports/REPORT.md)、[EDA Notebook](notebooks/EDA.ipynb)、Streamlit 看板、9 组单元测试 |

## 🔢 关键数字（全部由流水线计算，可复现）

| 指标 | gate_30 对照 | gate_40 实验 | Δ | 检验 |
|---|---|---|---|---|
| 玩家数 | 44,700 | 45,489 | 分流偏差 0.44pp | SRM p=0.009（预警级，不阻断） |
| 次日留存 | 44.82% | 44.23% | −0.59pp | z p=0.074，不显著 |
| 7 日留存 | 19.02% | 18.20% | **−0.82pp（相对 −4.3%）** | **z p=0.0016，bootstrap/贝叶斯 P=99.9%** |
| 14 天局数中位数 | 两组相同量级 | — | — | MWU p=0.050，r=−0.008（无业务意义） |
| 参与度集中度 | 1% 超核玩家贡献 14.6% 局数，Gini=0.709 | | | 4.4% 玩家安装后 0 局 |

- 当前样本量（4.47 万/组）对 D7 留存的最小可检出效应 MDE≈**0.74pp**（α=5%, power=80%），
  实测 −0.82pp 刚过灵敏度线；若要检出 0.5pp 需约 9.8 万人/组，0.3pp 需 27 万人/组。
- 单个账号 14 天内玩 **49,854 局**（疑似脚本/测试号）：全量数据局数均值差 bootstrap
  给出 80% 负向概率，剔除 91 名 p99.9 以上玩家后回落到 68%——
  这是"重尾指标 + 小效应"场景下必须做敏感性分析的典型案例。

完整推导见 [`reports/REPORT.md`](reports/REPORT.md) 与
[`notebooks/EDA.ipynb`](notebooks/EDA.ipynb)（notebook 已执行，GitHub 直接渲染）。

## 🗂️ 目录结构

```text
game-ops-analytics/
├── main.py                   # 一键流水线：真实数据 → 检验 → 图表 → 报告
├── app.py                    # Streamlit 交互决策看板（含交互式 bootstrap）
├── requirements.txt
├── data/
│   ├── raw/cookie_cats.csv       # 90,189 名玩家真实实验数据（2.7MB，入库）
│   └── processed/                # 分组指标/检验结果/敏感性/MDE 曲线/summary.json
├── notebooks/
│   ├── EDA.ipynb                 # 已执行的探索分析（含输出与图）
│   └── build_notebook.py         # notebook 生成脚本
├── src/
│   ├── config.py                 # 实验元数据、数据来源/sha256、色盲安全色板
│   ├── data.py                   # 校验/清洗（重复、缺失、域值）+ 分位数分层
│   ├── ab_test.py                # z 检验/bootstrap/贝叶斯/SRM/MWU/功效（纯函数）
│   ├── charts.py                 # matplotlib 图表层（10 张）
│   ├── pipeline.py               # 编排 + 加工表落库 + REPORT.md 生成
│   └── download_data.py          # 重新下载并校验真实数据集
├── reports/
│   ├── REPORT.md                 # 完整分析报告（自动生成）
│   └── figures/                  # 10 张 PNG（README/报告直接引用）
└── tests/test_metrics.py         # 9 组口径测试（无需 pytest）
```

## 🚀 快速开始

```bash
pip install -r requirements.txt

# 1. 一键复现全部指标、检验、图表与报告（约 40 秒，含 1 万次 bootstrap）
python main.py
# 想更快：python main.py --bootstrap 2000

# 2. 启动交互看板（可现场调整 bootstrap 次数）
streamlit run app.py

# 3. 重新执行 notebook
jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb

# 4.（可选）重新下载真实数据集并做 sha256 校验
python -m src.download_data
```

只想看结果：直接打开 [`reports/REPORT.md`](reports/REPORT.md) 或
[`notebooks/EDA.ipynb`](notebooks/EDA.ipynb)，`data/processed/` 已附带全部加工结果。

## 🧮 方法论要点

- **先数据质量后指标**：SRM 是实验可信度的第一道闸门。本数据卡方 p=0.009，
  但偏差仅 0.44pp；采用 p 值 + 实际偏差双门槛，避免大样本下把哈希分流抖动误判为事故。
- **率值指标**：两比例 z 检验用合并 SE、CI 用非合并 SE；留存率的 bootstrap 用
  与伯努利重采样同分布的二项抽样（速度快三个数量量、结果等价）。
- **连续指标**：游玩局数离散重尾，用 Mann-Whitney U 秩检验 + rank-biserial 效应量，
  均值只通过 bootstrap 推断并做截尾敏感性分析（单个 49,854 局账号即可翻转结论倾向）。
- **频率派与贝叶斯互证**：z 检验回答"在无效应假设下数据有多极端"，
  贝叶斯（Jeffreys 无信息先验）直接回答"实验组更差的后验概率是多少"，
  bootstrap 不依赖正态近似——三套语言结论一致才下业务结论。
- **功效前置/复盘**：以 MDE 反推样本量，并据此解释 D1"不显著"是功效不足而非零效应。
- **处理后变量纪律**：参与度分层受实验影响，分层内对比只描述效应分布，不声称因果。
- 全部随机过程固定种子（20260919），结果逐位可复现。

## ✅ 测试

```bash
python tests/test_metrics.py     # 无需 pytest
```

用手工小样本验证：z 检验数值与 scipy 卡方一致、SRM 三级判读、bootstrap 种子可复现、
MWU 与 scipy 一致、Beta 后验均值、样本量/MDE 公式自洽、脏数据被拒绝、
Gini 退化情形与真实数据集加载（含 sha256）。

## 🔧 可扩展方向

- 接入收入/付费埋点：门禁同时是付费点，留存损失与 ARPU 变化需做联合决策（本数据集无收入字段）；
- 观察窗延长到 D14/D30 留存并做新奇效应（novelty effect）周度监控；
- 多臂实验探索最优门禁位置（25/30/35 关），配合序贯检验缩短实验周期；
- CUPED 等方差削减方法，用实验前行为降低指标方差、等效提升功效。

## 🛠️ 技术栈

pandas · NumPy · SciPy · matplotlib · Streamlit · Jupyter（Python 3.11）

## 📄 数据版权

数据集 © 其原始权利人（Tactile Entertainment / Kaggle 发布者），
本仓库出于学习与求职作品展示目的附带数据文件；若权利人要求将立即移除。
