"""一次性脚本：生成 EDA notebook（随后用 nbconvert 执行）。"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 《Cookie Cats》门禁位置 A/B 实验 — 探索分析

**数据**：Tactile Entertainment 三消手游 Cookie Cats 的线上随机对照实验（真实数据，
[Kaggle 公开数据集](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)）。

**实验**：关卡门禁从第 30 关后移到第 40 关。`gate_30` 对照组 vs `gate_40` 实验组，
观察窗为安装后 14 天，共 90,189 名玩家。

**字段**：`userid` 玩家，`version` 分组，`sum_gamerounds` 14 天内总游玩局数，
`retention_1` 次日是否回流，`retention_7` 第 7 天是否回流。

> 结论速览：门禁后移使 **7 日留存显著下降 0.82pp（p=0.0016）**，
> 次日留存方向为负但不显著；游玩局数差异被极端账号主导、不稳健。
> 建议门禁维持在 30 关，并把收入指标纳入后续实验。""")

code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # 允许从 src 导入

import numpy as np
import pandas as pd
from IPython.display import Image, display

from src import config as C
from src import data as datalib
from src import ab_test as ab
from src import charts

pd.set_option("display.width", 120)
charts.apply_style()
df, rule = datalib.prepare()
df.shape""")

md("## 1. 数据理解与质量审计\n\n先看字段、分布、缺失与主键重复。")
code("""display(df.head())
display(df.dtypes)
print("缺失值:\\n", df[['userid','version','sum_gamerounds','retention_1','retention_7']].isna().sum())
print("userid 重复数:", df['userid'].duplicated().sum())
print("分组取值:", df['version'].value_counts().to_dict())
print("局数为负:", (df['sum_gamerounds'] < 0).sum())""")

code("""# SRM：样本比例失衡检验（实验可信度的第一道闸门）
srm = ab.srm_test(df)
pd.Series(srm)""")

md("""**SRM 判读**：分流比 49.56% / 50.44%，卡方 p=0.009 触发统计预警，
但实际偏差仅 0.44pp（<1pp 业务阈值）。9 万样本下分流哈希的轻微抖动也会被检出，
本项目采用 p<0.005 或偏差>1pp 的**双门槛**，未达阻断线；
且分流发生在安装时刻、统计口径是全量玩家，选择性流失无法解释该偏差。""")

md("## 2. 大盘画像：留存漏斗与参与度集中度")
code("""summary = ab.group_summary(df)
summary.T""")

code("""display(ab.funnel(df))
tier_sizes = ab.tier_sizes(df)
display(tier_sizes)
lorenz, gini = ab.rounds_concentration(df)
print("游玩局数 Gini =", round(gini, 3))""")

md("""- 次日留存 ~44%、7 日留存 ~19%，三消手游典型的早期陡降；
- **4.4% 的玩家安装后一局未玩**；1% 的超核玩家贡献 14.6% 的局数，Gini=0.709，
  参与度高度集中——留存运营面向全体新手，资源分配要接受头部集中现实。""")

code("""Image(filename=str(C.FIG_DIR / '08_concentration.png'))""")

md("## 3. 核心推断：门禁后移对留存的影响")
code("""results = {}
for col, name in [('retention_1', '次日留存'), ('retention_7', '7日留存')]:
    n_c = (df.version == C.CONTROL).sum()
    n_t = (df.version == C.TREATMENT).sum()
    s_c = df.loc[df.version == C.CONTROL, col].sum()
    s_t = df.loc[df.version == C.TREATMENT, col].sum()
    results[name] = ab.two_proportion_test(s_c, n_c, s_t, n_t)
pd.DataFrame(results).T""")

md("""- 次日留存：44.82% → 44.23%，Δ=-0.59pp，p=0.074，**不显著但方向为负**；
- 7 日留存：19.02% → 18.20%，Δ=-0.82pp，p=0.0016，**显著为负**，相对降幅 4.3%。

### 3.1 Bootstrap（非参数，10,000 次）""")
code("""for col, name in [('retention_1', '次日留存'), ('retention_7', '7日留存')]:
    b = ab.bootstrap_rate(df, col, n_boot=10_000)
    print(name, f"95% CI [{b['ci_low_pp']:+.2f}, {b['ci_high_pp']:+.2f}] pp,",
          f"P(实验组更低) = {b['p_treatment_lower']:.1%}")""")

md("### 3.2 贝叶斯：Jeffreys 先验 Beta(½,½)")
code("""for col, name in [('retention_1', '次日留存'), ('retention_7', '7日留存')]:
    y = ab.bayes_retention(df, col, n_mc=200_000)
    print(name, f"95% CrI [{y['ci_low_pp']:+.2f}, {y['ci_high_pp']:+.2f}] pp,",
          f"后验 P(实验组更低) = {y['p_treatment_lower']:.1%}")
Image(filename=str(C.FIG_DIR / '06_bayes_posterior.png'))""")

md("""**三套方法（z 检验 / bootstrap / 贝叶斯）结论一致**：
D7 受损证据充分（频率派 p=0.0016，95% 区间不含 0，后验概率 99.9%）；
D1 方向一致但证据不足。""")
code("""Image(filename=str(C.FIG_DIR / '03_bootstrap_retention.png'))""")

md("## 4. 护栏指标：游玩局数与极端值敏感性")
code("""mwu = ab.mannwhitney_rounds(df)
print('Mann-Whitney U p =', round(mwu['p_value'], 4),
      ' rank-biserial r =', round(mwu['rank_biserial'], 4))
full = ab.bootstrap_rounds(df, n_boot=2000)
trim = ab.bootstrap_rounds(df, n_boot=2000, cutoff=int(rule.outlier_cut))
print(f"全量: 均值差 CI [{full['ci_low']:+.2f}, {full['ci_high']:+.2f}],"
      f" P(更低)={full['p_treatment_lower']:.1%}")
n_excl = int((df.sum_gamerounds > rule.outlier_cut).sum())
print(f"剔除 {n_excl} 名 >{int(rule.outlier_cut)} 局玩家: "
      f"CI [{trim['ci_low']:+.2f}, {trim['ci_high']:+.2f}], P(更低)={trim['p_treatment_lower']:.1%}")
print("最大局数:", df['sum_gamerounds'].max())""")

md("""局数分布极度重尾，单个玩家玩了 49,854 局（疑似脚本/测试号）。
全量时均值差 bootstrap 的 P(更低)=80%，剔除 91 名极端玩家后降到 68%——
**结论被少数账号支配，不能作为决策依据**；MWU 效应量 r 仅 -0.0075，无业务意义。
决策以留存指标为准。""")
code("""Image(filename=str(C.FIG_DIR / '05_rounds_sensitivity.png'))""")

md("## 5. 分层观察（处理后变量，仅描述）\n\n参与度是实验的处理后变量，分层内差异不能做因果解释，只看负效应是否集中。")
code("""tier_df = ab.tier_table(df)
display(tier_df.pivot(index='tier', columns='version',
                      values=['n','retention_1','retention_7']))
Image(filename=str(C.FIG_DIR / '07_tier_retention.png'))""")

md("## 6. 实验灵敏度复盘")
code("""n_per_arm = int(summary['n_users'].min())
p_base = results['7日留存']['rate_control']
curve = ab.mde_curve(p_base, n_per_arm)
mde_now = curve['mde_pp'].iloc[-1]
print(f"当前单组 {n_per_arm:,} 人，可检出 MDE ≈ {mde_now:.2f}pp（α=5%, power=80%）")
print("检出 0.5pp 需单组", f"{ab.required_n_per_arm(p_base, 0.5)/1e4:.1f} 万人")
print("检出 0.3pp 需单组", f"{ab.required_n_per_arm(p_base, 0.3)/1e4:.1f} 万人")
Image(filename=str(C.FIG_DIR / '09_power_mde.png'))""")

md("""## 7. 结论与业务建议

1. **门禁维持在第 30 关（回滚 gate_40）**：后移使 7 日留存显著下降 0.82pp（相对 4.3%），
   折合每 10 万新增在观察窗内多流失约 820 名 D7 玩家；留存是 LTV 先行指标，长期损失放大。
2. **机制解释**：门禁是节奏断点 + 社交触达 + 付费点；过早取消断点，玩家更快耗尽内容。
3. **必要的对冲检查**：门禁本身驱动付费（等待可花钱跳过），本数据无收入字段，
   正式拍板要把 ARPU/首充率纳入共同主指标。
4. **后续实验**：按 0.5pp MDE 备样本（单组 ~9.8 万）、延长到 D14/D30 观察窗并监控新奇效应、
   尝试 25/30/35 关多臂实验；重尾指标固定用秩检验 + 敏感性分析。
""")

nb["cells"] = cells
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
out = Path(__file__).parent / "EDA.ipynb"
nbf.write(nb, out)
print("written", out)
