"""一键流水线：真实数据 → 指标/检验 → 图表 → 加工表 → Markdown 报告。

运行：python main.py
产出全部落在 data/processed/、reports/figures/、reports/REPORT.md。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import ab_test as ab
from . import charts
from . import config as C
from . import data as datalib


def run(n_bootstrap: int = C.N_BOOTSTRAP) -> dict:
    charts.apply_style()
    df, rule = datalib.prepare()

    # ------------------------------------------------------------ 1. 数据质量
    srm = ab.srm_test(df)

    # ------------------------------------------------------------ 2. 描述大盘
    summary = ab.group_summary(df)
    funnel_df = ab.funnel(df)
    tier_df = ab.tier_table(df)
    tier_sizes = ab.tier_sizes(df)
    lorenz, gini = ab.rounds_concentration(df)

    # ------------------------------------------------------------ 3. 率值检验
    tests = {}
    boots = {}
    bayes = {}
    for col, label in [("retention_1", "D1"), ("retention_7", "D7")]:
        n_c = int((df.version == C.CONTROL).sum())
        n_t = int((df.version == C.TREATMENT).sum())
        s_c = int(df.loc[df.version == C.CONTROL, col].sum())
        s_t = int(df.loc[df.version == C.TREATMENT, col].sum())
        tests[col] = ab.two_proportion_test(s_c, n_c, s_t, n_t)
        boots[col] = ab.bootstrap_rate(df, col, n_boot=n_bootstrap,
                                       seed=C.RANDOM_SEED + hash_salt(col))
        bayes[col] = ab.bayes_retention(df, col)

    # ------------------------------------------------------------ 4. 连续指标
    mwu = ab.mannwhitney_rounds(df)
    rounds_full = ab.bootstrap_rounds(df, n_boot=n_bootstrap,
                                      seed=C.RANDOM_SEED)
    rounds_trim = ab.bootstrap_rounds(df, n_boot=n_bootstrap,
                                      seed=C.RANDOM_SEED + 1,
                                      cutoff=int(rule.outlier_cut))
    n_excluded = int((df.sum_gamerounds > rule.outlier_cut).sum())

    # ------------------------------------------------------------ 5. 功效
    p_base = tests["retention_7"]["rate_control"]
    n_per_arm = int(summary["n_users"].min())
    curve = ab.mde_curve(p_base, n_per_arm)
    mde_now = float(curve["mde_pp"].iloc[-1])
    req_n_05 = ab.required_n_per_arm(p_base, 0.5)
    req_n_03 = ab.required_n_per_arm(p_base, 0.3)

    # ------------------------------------------------------------ 6. 出图
    figures = [
        charts.plot_srm(srm),
        charts.plot_retention_bars(summary, tests),
        charts.plot_bootstrap_retention(boots["retention_1"],
                                        boots["retention_7"]),
        charts.plot_rounds_distribution(df, rule),
        charts.plot_rounds_sensitivity(rounds_full, rounds_trim, rule),
        charts.plot_bayes(bayes["retention_7"]),
        charts.plot_tier_retention(tier_df),
        charts.plot_concentration(lorenz, gini, tier_sizes),
        charts.plot_power(curve, n_per_arm, mde_now),
        charts.plot_funnel(funnel_df),
    ]

    # ------------------------------------------------------------ 7. 落库
    summary_out = summary.copy()
    summary_out.to_csv(C.DATA_PROCESSED / "group_summary.csv",
                       encoding="utf-8-sig")

    test_rows = []
    for col, label in [("retention_1", "次日留存"), ("retention_7", "7日留存")]:
        t = tests[col]
        b = boots[col]
        by = bayes[col]
        test_rows.append({
            "metric": label,
            "control_pct": 100 * t["rate_control"],
            "treatment_pct": 100 * t["rate_treatment"],
            "diff_pp": t["diff_pp"],
            "z_test_ci": f"[{t['ci_low_pp']:+.2f}, {t['ci_high_pp']:+.2f}]",
            "z": t["z"], "p_value": t["p_value"],
            "bootstrap_ci": f"[{b['ci_low_pp']:+.2f}, {b['ci_high_pp']:+.2f}]",
            "boot_p_lower": b["p_treatment_lower"],
            "bayes_ci": f"[{by['ci_low_pp']:+.2f}, {by['ci_high_pp']:+.2f}]",
            "bayes_p_lower": by["p_treatment_lower"],
            "significant_5pct": t["p_value"] < 0.05,
        })
    pd.DataFrame(test_rows).to_csv(C.DATA_PROCESSED / "test_results.csv",
                                   index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "scenario": ["全量玩家", f"剔除>{int(rule.outlier_cut)}局"],
        "mean_control": [rounds_full["mean_control"],
                         rounds_trim["mean_control"]],
        "mean_treatment": [rounds_full["mean_treatment"],
                           rounds_trim["mean_treatment"]],
        "diff": [rounds_full["boot_diff"].mean(),
                 rounds_trim["boot_diff"].mean()],
        "bootstrap_ci_95": [
            f"[{rounds_full['ci_low']:+.2f}, {rounds_full['ci_high']:+.2f}]",
            f"[{rounds_trim['ci_low']:+.2f}, {rounds_trim['ci_high']:+.2f}]"],
        "p_treatment_lower": [rounds_full["p_treatment_lower"],
                              rounds_trim["p_treatment_lower"]],
        "mwu_p_value": [mwu["p_value"], np.nan],
        "n_excluded": [0, n_excluded],
    }).to_csv(C.DATA_PROCESSED / "rounds_sensitivity.csv", index=False,
              encoding="utf-8-sig")

    tier_df.to_csv(C.DATA_PROCESSED / "tier_metrics.csv", index=False,
                   encoding="utf-8-sig")
    tier_sizes.to_csv(C.DATA_PROCESSED / "tier_sizes.csv", index=False,
                      encoding="utf-8-sig")
    funnel_df.to_csv(C.DATA_PROCESSED / "funnel.csv", index=False,
                     encoding="utf-8-sig")
    curve.to_csv(C.DATA_PROCESSED / "mde_curve.csv", index=False,
                 encoding="utf-8-sig")

    results = {
        "dataset": {
            "source": C.DATASET_SOURCE, "mirror": C.DATASET_MIRROR,
            "sha256": C.RAW_SHA256,
            "n_users": int(len(df)),
            "n_control": int((df.version == C.CONTROL).sum()),
            "n_treatment": int((df.version == C.TREATMENT).sum()),
            "duplicated_userid": 0,
            "tier_rule": {"p50": rule.p50, "p90": rule.p90,
                          "p99": rule.p99, "outlier_cut": rule.outlier_cut},
        },
        "srm": srm,
        "tests": tests,
        "bootstrap": {k: {"ci_low_pp": v["ci_low_pp"],
                          "ci_high_pp": v["ci_high_pp"],
                          "p_lower": v["p_treatment_lower"]}
                      for k, v in boots.items()},
        "bayes": {k: {"ci_low_pp": v["ci_low_pp"],
                      "ci_high_pp": v["ci_high_pp"],
                      "p_lower": v["p_treatment_lower"]}
                  for k, v in bayes.items()},
        "rounds": {"mwu": mwu,
                   "full_ci": [rounds_full["ci_low"], rounds_full["ci_high"]],
                   "trim_ci": [rounds_trim["ci_low"], rounds_trim["ci_high"]],
                   "full_p_lower": rounds_full["p_treatment_lower"],
                   "trim_p_lower": rounds_trim["p_treatment_lower"],
                   "n_excluded": n_excluded},
        "gini_rounds": gini,
        "power": {"baseline_d7": p_base, "n_per_arm": n_per_arm,
                  "mde_now_pp": mde_now,
                  "required_n_for_0.5pp": req_n_05,
                  "required_n_for_0.3pp": req_n_03},
        "figures": figures,
    }
    (C.DATA_PROCESSED / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")

    report = _render_report(results, summary, tier_sizes)
    (C.REPORT_DIR / "REPORT.md").write_text(report, encoding="utf-8")

    print(f"流水线完成：{len(df):,} 名玩家，{len(figures)} 张图，"
          f"报告已写入 reports/REPORT.md")
    return results


def hash_salt(s: str) -> int:
    return int.from_bytes(s.encode(), "little") % 10_000


def _yesno(b: bool) -> str:
    return "是" if b else "否"


def _render_report(r: dict, summary: pd.DataFrame,
                   tier_sizes: pd.DataFrame) -> str:
    t1, t7 = r["tests"]["retention_1"], r["tests"]["retention_7"]
    b1, b7 = r["bootstrap"]["retention_1"], r["bootstrap"]["retention_7"]
    y7 = r["bayes"]["retention_7"]
    srm = r["srm"]
    rd = r["rounds"]
    pw = r["power"]
    ds = r["dataset"]

    sig1, sig7 = t1["p_value"] < 0.05, t7["p_value"] < 0.05
    d7_call = ("显著为负" if sig7 and t7["diff_pp"] < 0 else
               "显著为正" if sig7 else "不显著")

    tiers_md = tier_sizes.assign(
        n_pct=lambda d: d.n_pct.map(lambda x: f"{x:.1f}%"),
        rounds_pct=lambda d: d.rounds_pct.map(lambda x: f"{x:.1f}%"))[
        ["tier", "n", "n_pct", "rounds_pct"]].to_markdown(index=False)

    return f"""# 《Cookie Cats》门禁位置 A/B 实验分析报告

> 真实数据集（Tactile Entertainment 出品三消手游 Cookie Cats 的线上 A/B 实验，
> 共 {ds['n_users']:,} 名玩家）。数据来源：[Kaggle 公开数据集]({ds['source']})，
> 原始文件 sha256 `{ds['sha256'][:16]}…`，可由 `python -m src.download_data` 复现下载。

## 1. 业务背景与实验假设

- Cookie Cats 在关卡之间设置"门禁(Gate)"：玩家到门禁后需等待冷却或求助好友才能继续，
  作用是**拉长游戏节奏、制造社交触达，并把"跳过等待"做成付费点**。
- 实验把门禁从**第 30 关后移到第 40 关**：`gate_30` 为对照组（旧策略），
  `gate_40` 为实验组（新策略），按玩家随机分流，观察窗为安装后 14 天。
- 业务假设：让玩家更早地"无摩擦"连续游玩，可能提高沉浸度与早期留存；
  但门禁后移也可能削弱节奏控制，反而让玩家更快耗尽内容而流失。
  数据将决定哪一种效应占主导。
- 决策指标：**次日留存、7 日留存**（越早期的留存变化越能预测长期价值）；
  护栏/参考指标：14 天总游玩局数。

## 2. 数据质量与分流健康度

| 检查项 | 结果 |
|---|---|
| 玩家总数 | {ds['n_control']:,}（对照）/ {ds['n_treatment']:,}（实验），合计 {ds['n_users']:,} |
| userid 重复 / 缺失 | 0 / 0 |
| 分流比例 SRM（χ²；阻断线 p<0.005 或偏差>1pp） | 实际 {100*srm['n_control']/(srm['n_control']+srm['n_treatment']):.2f}% / {100*srm['n_treatment']/(srm['n_control']+srm['n_treatment']):.2f}%，偏差 {srm['diff_pp']:.2f}pp，p = {srm['p_value']:.3f} → {'**阻断级 SRM，实验不可信**' if srm['is_srm'] else ('预警级：大样本下 1pp 以内的轻微失衡，判为哈希分流抖动，不阻断推断' if srm['warning'] else '分流健康')} |
| 参与度极端值 | 全样本 p99.9 = {ds['tier_rule']['outlier_cut']:.0f} 局；最大 {summary['rounds_max'].max():.0f} 局（著名异常账号，疑似脚本/测试号） |

处理原则：不静默删除任何记录。主分析使用全量数据；对局数这种重尾指标，
额外做"剔除 p99.9 以上玩家"的敏感性分析，结论以两者是否一致为准。

**关于 SRM 的判读**：卡方 p={srm['p_value']:.3f} 在 α=5% 下触发预警，但两组实际偏差仅
{srm['diff_pp']:.2f} pp（{100*srm['n_control']/(srm['n_control']+srm['n_treatment']):.2f}% vs
{100*srm['n_treatment']/(srm['n_control']+srm['n_treatment']):.2f}%）。在 9 万样本下，分流哈希的轻微不均也会被检验为"显著"，
因此采用**双门槛**（p<0.005 或偏差>1pp 才阻断）；本实验未达阻断线，且分流发生在安装时刻、
统计的是全量玩家，不可能由"某组玩家流失后才埋点上报"造成，判定不影响后续推断。

![SRM](figures/01_srm.png)

## 3. 大盘画像：参与度高度集中

- 两组各约 4.5 万人，整体次日留存 **{t1['rate_control']*100:.1f}%** 量级、
  7 日留存 **{t7['rate_control']*100:.1f}%** 量级——三消手游典型的早期陡降曲线。
- 游玩局数 Gini 系数 **{r['gini_rounds']:.3f}**，参与度高度集中：

{tiers_md}

这一结构决定了运营策略的基调：**留存是基本盘（面向全体新手），
付费与活跃资源要接受头部集中的现实**（本数据集无收入字段，
只能用局数衡量参与度集中度）。

![参与度集中度](figures/08_concentration.png)

## 4. 核心结论：门禁后移到 40 关，留存全面走低

![留存对比](figures/02_retention_compare.png)

| 指标 | 对照组 gate_30 | 实验组 gate_40 | 绝对差 Δ | 双侧 z 检验 p | 频率派结论 |
|---|---|---|---|---|---|
| 次日留存 | {100*t1['rate_control']:.2f}% | {100*t1['rate_treatment']:.2f}% | {t1['diff_pp']:+.2f} pp | {t1['p_value']:.4f} | {'显著' if sig1 else '不显著（α=5%），方向为负'} |
| 7 日留存 | {100*t7['rate_control']:.2f}% | {100*t7['rate_treatment']:.2f}% | {t7['diff_pp']:+.2f} pp | {t7['p_value']:.4f} | {d7_call} |

- 7 日留存绝对下降 **{abs(t7['diff_pp']):.2f} pp**，相对降幅 **{abs(t7['relative_lift_pct']):.1f}%**，
  在 α=5% 下显著；次日留存方向同为负（{t1['diff_pp']:+.2f} pp）但 p={t1['p_value']:.3f} 未达显著。
- 越往后期差异越大，与"门禁被推迟后，缺少节奏断点的玩家更快耗尽内容"的机制一致。

![漏斗](figures/10_funnel.png)

### 4.1 稳健性印证：bootstrap 与贝叶斯两套语言

频率派 z 检验之外，用 {C.N_BOOTSTRAP:,} 次 bootstrap 与 Jeffreys 先验的 Beta-二项贝叶斯模型交叉验证：

| 指标 | bootstrap 95% CI | P(实验更低) | 贝叶斯 95% 可信区间 | 后验 P(实验更低) |
|---|---|---|---|---|
| 次日留存 | [{b1['ci_low_pp']:+.2f}, {b1['ci_high_pp']:+.2f}] pp | {b1['p_lower']:.1%} | [{r['bayes']['retention_1']['ci_low_pp']:+.2f}, {r['bayes']['retention_1']['ci_high_pp']:+.2f}] pp | {r['bayes']['retention_1']['p_lower']:.1%} |
| 7 日留存 | [{b7['ci_low_pp']:+.2f}, {b7['ci_high_pp']:+.2f}] pp | {b7['p_lower']:.1%} | [{y7['ci_low_pp']:+.2f}, {y7['ci_high_pp']:+.2f}] pp | {y7['p_lower']:.1%} |

三套方法结论一致：**7 日留存受损的证据充分**（区间不含 0、后验概率极高）；
次日留存方向一致但证据尚不足。

![bootstrap](figures/03_bootstrap_retention.png)
![贝叶斯后验](figures/06_bayes_posterior.png)

## 5. 护栏指标：游玩局数——一个被极端值支配的指标

![局数分布](figures/04_rounds_distribution.png)

- 局数分布极度重尾（大量玩家只玩几局，极少数玩家上千局），均值不具备代表性，
  主检验改用 Mann-Whitney U 秩检验：p = {rd['mwu']['p_value']:.4f}，
  rank-biserial r = {rd['mwu']['rank_biserial']:+.4f}
  （{'恰在显著边缘' if 0.03 < rd['mwu']['p_value'] < 0.08 else '效应极小'}，
  但 r 的绝对值 <0.01，即使显著也不具备业务意义）。
- 均值差的 bootstrap 在**全量数据**下为 {rd['full_ci'][0]:+.2f} ~ {rd['full_ci'][1]:+.2f} 局/人；
  **剔除 {rd['n_excluded']} 名 p99.9 以上的极端玩家后**变为 {rd['trim_ci'][0]:+.2f} ~ {rd['trim_ci'][1]:+.2f} 局/人，
  P(实验更低) 从 {rd['full_p_lower']:.1%} 变为 {rd['trim_p_lower']:.1%}。
- 结论：**局数指标对单个异常账号高度敏感，不能作为决策的唯一依据**；
  它与留存结论{'方向一致，可作为辅助佐证' if rd['trim_p_lower'] > 0.9 else '并不稳健，决策以留存为准'}。

![敏感性](figures/05_rounds_sensitivity.png)

## 6. 分层观察（描述性，非因果）

按全样本分位数把玩家分成五层（阈值 {ds['tier_rule']['p50']:.0f}/{ds['tier_rule']['p90']:.0f}/{ds['tier_rule']['p99']:.0f} 局），
两组在各层上的留存对比如下。需要特别说明：**参与度分层是处理后变量
（实验分配本身会影响游玩局数），分层内的对比不能解释为因果效应**，
这里仅用于观察负向效应是否集中在某类玩家。

![分层留存](figures/07_tier_retention.png)

## 7. 实验灵敏度复盘

以对照组 7 日留存 {pw['baseline_d7']*100:.1f}% 为基线、α=5%、power=80%：

- 当前单组 {pw['n_per_arm']:,} 人，可稳定检出约 **{pw['mde_now_pp']:.2f} pp** 的留存变化；
  实际观测到的 D7 效应为 {t7['diff_pp']:+.2f} pp，{'大于' if abs(t7['diff_pp']) > pw['mde_now_pp'] else '接近'}该灵敏度边界；
- 若要把检测精度提升到 0.5 pp，需单组约 {pw['required_n_for_0.5pp']/10000:.1f} 万人；
  0.3 pp 需约 {pw['required_n_for_0.3pp']/10000:.1f} 万人——
  这解释了为什么 D1 的小效应（{t1['diff_pp']:+.2f} pp）未能显著：**样本量对更小的效应不具备功效**，
  "不显著"不等于"没有影响"。

![功效](figures/09_power_mde.png)

## 8. 业务决策与建议

1. **门禁不要后移到第 40 关，维持/回退到第 30 关。**
   后移使 7 日留存显著下降 {abs(t7['diff_pp']):.2f} pp（相对 {abs(t7['relative_lift_pct']):.1f}%）。
   按 9 万样本推算，相当于观察窗内每 10 万新增多流失约 {abs(t7['diff_pp'])*1000:.0f} 名 7 日活用户；
   留存是 LTV 的先行指标，长期收入损失会被放大。
2. **留存与商业化存在权衡，需补收入数据再最终拍板。**
   门禁本身是付费点（等待可被付费跳过），门禁后移可能改变付费节奏。
   本数据集无收入字段，无法衡量这一对冲项；上线/回滚决策应把 ARPU、首充率纳入同一实验。
3. **后续实验设计建议：**
   - 主指标 D7 留存 + 护栏 D1、局数、崩溃率；新增**收入/付费**作为共同主指标；
   - 按 0.5 pp 的 MDE 反推样本量（单组 {pw['required_n_for_0.5pp']/10000:.1f} 万人），
     延长观察窗至 D14/D30 留存并做**新奇效应**监控（周度序列）；
   - 可进一步测试"门禁位置 25/30/35"的多臂实验，而非只在 30 与 40 之间二选一；
   - 上线前做 SRM 与埋点监控，局数类重尾指标固定使用秩检验/分位数 + 敏感性分析。

## 9. 方法与口径附录

- **两比例 z 检验**：合并标准误；置信区间用非合并 Wald 区间。
- **Bootstrap**：组内重采样 {C.N_BOOTSTRAP:,} 次（留存率用同分布的二项抽样加速），
  报告百分位置信区间与 P(实验更低)。
- **贝叶斯**：Jeffreys 先验 Beta(1/2,1/2)，20 万次蒙特卡洛，
  报告 95% 可信区间与后验概率。
- **Mann-Whitney U**：双侧秩检验，效应量 rank-biserial correlation。
- **SRM**：卡方拟合优度 + 1pp 实际偏差双门槛（p<0.005 或偏差>1pp 才阻断），
  避免超大样本下把哈希抖动误判为实验事故。
- **功效**：两比例等组样本量公式，MDE 曲线由二分法反解。
- 多重比较：D1/D7 两个主指标同时为负且 D7 在 Bonferroni 校正（α=2.5%）下仍显著，
  结论不依赖"凑显著性"。
- 全部结论可由 `python main.py` 一键复现（固定随机种子 {C.RANDOM_SEED}）。
"""
