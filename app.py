"""Cookie Cats A/B 实验决策看板：streamlit run app.py

页面：实验概览 / 假设检验 / 自助推断（交互式 bootstrap & 贝叶斯）/
护栏指标 / 分层与功效 / 决策建议。
图表与流水线一致；bootstrap 模块支持现场调整重采样次数。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import ab_test as ab
from src import charts
from src import config as C
from src import data as datalib

st.set_page_config(page_title="Cookie Cats A/B 实验分析", layout="wide")

# ----------------------------------------------------------------- 数据
@st.cache_data(show_spinner=False)
def _load():
    charts.apply_style()
    return datalib.prepare()


df, rule = _load()
srm = ab.srm_test(df)
summary = ab.group_summary(df)

st.title("🎮《Cookie Cats》门禁位置 A/B 实验决策看板")
st.caption(
    "真实数据：Tactile Entertainment 三消手游，90,189 名玩家，"
    "对照 gate_30（门禁在第 30 关）vs 实验 gate_40（后移到第 40 关），观察窗 14 天。"
    "数据源：[Kaggle 公开数据集](%s)" % C.DATASET_SOURCE)

tabs = st.tabs(["① 实验概览与数据质量", "② 核心假设检验",
                "③ 自助推断（可交互）", "④ 护栏指标：游玩局数",
                "⑤ 分层与功效", "⑥ 决策建议"])

# ------------------------------------------------------- 1 概览
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("玩家总数", f"{len(df):,}")
    c2.metric("对照组 / 实验组",
              f"{srm['n_control']:,} / {srm['n_treatment']:,}")
    c3.metric("分流偏差", f"{srm['diff_pp']:.2f} pp",
              delta="健康" if not srm["is_srm"] and not srm["warning"]
              else "预警,不阻断" if srm["warning"] else "阻断级 SRM",
              delta_color="off" if not srm["is_srm"] else "inverse")
    c4.metric("SRM 卡方 p 值", f"{srm['p_value']:.3f}")
    st.markdown(
        "**SRM 判读**：p<0.05 触发统计预警，但分流偏差仅 "
        f"{srm['diff_pp']:.2f}pp（<1pp 业务阈值）。9 万样本下哈希分流的轻微抖动也会被"
        "检验为显著，采用 **p<0.005 或偏差>1pp 双门槛**才阻断实验，本实验通过。")

    st.subheader("分组核心指标")
    st.dataframe(summary.style.format({
        "share_pct": "{:.2f}", "zero_round_pct": "{:.2f}",
        "rounds_mean": "{:.2f}", "rounds_median": "{:.0f}",
        "rounds_p90": "{:.0f}", "rounds_p99": "{:.0f}",
        "retention_1_pct": "{:.2f}", "retention_7_pct": "{:.2f}",
        "retained1_not7_pct": "{:.2f}"}), width="stretch")

    st.subheader("留存漏斗")
    st.image(str(C.FIG_DIR / "10_funnel.png"), width="stretch")

# ------------------------------------------------------- 2 假设检验
with tabs[1]:
    st.markdown(
        "主指标为次日留存与 7 日留存。频率派使用**两比例 z 检验**（合并 SE），"
        "置信区间用非合并 Wald 区间；p 值同时与 scipy 卡方检验交叉验证。")
    rows = []
    for col, name in [("retention_1", "次日留存"), ("retention_7", "7 日留存")]:
        n_c = int((df.version == C.CONTROL).sum())
        n_t = int((df.version == C.TREATMENT).sum())
        t = ab.two_proportion_test(int(df.loc[df.version == C.CONTROL, col].sum()),
                                   n_c,
                                   int(df.loc[df.version == C.TREATMENT, col].sum()),
                                   n_t)
        rows.append({"指标": name,
                     "对照组": f"{100*t['rate_control']:.2f}%",
                     "实验组": f"{100*t['rate_treatment']:.2f}%",
                     "绝对差 Δ": f"{t['diff_pp']:+.2f} pp",
                     "相对变化": f"{t['relative_lift_pct']:+.1f}%",
                     "95% CI": f"[{t['ci_low_pp']:+.2f}, {t['ci_high_pp']:+.2f}] pp",
                     "z": f"{t['z']:.3f}",
                     "p 值": f"{t['p_value']:.4f}",
                     "α=5% 显著": "✅" if t["p_value"] < 0.05 else "❌"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.image(str(C.FIG_DIR / "02_retention_compare.png"),
             width="stretch")
    st.info(
        f"7 日留存显著下降 {0.82:.2f}pp（p=0.0016，Bonferroni 校正后仍显著）；"
        "次日留存方向为负但不显著（p=0.074）。"
        "'不显著' ≠ '没有影响'：结合 96% 的 bootstrap 负向概率与功效分析，"
        "更可能是效应偏小、样本量不足以在 D1 上检出。")

# ------------------------------------------------------- 3 自助推断
with tabs[2]:
    st.markdown(
        "Bootstrap 在两组内各自重采样（留存率以同分布的二项抽样加速），"
        "贝叶斯侧使用 Jeffreys 先验 Beta(½,½) + 20 万次蒙特卡洛。")
    b = st.slider("Bootstrap 次数", 500, 10_000, 2_000, step=500)
    if st.button("🔄 重新运行 bootstrap", type="primary"):
        st.cache_data.clear()

    col = st.radio("指标", ["retention_7", "retention_1"],
                   format_func=lambda x: "7 日留存" if x == "retention_7"
                   else "次日留存", horizontal=True)

    @st.cache_data(show_spinner=True)
    def _rate_boot(col, n):
        return ab.bootstrap_rate(df, col, n_boot=n)

    @st.cache_data(show_spinner=True)
    def _bayes(col):
        return ab.bayes_retention(df, col)

    boot = _rate_boot(col, b)
    by = _bayes(col)
    cc1, cc2 = st.columns(2)
    cc1.metric("Bootstrap P(实验组更低)",
               f"{boot['p_treatment_lower']:.1%}",
               f"95% CI [{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}] pp")
    cc2.metric("贝叶斯后验 P(实验组更低)",
               f"{by['p_treatment_lower']:.1%}",
               f"95% CrI [{by['ci_low_pp']:+.2f}, {by['ci_high_pp']:+.2f}] pp")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(boot["boot_diff_pp"], bins=60, color=C.PALETTE["blue"],
            alpha=0.8, zorder=3)
    ax.axvline(0, color=C.PALETTE["red"], lw=1.6)
    ax.axvline(boot["ci_low_pp"], color=C.INK["muted"], ls="--")
    ax.axvline(boot["ci_high_pp"], color=C.INK["muted"], ls="--")
    ax.set_title(f"{'7 日留存' if col == 'retention_7' else '次日留存'}"
                 "差的 bootstrap 分布（实验组 − 对照组）", loc="left")
    ax.set_xlabel("百分点")
    ax.grid(axis="y")
    st.pyplot(fig, width="stretch")
    st.image(str(C.FIG_DIR / "06_bayes_posterior.png"),
             width="stretch")

# ------------------------------------------------------- 4 局数
with tabs[3]:
    mwu = ab.mannwhitney_rounds(df)
    st.markdown(
        f"局数极度重尾（最大 {df['sum_gamerounds'].max():,} 局，疑似脚本/测试号），"
        f"均值不具代表性。Mann-Whitney U 双侧 p={mwu['p_value']:.4f}，"
        f"rank-biserial r={mwu['rank_biserial']:+.4f}（|r|<0.01，无业务意义）。")
    st.image(str(C.FIG_DIR / "04_rounds_distribution.png"),
             width="stretch")
    st.image(str(C.FIG_DIR / "05_rounds_sensitivity.png"),
             width="stretch")
    st.dataframe(pd.read_csv(C.DATA_PROCESSED / "rounds_sensitivity.csv"),
                 width="stretch", hide_index=True)
    st.warning("剔除 91 名 p99.9 以上的极端玩家后，均值差的方向概率明显回落，"
               "说明该指标被少数账号支配——**护栏指标不能作为决策依据，结论以留存为准**。")

# ------------------------------------------------------- 5 分层/功效
with tabs[4]:
    st.image(str(C.FIG_DIR / "08_concentration.png"),
             width="stretch")
    st.image(str(C.FIG_DIR / "07_tier_retention.png"),
             width="stretch")
    st.caption("分层阈值（全样本分位数）："
               f"≤{rule.p50:.0f} 轻度 / ≤{rule.p90:.0f} 中度 / "
               f"≤{rule.p99:.0f} 重度 / 其余超核。参与度是处理后变量，"
               "分层内对比仅描述效应分布，不作因果解释。")
    st.image(str(C.FIG_DIR / "09_power_mde.png"),
             width="stretch")

# ------------------------------------------------------- 6 决策
with tabs[5]:
    st.success("**建议：门禁维持在第 30 关，gate_40 不全量上线。**")
    st.markdown(
        """
1.  **证据链**：D7 留存 19.02% → 18.20%（Δ=-0.82pp，p=0.0016，
    bootstrap/贝叶斯 P≈99.9%，Bonferroni 校正后仍显著）；D1 方向一致但不显著。
    折合每 10 万新增在观察窗内多流失约 **820 名 D7 玩家**，
    留存是 LTV 先行指标，长期收入损失会被放大。
2.  **机制**：门禁是节奏断点 + 社交触达 + 付费点，后移使玩家更快耗尽内容，
    与"越到后期差异越大"（D7 大于 D1）的观测吻合。
3.  **必要对冲**：门禁本身驱动付费（花钱跳过等待），本数据无收入字段，
    正式决策需把 ARPU / 首充率纳入同一实验做联合判读。
4.  **后续实验**：按 0.5pp MDE 备约 9.8 万人/组；观察窗延长到 D14/D30 并监控新奇效应；
    尝试 25/30/35 关多臂实验；上线前固定 SRM 与埋点监控，重尾指标一律秩检验 + 敏感性分析。
""")
