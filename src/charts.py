"""matplotlib 图表层（实验分析专用）。

规范：
- 对照组永远蓝、实验组永远橙（config.GROUP_COLORS，颜色跟随实体）；
- 2px 线条、细柱、网格退居背景；>=2 系列必有图例；
- 差值为负的面板用红色 0 参考线与文字标注，不靠颜色单独传达结论。
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C


def apply_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS",
                            "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "figure.facecolor": C.INK["surface"],
        "axes.facecolor": C.INK["surface"],
        "savefig.facecolor": C.INK["surface"],
        "axes.edgecolor": C.INK["baseline"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": C.INK["secondary"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": C.INK["primary"],
        "text.color": C.INK["primary"],
        "xtick.color": C.INK["muted"],
        "ytick.color": C.INK["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": C.INK["grid"],
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": C.INK["secondary"],
    })


def _finalize(ax, *, ygrid: bool = True, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ygrid:
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, name: str) -> str:
    path = C.FIG_DIR / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _color(g: str) -> str:
    return C.GROUP_COLORS[g]


# 1. SRM 分流健康度
def plot_srm(srm: dict) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = ["对照组\ngate_30", "实验组\ngate_40"]
    vals = [srm["n_control"], srm["n_treatment"]]
    bars = ax.bar(labels, vals, color=[_color(C.CONTROL), _color(C.TREATMENT)],
                  width=0.5, zorder=3)
    exp = (vals[0] + vals[1]) / 2
    ax.axhline(exp, color=C.INK["muted"], ls="--", lw=1.2,
               label="期望 50% 分流线")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 300, f"{v:,}",
                ha="center", fontsize=10, fontweight="bold")
    if srm["is_srm"]:
        verdict = "阻断级失衡，实验不可用"
    elif srm["warning"]:
        verdict = "统计预警（偏差<1pp，记录根因，不阻断推断）"
    else:
        verdict = "分流健康"
    r_c = 100 * srm["n_control"] / (srm["n_control"] + srm["n_treatment"])
    _finalize(ax, title=f"分流比例检查：实际 {r_c:.2f}% / {100-r_c:.2f}%，"
                        f"偏差 {srm['diff_pp']:.2f}pp，χ² p={srm['p_value']:.3f}"
                        f" → {verdict}",
              ylabel="玩家数")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=2)
    return _save(fig, "01_srm.png")


# 2. 两组核心留存率
def plot_retention_bars(summary: pd.DataFrame, tests: dict) -> str:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    metrics = [("retention_1_pct", "retention_1", "次日留存"),
               ("retention_7_pct", "retention_7", "7 日留存")]
    x = np.arange(2)
    width = 0.34
    for i, g in enumerate(C.GROUPS):
        vals = [summary.loc[g, col] for col, _, _ in metrics]
        b = ax.bar(x + (i - 0.5) * width, vals, width,
                   label=C.GROUP_LABEL[g], color=_color(g), zorder=3)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.4,
                    f"{v:.2f}%", ha="center", fontsize=9)
    for j, (_, key, name) in enumerate(metrics):
        t = tests[key]
        ax.text(j, 2, f"Δ = {t['diff_pp']:+.2f} pp\np = {t['p_value']:.3f}",
                ha="center", fontsize=9, color=C.PALETTE["red"])
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, _, n in metrics])
    ax.set_ylim(0, max(summary[["retention_1_pct", "retention_7_pct"]].max())
                + 6)
    _finalize(ax, title="核心指标：实验组（门禁后移到 40 关）留存全面走低",
              ylabel="留存率 %")
    ax.legend(loc="upper right")
    return _save(fig, "02_retention_compare.png")


# 3. 留存差 bootstrap
def plot_bootstrap_retention(boot1: dict, boot7: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), sharey=True)
    for ax, boot, name in zip(axes, [boot1, boot7], ["次日留存", "7 日留存"]):
        d = boot["boot_diff_pp"]
        ax.hist(d, bins=60, color=C.PALETTE["blue"], alpha=0.75, zorder=3)
        ax.axvline(0, color=C.PALETTE["red"], lw=1.6)
        ax.axvline(boot["ci_low_pp"], color=C.INK["muted"], ls="--", lw=1.2)
        ax.axvline(boot["ci_high_pp"], color=C.INK["muted"], ls="--", lw=1.2)
        ax.text(0.97, 0.95,
                f"95% CI [{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}] pp\n"
                f"P(实验组更低) = {boot['p_treatment_lower']:.1%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        _finalize(ax, title=name, xlabel="实验组 − 对照组（百分点）")
    axes[0].set_ylabel("bootstrap 频次")
    return _save(fig, "03_bootstrap_retention.png")


# 4. 局数分布（重尾 + 极端值）
def plot_rounds_distribution(df: pd.DataFrame, rule) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    bins = np.logspace(0, np.log10(df["sum_gamerounds"].max() + 1), 60)
    for g in C.GROUPS:
        x = df.loc[df.version == g, "sum_gamerounds"]
        axes[0].hist(x, bins=bins, density=True, histtype="step",
                     lw=2, label=C.GROUP_LABEL[g], color=_color(g))
    axes[0].set_xscale("log")
    axes[0].axvline(rule.outlier_cut, color=C.PALETTE["red"], ls="--", lw=1.2)
    axes[0].text(rule.outlier_cut * 1.05, axes[0].get_ylim()[1] * 0.6,
                 f"p99.9={rule.outlier_cut:.0f}", color=C.PALETTE["red"],
                 fontsize=8.5, rotation=90, va="center")
    _finalize(axes[0], title="14 天内总游玩局数分布（对数轴，重度重尾）",
              xlabel="局数（log）", ylabel="玩家密度")
    axes[0].legend(loc="upper right")

    # 右图：CDF（0~200 局覆盖绝大多数玩家）
    grid = np.linspace(0, 200, 201)
    for g in C.GROUPS:
        x = np.sort(df.loc[df.version == g, "sum_gamerounds"].to_numpy())
        cdf = np.searchsorted(x, grid, side="right") / len(x)
        axes[1].plot(grid, 100 * cdf, lw=2, label=C.GROUP_LABEL[g],
                     color=_color(g))
    _finalize(axes[1], title="局数经验 CDF（0–200 局区间）",
              xlabel="14 天内总游玩局数", ylabel="累计玩家 %")
    axes[1].legend(loc="lower right")
    return _save(fig, "04_rounds_distribution.png")


# 5. 局数均值差 bootstrap：全量 vs 剔除极端值
def plot_rounds_sensitivity(full: dict, trimmed: dict, rule) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), sharey=True)
    for ax, res, title in [
        (axes[0], full, "全量玩家"),
        (axes[1], trimmed, f"剔除 >{rule.outlier_cut:.0f} 局的极端玩家")]:
        d = res["boot_diff"]
        ax.hist(d, bins=60, color=C.PALETTE["violet"], alpha=0.75, zorder=3)
        ax.axvline(0, color=C.PALETTE["red"], lw=1.6)
        ax.axvline(res["ci_low"], color=C.INK["muted"], ls="--", lw=1.2)
        ax.axvline(res["ci_high"], color=C.INK["muted"], ls="--", lw=1.2)
        ax.text(0.97, 0.95,
                f"均值差 95% CI [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]\n"
                f"P(实验组更低) = {res['p_treatment_lower']:.1%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        _finalize(ax, title=title, xlabel="实验组 − 对照组（局/人）")
    axes[0].set_ylabel("bootstrap 频次")
    return _save(fig, "05_rounds_sensitivity.png")


# 6. 贝叶斯后验
def plot_bayes(bayes: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    lo = min(bayes["posterior_control"].min(),
             bayes["posterior_treatment"].min())
    hi = max(bayes["posterior_control"].max(),
             bayes["posterior_treatment"].max())
    grid = np.linspace(lo, hi, 400)
    from scipy import stats
    # 从蒙特卡洛样本恢复 Beta 参数用于画密度（矩估计）
    for g in C.GROUPS:
        s = bayes[f"posterior_{'control' if g == C.CONTROL else 'treatment'}"]
        m, v = s.mean(), s.var(ddof=1)
        a = m * (m * (1 - m) / v - 1)
        b = (1 - m) * (m * (1 - m) / v - 1)
        axes[0].plot(grid, stats.beta.pdf(grid, a, b), lw=2,
                     label=C.GROUP_LABEL[g], color=_color(g))
        axes[0].fill_between(grid, stats.beta.pdf(grid, a, b), alpha=0.12,
                             color=_color(g))
    _finalize(axes[0], title="7 日留存率后验（Jeffreys 先验）",
              xlabel="7 日留存率", ylabel="后验密度")
    axes[0].legend(loc="upper right")

    d = bayes["post_diff_pp"]
    axes[1].hist(d, bins=60, color=C.PALETTE["blue"], alpha=0.75, zorder=3)
    axes[1].axvline(0, color=C.PALETTE["red"], lw=1.6)
    axes[1].text(0.97, 0.95,
                 f"95% 可信区间 [{bayes['ci_low_pp']:+.2f}, "
                 f"{bayes['ci_high_pp']:+.2f}] pp\n"
                 f"后验 P(实验组更低) = {bayes['p_treatment_lower']:.1%}",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=9)
    _finalize(axes[1], title="后验差值分布", xlabel="实验组 − 对照组（百分点）")
    return _save(fig, "06_bayes_posterior.png")


# 7. 分层留存
def plot_tier_retention(tier_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    x = np.arange(len(C.TIER_ORDER))
    width = 0.34
    for ax, metric, title in [(axes[0], "retention_1", "次日留存"),
                              (axes[1], "retention_7", "7 日留存")]:
        for i, g in enumerate(C.GROUPS):
            sub = tier_df[tier_df.version == g].set_index("tier")
            vals = [sub.loc[t, metric] if t in sub.index else np.nan
                    for t in C.TIER_ORDER]
            ax.bar(x + (i - 0.5) * width, vals, width,
                   label=C.GROUP_LABEL[g], color=_color(g), zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(C.TIER_ORDER, rotation=18, ha="right")
        _finalize(ax, title=title, ylabel="留存率 %")
    axes[0].legend(loc="upper left")
    fig.suptitle("按参与度分层的留存（分层变量是处理后行为，只作描述性佐证）",
                 fontsize=11, color=C.INK["secondary"], y=1.03)
    return _save(fig, "07_tier_retention.png")


# 8. 参与度集中度
def plot_concentration(lorenz: pd.DataFrame, gini: float,
                       sizes: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    axes[0].plot(lorenz.cum_pop * 100, lorenz.cum_rounds * 100,
                 lw=2, color=C.PALETTE["blue"], label="洛伦兹曲线")
    axes[0].plot([0, 100], [0, 100], ls="--", lw=1.2,
                 color=C.INK["muted"], label="绝对平均线")
    axes[0].text(0.04, 0.80, f"Gini = {gini:.3f}",
                 transform=axes[0].transAxes, fontsize=11,
                 fontweight="bold")
    _finalize(axes[0], title="游玩局数集中度：头部玩家贡献绝大部分游玩时长",
              xlabel="累计玩家 %（按局数升序）", ylabel="累计局数 %")
    axes[0].legend(loc="upper left")

    x = np.arange(len(sizes))
    axes[1].bar(x - 0.2, sizes["n_pct"], 0.4, label="人数占比",
                color=C.PALETTE["aqua"], zorder=3)
    axes[1].bar(x + 0.2, sizes["rounds_pct"], 0.4, label="局数占比",
                color=C.PALETTE["orange"], zorder=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sizes["tier"], rotation=18, ha="right")
    _finalize(axes[1], title="参与度分层：人数 vs 局数贡献", ylabel="%")
    axes[1].legend(loc="upper left")
    return _save(fig, "08_concentration.png")


# 9. 功效 / MDE
def plot_power(curve: pd.DataFrame, n_per_arm: int, mde_now: float) -> str:
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    ax.plot(curve["n_per_arm"], curve["mde_pp"], lw=2,
            color=C.PALETTE["blue"])
    ax.axvline(n_per_arm, color=C.PALETTE["red"], ls="--", lw=1.4)
    ax.scatter([n_per_arm], [mde_now], color=C.PALETTE["red"], zorder=4)
    ax.annotate(f"当前样本量\n单组 {n_per_arm:,} 人\nMDE ≈ {mde_now:.2f} pp",
                (n_per_arm, mde_now), xytext=(-160, 26),
                textcoords="offset points", fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=C.INK["muted"]))
    _finalize(ax, title="实验灵敏度：以 D7 留存 19% 为基线，当前可检出的最小效应",
              xlabel="单组样本量", ylabel="可检出最小效应 MDE（百分点，80% power）")
    return _save(fig, "09_power_mde.png")


# 10. 漏斗
def plot_funnel(funnel_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    stages = ["安装", "次日留存", "7日留存"]
    x = np.arange(3)
    width = 0.34
    for i, g in enumerate(C.GROUPS):
        sub = funnel_df[funnel_df.group == g].set_index("stage")
        vals = [sub.loc[s, "pct_of_install"] for s in stages]
        ns = [sub.loc[s, "n"] for s in stages]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=C.GROUP_LABEL[g], color=_color(g), zorder=3)
        for rect, v, n in zip(bars, vals, ns):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.2,
                    f"{v:.1f}%\n({n:,})", ha="center", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylim(0, 108)
    _finalize(ax, title="新增玩家留存漏斗：门禁后移在 D1、D7 均有流失放大",
              ylabel="占安装人数 %")
    ax.legend(loc="upper right")
    return _save(fig, "10_funnel.png")
