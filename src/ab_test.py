"""实验统计核心：指标、假设检验、bootstrap、贝叶斯后验与功效计算。

所有函数均为纯函数，便于 tests/ 用手工构造的小样本校验。
方法约定：
- 率值指标（留存）：主检验用两比例 z 检验（合并标准误），
  置信区间用非合并 Wald 区间；同时给出 bootstrap 分布做稳健性印证；
- 连续指标（局数）：分布极度重尾且离散，用 Mann-Whitney U 秩检验，
  均值差异只通过 bootstrap 推断，并附"剔除极端值"敏感性分析；
- 贝叶斯侧：Jeffreys 先验 Beta(1/2,1/2) + 蒙特卡洛后验；
- 分流健康度：SRM 卡方检验（期望 1:1）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

Z975 = stats.norm.ppf(0.975)


# ----------------------------------------------------------------- 描述统计
def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """各组规模、参与度、留存的描述统计。"""
    rows = []
    for g in C.GROUPS:
        d = df[df.version == g]
        n = len(d)
        rounds = d["sum_gamerounds"]
        rows.append({
            "group": g,
            "n_users": n,
            "share_pct": 100 * n / len(df),
            "zero_round_pct": 100 * (rounds == 0).mean(),
            "rounds_mean": rounds.mean(),
            "rounds_median": rounds.median(),
            "rounds_p90": rounds.quantile(0.90),
            "rounds_p99": rounds.quantile(0.99),
            "rounds_max": rounds.max(),
            "retention_1_pct": 100 * d["retention_1"].mean(),
            "retention_7_pct": 100 * d["retention_7"].mean(),
            "retained1_not7_pct": 100 * ((d.retention_1 == 1) &
                                         (d.retention_7 == 0)).mean(),
        })
    return pd.DataFrame(rows).set_index("group")


# ----------------------------------------------------------------- SRM
# SRM 判定双门槛：p < 0.005（大样本下比教科书 1% 更稳健的护栏阈值），
# 或实际分流偏差超过 1 个百分点（业务可接受阈值）。单看 p 值会在超大样本下
# 把 0.x pp 的哈希抖动误报为事故。
SRM_P_ALPHA = 0.005
SRM_REL_TOL_PP = 1.0


def srm_test(df: pd.DataFrame, expected_ratio: float = 0.5) -> dict:
    """样本比例失衡检验（卡方拟合优度，双侧），返回分级结论。"""
    counts = df["version"].value_counts().reindex(C.GROUPS).to_numpy()
    n = counts.sum()
    expected = np.array([expected_ratio, 1 - expected_ratio]) * n
    chi2 = ((counts - expected) ** 2 / expected).sum()
    p = stats.chi2.sf(chi2, df=1)
    ratio_c = counts[0] / n
    diff_pp = abs(ratio_c - expected_ratio) * 100
    blocking = (p < SRM_P_ALPHA) or (diff_pp > SRM_REL_TOL_PP)
    warning = (p < 0.05) and not blocking
    return {"n_control": int(counts[0]), "n_treatment": int(counts[1]),
            "expected_ratio": expected_ratio,
            "observed_ratio": float(ratio_c),
            "diff_pp": float(diff_pp),
            "chi2": float(chi2), "p_value": float(p),
            "is_srm": blocking,          # 阻断级：实验结论不可用
            "warning": warning}          # 预警级：记录根因，不阻断推断


# ----------------------------------------------------------------- 率值检验
def two_proportion_test(s_c: int, n_c: int, s_t: int, n_t: int) -> dict:
    """两比例 z 检验（合并 SE）+ Wald 95% CI（非合并 SE）。"""
    p_c, p_t = s_c / n_c, s_t / n_t
    p_pool = (s_c + s_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pool
    p_value = 2 * stats.norm.sf(abs(z))

    se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    diff = p_t - p_c
    return {
        "rate_control": p_c, "rate_treatment": p_t,
        "diff_pp": 100 * diff,                       # 绝对差，百分点
        "relative_lift_pct": 100 * (p_t - p_c) / p_c,
        "ci_low_pp": 100 * (diff - Z975 * se_diff),
        "ci_high_pp": 100 * (diff + Z975 * se_diff),
        "z": float(z), "p_value": float(p_value),
    }


def bootstrap_rate(df: pd.DataFrame, col: str, n_boot: int = C.N_BOOTSTRAP,
                   seed: int = C.RANDOM_SEED) -> dict:
    """留存率差的 bootstrap：二项抽样与重采样伯努利样本同分布，速度快 3 个量级。"""
    rng = np.random.default_rng(seed)
    out = {}
    for g in C.GROUPS:
        d = df.loc[df.version == g, col]
        n, s = len(d), int(d.sum())
        out[g] = rng.binomial(n, s / n, size=n_boot) / n
    diff = 100 * (out[C.TREATMENT] - out[C.CONTROL])
    return {"boot_diff_pp": diff,
            "ci_low_pp": float(np.quantile(diff, 0.025)),
            "ci_high_pp": float(np.quantile(diff, 0.975)),
            "p_treatment_lower": float((diff < 0).mean())}


# ----------------------------------------------------------------- 连续指标
def mannwhitney_rounds(df: pd.DataFrame) -> dict:
    """两组局数分布的 Mann-Whitney U 双侧检验。"""
    x = df.loc[df.version == C.CONTROL, "sum_gamerounds"]
    y = df.loc[df.version == C.TREATMENT, "sum_gamerounds"]
    u, p = stats.mannwhitneyu(y, x, alternative="two-sided")
    # rank-biserial correlation：U 口径的效应量
    rbc = 2 * u / (len(x) * len(y)) - 1
    return {"U": float(u), "p_value": float(p),
            "rank_biserial": float(rbc)}


def bootstrap_rounds(df: pd.DataFrame, n_boot: int = C.N_BOOTSTRAP,
                     seed: int = C.RANDOM_SEED,
                     chunk: int = 500, cutoff: int | None = None) -> dict:
    """两组局数均值差的非参数 bootstrap（分块控制内存）。

    cutoff 给定时剔除局数 > cutoff 的玩家，用于极端值敏感性分析。
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for g in C.GROUPS:
        x = df.loc[df.version == g, "sum_gamerounds"].to_numpy(dtype=float)
        if cutoff is not None:
            x = x[x <= cutoff]
        n = len(x)
        means = np.empty(n_boot)
        for i in range(0, n_boot, chunk):
            b = min(chunk, n_boot - i)
            idx = rng.integers(0, n, size=(b, n))
            means[i:i + b] = x[idx].mean(axis=1)
        samples[g] = means
    diff = samples[C.TREATMENT] - samples[C.CONTROL]
    return {"mean_control": float(samples[C.CONTROL].mean()),
            "mean_treatment": float(samples[C.TREATMENT].mean()),
            "boot_diff": diff,
            "ci_low": float(np.quantile(diff, 0.025)),
            "ci_high": float(np.quantile(diff, 0.975)),
            "p_treatment_lower": float((diff < 0).mean()),
            "n_control": int((df.version == C.CONTROL).sum()),
            "n_treatment": int((df.version == C.TREATMENT).sum())}


# ----------------------------------------------------------------- 贝叶斯
def bayes_retention(df: pd.DataFrame, col: str,
                    prior_a: float = 0.5, prior_b: float = 0.5,
                    n_mc: int = 200_000, seed: int = C.RANDOM_SEED) -> dict:
    """Jeffreys Beta 先验下的两组留存率后验与后验差值（蒙特卡洛）。"""
    rng = np.random.default_rng(seed)
    post = {}
    for g in C.GROUPS:
        d = df.loc[df.version == g, col]
        a = prior_a + int(d.sum())
        b = prior_b + len(d) - int(d.sum())
        post[g] = rng.beta(a, b, size=n_mc)
    diff = 100 * (post[C.TREATMENT] - post[C.CONTROL])
    return {"posterior_control": post[C.CONTROL],
            "posterior_treatment": post[C.TREATMENT],
            "post_diff_pp": diff,
            "ci_low_pp": float(np.quantile(diff, 0.025)),
            "ci_high_pp": float(np.quantile(diff, 0.975)),
            "p_treatment_lower": float((diff < 0).mean())}


# ----------------------------------------------------------------- 功效
def required_n_per_arm(p_baseline: float, mde_pp: float,
                       alpha: float = 0.05, power: float = 0.8) -> float:
    """两比例等组检验，检出 mde（百分点）所需的单组样本量。"""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p1, p2 = p_baseline, p_baseline + mde_pp / 100
    pbar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar))
           + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return num / (mde_pp / 100) ** 2


def mde_curve(p_baseline: float, n_per_arm: int,
              n_grid: int = 60) -> pd.DataFrame:
    """给定当前样本量，不同 n 下可检出的最小效应 MDE（80% power）。"""
    ns = np.linspace(2_000, n_per_arm, n_grid)
    mdes = [_solve_mde(p_baseline, n, 0.05, 0.8) for n in ns]
    return pd.DataFrame({"n_per_arm": ns.astype(int), "mde_pp": mdes})


def _solve_mde(p: float, n: int, alpha: float, power: float) -> float:
    """反解单组样本量 n 对应的 MDE（百分点），二分法。"""
    def power_at(delta_pp: float) -> float:
        req = required_n_per_arm(p, delta_pp, alpha, power)
        return n - req
    lo, hi = 0.05, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_at(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ----------------------------------------------------------------- 分层与漏斗
def tier_table(df: pd.DataFrame) -> pd.DataFrame:
    """参与度分层 × 实验组：人数、局数、D1/D7 留存。"""
    g = (df.groupby(["tier", "version"], observed=True)
           .agg(n=("userid", "size"),
                rounds_mean=("sum_gamerounds", "mean"),
                retention_1=("retention_1", "mean"),
                retention_7=("retention_7", "mean"))
           .reset_index())
    g["retention_1"] *= 100
    g["retention_7"] *= 100
    return g


def tier_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """各分层人数与局数集中度（全样本）。"""
    t = (df.groupby("tier", observed=True)
           .agg(n=("userid", "size"),
                total_rounds=("sum_gamerounds", "sum"))
           .reindex(C.TIER_ORDER).dropna())
    t["n_pct"] = 100 * t["n"] / t["n"].sum()
    t["rounds_pct"] = 100 * t["total_rounds"] / t["total_rounds"].sum()
    return t.reset_index()


def funnel(df: pd.DataFrame) -> pd.DataFrame:
    """安装 → D1 → D7 漏斗（分组转化率）。"""
    rows = []
    for gname in C.GROUPS:
        d = df[df.version == gname]
        n = len(d)
        d1 = int(d["retention_1"].sum())
        d7 = int(d["retention_7"].sum())
        rows.append({"group": gname, "stage": "安装", "n": n,
                     "pct_of_install": 100.0})
        rows.append({"group": gname, "stage": "次日留存", "n": d1,
                     "pct_of_install": 100 * d1 / n})
        rows.append({"group": gname, "stage": "7日留存", "n": d7,
                     "pct_of_install": 100 * d7 / n})
    return pd.DataFrame(rows)


def rounds_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """游玩局数的洛伦兹曲线数据（参与度集中度）。"""
    x = np.sort(df["sum_gamerounds"].to_numpy(dtype=float))
    n = len(x)
    # 补上原点 (0,0)，否则离散梯形积分对 Gini 有约 1/n 的系统性偏差
    cum_pop = np.r_[0.0, np.arange(1, n + 1) / n]
    cum_rounds = np.r_[0.0, np.cumsum(x) / x.sum()]
    lorenz = pd.DataFrame({"cum_pop": cum_pop, "cum_rounds": cum_rounds})
    gini = float(1 - 2 * np.trapz(cum_rounds, cum_pop))
    return lorenz, gini
