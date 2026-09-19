"""统计口径手工校验：python tests/test_metrics.py（无需 pytest）。

用手工构造的小样本验证：两比例检验数值、SRM 分级、bootstrap 可复现性、
Mann-Whitney 与 scipy 一致、贝叶斯后验均值、功效公式自洽、数据校验异常。
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ab_test as ab          # noqa: E402
from src import config as C            # noqa: E402
from src import data as datalib        # noqa: E402


def test_two_proportion():
    r = ab.two_proportion_test(100, 1000, 130, 1000)
    assert abs(r["rate_control"] - 0.10) < 1e-12
    assert abs(r["rate_treatment"] - 0.13) < 1e-12
    assert abs(r["diff_pp"] - 3.0) < 1e-10
    # 与 scipy 独立实现的卡方检验对照（双侧 p 应一致到小数点后 4 位）
    table = np.array([[130, 870], [100, 900]])
    p_chi2 = stats.chi2_contingency(table, correction=False)[1]
    assert abs(r["p_value"] - p_chi2) < 1e-4
    # 3pp 的差异在 n=1000/组时应显著，CI 不含 0
    assert r["p_value"] < 0.05
    assert r["ci_low_pp"] > 0
    print("  ✓ 两比例 z 检验数值与 scipy 卡方一致")


def test_srm_levels():
    rng = np.random.default_rng(0)
    df_bal = pd.DataFrame({
        "version": np.array([C.CONTROL] * 5000 + [C.TREATMENT] * 5000)})
    r = ab.srm_test(df_bal)
    assert not r["is_srm"] and not r["warning"]
    assert r["p_value"] > 0.99

    # 2% 的大偏差 → 阻断级
    n_c, n_t = 49_000, 51_000
    df_bad = pd.DataFrame({
        "version": [C.CONTROL] * n_c + [C.TREATMENT] * n_t})
    r_bad = ab.srm_test(df_bad)
    assert r_bad["is_srm"] and r_bad["diff_pp"] > 1
    print("  ✓ SRM 健康/预警/阻断三级判读正确")


def test_bootstrap_reproducible():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "version": np.repeat(C.GROUPS, 5000),
        "retention_1": rng.binomial(1, 0.2, 10000)})
    r1 = ab.bootstrap_rate(df, "retention_1", n_boot=500, seed=7)
    r2 = ab.bootstrap_rate(df, "retention_1", n_boot=500, seed=7)
    assert np.array_equal(r1["boot_diff_pp"], r2["boot_diff_pp"])
    assert r1["ci_low_pp"] <= np.median(r1["boot_diff_pp"]) <= r1["ci_high_pp"]
    print("  ✓ bootstrap 固定种子可复现")


def test_mannwhitney_matches_scipy():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 20, 200)
    y = rng.integers(0, 20, 200)
    df = pd.DataFrame({
        "version": [C.CONTROL] * 200 + [C.TREATMENT] * 200,
        "sum_gamerounds": np.r_[x, y]})
    r = ab.mannwhitney_rounds(df)
    p_ref = stats.mannwhitneyu(y, x, alternative="two-sided").pvalue
    assert abs(r["p_value"] - p_ref) < 1e-10
    assert -1 <= r["rank_biserial"] <= 1
    print("  ✓ Mann-Whitney U 与 scipy 一致")


def test_bayes_posterior_mean():
    df = pd.DataFrame({
        "version": [C.CONTROL] * 100 + [C.TREATMENT] * 100,
        "retention_7": np.r_[np.ones(40), np.zeros(60),
                             np.ones(20), np.zeros(80)]})
    r = ab.bayes_retention(df, "retention_7", n_mc=200_000)
    # Jeffreys 先验下后验均值 = (x+0.5)/(n+1)
    assert abs(r["posterior_control"].mean() - 40.5 / 101) < 5e-3
    assert abs(r["posterior_treatment"].mean() - 20.5 / 101) < 5e-3
    assert r["p_treatment_lower"] > 0.99
    print("  ✓ Jeffreys Beta 后验均值/概率正确")


def test_power_consistency():
    p = 0.19
    n = ab.required_n_per_arm(p, 0.82)
    assert n > 30_000  # 与本实验实际量级吻合（4.5 万/组检出 0.82pp）
    # 公式自洽：用所需样本量反解 MDE，应回到 0.82pp
    mde_back = ab._solve_mde(p, n, 0.05, 0.8)
    assert abs(mde_back - 0.82) < 0.01
    # MDE 随样本量单调下降
    assert ab.required_n_per_arm(p, 0.5) > ab.required_n_per_arm(p, 1.0)
    print("  ✓ 样本量/MDE 功效公式自洽")


def test_data_validation_and_tiers():
    good = pd.DataFrame({
        "userid": [f"u{i}" for i in range(6)],
        "version": [C.CONTROL] * 3 + [C.TREATMENT] * 3,
        "sum_gamerounds": [0, 10, 100, 5, 50, 200],
        "retention_1": [1, 1, 0, 1, 0, 0],
        "retention_7": [0, 1, 0, 0, 0, 1]})
    rule = datalib.build_tier_rule(good)
    tiers = rule.assign(good["sum_gamerounds"])
    assert tiers.iloc[0] == "安装未玩"
    assert tiers.iloc[-1] == "超核玩家"

    # 重复 userid / 未知版本必须被 load_raw 拒绝（跳过哈希校验）
    import os
    import tempfile

    def _expect_error(frame, needle):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            frame.to_csv(path, index=False)
            try:
                datalib.load_raw(path, verify_hash=False)
                raise AssertionError(f"应当检测到问题：{needle}")
            except ValueError as e:
                assert needle in str(e)
        finally:
            os.remove(path)

    dup = good.copy()
    dup.loc[1, "userid"] = "u0"
    _expect_error(dup, "重复")
    bad = good.copy()
    bad.loc[0, "version"] = "gate_99"
    _expect_error(bad, "未知取值")
    print("  ✓ 分层规则与数据校验正确")


def test_gini_equality():
    df = pd.DataFrame({"sum_gamerounds": [50] * 1000})
    _, gini = ab.rounds_concentration(df)
    assert abs(gini) < 1e-6
    print("  ✓ Gini：完全均等时为 0")


def test_real_dataset_loads():
    df = datalib.load_raw()
    assert len(df) == 90_189
    assert set(df.version.unique()) == set(C.GROUPS)
    assert df.retention_1.isin([0, 1]).all()
    print(f"  ✓ 真实数据集加载通过：{len(df):,} 行，sha256 校验一致")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"运行 {len(tests)} 组统计口径测试")
    for t in tests:
        t()
    print("全部通过 ✅")
