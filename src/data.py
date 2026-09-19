"""数据接入、校验与清洗。

原始表只有 5 列、9 万行，但"小数据"更要把数据质量关：
- 模式校验（列名/取值域/类型）；
- 主键重复与缺失；
- 布尔列以 True/False 字符串存储，统一转 0/1；
- 极端值识别（sum_gamerounds 存在上万局的玩家）：
  不做静默删除，主分析保留全量并增加"剔除极端值"的敏感性分析；
- 分流比例检查（SRM）见 ab_test 模块。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

REQUIRED_COLUMNS = ["userid", "version", "sum_gamerounds",
                    "retention_1", "retention_7"]

# 参与度分层：在"至少玩过 1 局"的玩家中按全样本分位数切
TIER_BOUNDS = (0.50, 0.90, 0.99)
# 极端值敏感性阈值：p99.9 以上的玩家单独剔除再算一遍
OUTLIER_Q = 0.999


def raw_sha256(path=C.RAW_CSV) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class TierRule:
    """参与度分层阈值，由全样本分位数估计后随结果落库，保证可复现。"""
    p50: float
    p90: float
    p99: float
    outlier_cut: float

    def assign(self, rounds: pd.Series) -> pd.Series:
        r = rounds.to_numpy()
        tier = np.empty(len(r), dtype=object)
        tier[r == 0] = "安装未玩"
        tier[(r > 0) & (r <= self.p50)] = "轻度玩家"
        tier[(r > self.p50) & (r <= self.p90)] = "中度玩家"
        tier[(r > self.p90) & (r <= self.p99)] = "重度玩家"
        tier[r > self.p99] = "超核玩家"
        return pd.Series(tier, index=rounds.index)


def load_raw(path=C.RAW_CSV, verify_hash: bool = True) -> pd.DataFrame:
    """读入原始 CSV，做模式/取值域校验并规范类型。"""
    if verify_hash:
        digest = raw_sha256(path)
        if digest != C.RAW_SHA256:
            raise ValueError(
                f"原始文件校验失败：期望 {C.RAW_SHA256[:12]}…，实际 {digest[:12]}…。"
                "请删除文件后用 `python -m src.download_data` 重新获取。")

    df = pd.read_csv(path)
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"列模式不匹配：{list(df.columns)}")
    if df["userid"].duplicated().any():
        n = int(df["userid"].duplicated().sum())
        raise ValueError(f"userid 存在 {n} 条重复")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("存在缺失值")

    bad_ver = set(df["version"].unique()) - set(C.GROUPS)
    if bad_ver:
        raise ValueError(f"version 出现未知取值：{bad_ver}")
    if (df["sum_gamerounds"] < 0).any():
        raise ValueError("sum_gamerounds 存在负数")

    # 源文件里布尔列是 True/False 字符串
    for col in ("retention_1", "retention_7"):
        if df[col].dtype == object:
            df[col] = df[col].map({"True": 1, "False": 0})
    df["retention_1"] = df["retention_1"].astype("int8")
    df["retention_7"] = df["retention_7"].astype("int8")
    df["sum_gamerounds"] = df["sum_gamerounds"].astype("int64")
    return df


def build_tier_rule(df: pd.DataFrame) -> TierRule:
    played = df.loc[df["sum_gamerounds"] > 0, "sum_gamerounds"]
    p50, p90, p99 = played.quantile(list(TIER_BOUNDS))
    return TierRule(p50=float(p50), p90=float(p90), p99=float(p99),
                    outlier_cut=float(df["sum_gamerounds"].quantile(OUTLIER_Q)))


def prepare(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, TierRule]:
    """完整清洗：校验 + 分层列 + 漏斗标记，返回 (df, rule)。"""
    df = load_raw() if df is None else df.copy()
    rule = build_tier_rule(df)
    df["tier"] = rule.assign(df["sum_gamerounds"])
    # 安装 → 次日留存 → 7 日留存 的漏斗标记
    df["funnel_install"] = 1
    df["funnel_d1"] = df["retention_1"]
    df["funnel_d7"] = df["retention_7"]
    return df, rule
