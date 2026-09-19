"""全局配置：路径、实验元数据、可视化色板。

实验背景（真实数据集 Cookie Cats）：
- Cookie Cats 是 Tactile Entertainment 发行的三消手游；
- 游戏中"关卡门禁(Gate)"会在关卡之间强制暂停（等冷却或求助好友），
  是留存调节与付费点设计的一部分；
- 实验组把门禁从第 30 关后移到第 40 关：
  对照组 gate_30，实验组 gate_40；
- 每名玩家以安装后 14 天为观察窗，记录总游玩局数 sum_gamerounds、
  次日留存 retention_1、7 日留存 retention_7。

色板取自经过色盲安全校验的参考设计系统（OKLab CVD ΔE >= 8），
分类色按固定槽位使用，颜色跟随实体，不随排名变化。
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
FIG_DIR = ROOT_DIR / "reports" / "figures"
REPORT_DIR = ROOT_DIR / "reports"

for _d in (DATA_RAW, DATA_PROCESSED, FIG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_RAW / "cookie_cats.csv"

# ---------------------------------------------------------------- 实验元数据
CONTROL = "gate_30"      # 门禁在第 30 关（对照/旧策略）
TREATMENT = "gate_40"    # 门禁后移到第 40 关（实验/新策略）
GROUPS = [CONTROL, TREATMENT]
GROUP_LABEL = {CONTROL: "对照组 gate_30", TREATMENT: "实验组 gate_40"}

RANDOM_SEED = 20260919
N_BOOTSTRAP = 10_000     # 流水线 bootstrap 次数（看板里默认更小）

# 数据来源（可复现性）：原始数据为 Kaggle 公开发布的 Tactile Entertainment
# 游戏真实 A/B 数据，仓库内随附一份经 sha256 校验的完整副本。
DATASET_SOURCE = "https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats"
DATASET_MIRROR = ("https://raw.githubusercontent.com/"
                  "ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats/"
                  "master/cookie_cats.csv")
RAW_SHA256 = "5ab54d761fbddcd50de7b88e4eaf7837cba4569474f50c043a4d17ee342c46bd"

# ---------------------------------------------------------------- 可视化
PALETTE = {
    "blue":   "#2a78d6",
    "orange": "#eb6834",
    "aqua":   "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green":  "#008300",
    "violet": "#4a3aa7",
    "red":    "#e34948",
}
# 对照 = 蓝（基线色），实验 = 橙（变化色），全项目固定
GROUP_COLORS = {CONTROL: PALETTE["blue"], TREATMENT: PALETTE["orange"]}

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#104281"]

INK = {
    "surface":   "#fcfcfb",
    "primary":   "#0b0b0b",
    "secondary": "#52514e",
    "muted":     "#898781",
    "grid":      "#e1e0d9",
    "baseline":  "#c3c2b7",
    "good":      "#006300",
    "bad":       "#d03b3b",
}

# 参与度分层（按 sum_gamerounds 的全样本分位数，阈值由数据计算后落库）
TIER_ORDER = ["安装未玩", "轻度玩家", "中度玩家", "重度玩家", "超核玩家"]
TIER_COLORS = {
    "安装未玩": PALETTE["magenta"],
    "轻度玩家": PALETTE["aqua"],
    "中度玩家": PALETTE["blue"],
    "重度玩家": PALETTE["yellow"],
    "超核玩家": PALETTE["red"],
}
