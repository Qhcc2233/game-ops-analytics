"""重新下载真实数据集并校验 sha256。

    python -m src.download_data

数据来源：Tactile Entertainment 旗下手游 Cookie Cats 的线上 A/B 实验数据，
经 Kaggle 公开发布（Mobile Games A/B Testing with Cookie Cats）。
仓库随附的副本已通过 sha256 校验，通常不需要重新下载。
"""
from __future__ import annotations

import urllib.request

from . import config as C
from .data import raw_sha256


def main() -> None:
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"下载：{C.DATASET_MIRROR}")
    urllib.request.urlretrieve(C.DATASET_MIRROR, C.RAW_CSV)
    digest = raw_sha256()
    if digest != C.RAW_SHA256:
        raise SystemExit(
            f"校验失败：期望 {C.RAW_SHA256}\n实际 {digest}\n"
            "镜像文件可能已变更，请改用 Kaggle 原始来源：" + C.DATASET_SOURCE)
    print(f"完成：{C.RAW_CSV}（sha256 校验通过）")


if __name__ == "__main__":
    main()
