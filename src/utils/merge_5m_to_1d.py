#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_5m_to_1d.py

从 project_root/data/merged/btc/btc_5m 读取已合并的 5 分钟 parquet 文件（含 datetime (UTC) + open/high/low/close [+volume] + 玄学列）
按中国时区（UTC+8）重采样为日线（1D），保留玄学列（当天最后非空值）。
输出到 project_root/data/merged/btc/btc_1d_cn，保留源的相对目录结构。
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# -----------------------
# 路径（基于脚本位置定位项目根）
# -----------------------
SCRIPT_PATH = Path(__file__).resolve()
ROOT_DIR = SCRIPT_PATH.parents[2]   # 假定脚本放在 src/utils/... -> ../.. 回到项目根
SRC_DIR = ROOT_DIR / "data" / "merged" / "btc" / "btc_5m"
DST_DIR = ROOT_DIR / "data" / "merged" / "btc" / "btc_1d_cn"
DST_DIR.mkdir(parents=True, exist_ok=True)

# 常用 K 线列名
OHLC = ["open", "high", "low", "close"]
VOLUME = "volume"

def last_nonnull(series):
    """返回 series 中最后一个非空值（如果全空返回 NaN）"""
    try:
        s = series.dropna()
        if s.empty:
            return np.nan
        return s.iloc[-1]
    except Exception:
        # 保险回退
        vals = series.values
        for v in reversed(vals):
            if pd.notna(v):
                return v
        return np.nan

def aggregate_daily(df):
    """
    输入 df，索引应为 tz-aware Asia/Shanghai 的 DatetimeIndex。
    返回按日聚合的 DataFrame（index = 日的午夜 Asia/Shanghai），并 reset_index。
    """
    if df.empty:
        return pd.DataFrame()

    # 确认索引为 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df 必须以 DatetimeIndex 为索引")

    # 确保索引为 Asia/Shanghai tz-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Shanghai")
    else:
        df.index = df.index.tz_convert("Asia/Shanghai")

    agg_dict = {}
    # OHLC
    for c in OHLC:
        if c in df.columns:
            if c == "open":
                agg_dict[c] = "first"
            elif c == "high":
                agg_dict[c] = "max"
            elif c == "low":
                agg_dict[c] = "min"
            elif c == "close":
                agg_dict[c] = "last"
    # volume
    if VOLUME in df.columns:
        agg_dict[VOLUME] = "sum"

    # extra columns -> last non-null
    extras = [c for c in df.columns if c not in agg_dict]
    for c in extras:
        agg_dict[c] = lambda s, _c=c: last_nonnull(s)

    # resample 按天聚合（resample 会以索引的 tz 为准）
    grouped = df.resample("1D").agg(agg_dict)

    # 删除整行都是 NaN（没有任何数据的日子）
    # 但通常如果 open 为 NaN 表示当天没有数据，可以删除
    if "open" in grouped.columns:
        grouped = grouped.dropna(subset=["open"], how="all")
    else:
        grouped = grouped.dropna(how="all")

    # reset_index 并整理列
    grouped = grouped.reset_index()  # index -> datetime index (tz-aware midnight)
    grouped = grouped.rename(columns={"index": "datetime_cn"}) if "index" in grouped.columns else grouped
    # pandas 已把 index 转为列名为索引名（通常为 'datetime'），但因为我们 resample 后 index 没命名，reset_index 给列名 'datetime'
    # 为通用处理，确保列名为 datetime_cn
    possible_idx_cols = [c for c in grouped.columns if isinstance(grouped[c].dtype, (pd.core.dtypes.dtypes.DatetimeTZDtype,)) or "datetime" in c.lower()]
    if "datetime_cn" not in grouped.columns:
        # 找第一个 datetime-like 列并 rename
        for c in grouped.columns:
            if pd.api.types.is_datetime64_any_dtype(grouped[c]) or "datetime" in c.lower():
                grouped = grouped.rename(columns={c: "datetime_cn"})
                break

    # 确保 datetime_cn 是 tz-aware Asia/Shanghai
    grouped["datetime_cn"] = pd.to_datetime(grouped["datetime_cn"])
    if grouped["datetime_cn"].dt.tz is None:
        grouped["datetime_cn"] = grouped["datetime_cn"].dt.tz_localize("Asia/Shanghai")
    else:
        grouped["datetime_cn"] = grouped["datetime_cn"].dt.tz_convert("Asia/Shanghai")

    # 添加 date_cn 字符串列（方便检索）
    grouped["date_cn"] = grouped["datetime_cn"].dt.strftime("%Y-%m-%d")

    # 整理列顺序：datetime_cn, date_cn, OHLC..., extras...
    ohlc_cols = [c for c in OHLC if c in grouped.columns]
    vol_cols = [VOLUME] if VOLUME in grouped.columns else []
    extras_after = [c for c in grouped.columns if c not in ["datetime_cn", "date_cn"] + ohlc_cols + vol_cols]
    ordered = ["datetime_cn", "date_cn"] + ohlc_cols + vol_cols + extras_after
    ordered = [c for c in ordered if c in grouped.columns]
    grouped = grouped[ordered]

    return grouped

def process_file(in_path: Path, out_path: Path):
    """读入单个 5m parquet，转时区、聚合为日线并保存"""
    try:
        df = pd.read_parquet(in_path)
    except Exception as e:
        print(f"[warn] 读取文件失败: {in_path} -> {e}")
        return False

    # 必须至少有 datetime 列
    if "datetime" not in df.columns:
        print(f"[warn] 文件缺少 'datetime' 列，跳过: {in_path}")
        return False

    df = df.copy()

    # 解析 datetime -> UTC
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if df["datetime"].isna().all():
        print(f"[warn] datetime 全解析失败，跳过: {in_path}")
        return False

    # 转为中国时区（tz-aware），放入 datetime_cn 列
    df["datetime_cn"] = df["datetime"].dt.tz_convert("Asia/Shanghai")

    # 将 datetime_cn 作为索引（避免保留原 datetime 列导致混淆）
    df = df.set_index(pd.DatetimeIndex(df["datetime_cn"]))
    # 删除列 datetime_cn（索引已包含），也可以删除原始 UTC datetime（用户要求不保留）
    if "datetime_cn" in df.columns:
        df = df.drop(columns=["datetime_cn"])
    if "datetime" in df.columns:
        df = df.drop(columns=["datetime"])

    # 聚合
    try:
        daily = aggregate_daily(df)
    except Exception as e:
        print(f"[error] 聚合失败 {in_path}: {e}")
        traceback.print_exc()
        return False

    if daily is None or daily.empty:
        print(f"[info] 聚合后无数据，跳过: {in_path}")
        return False

    # 保存（保持目录结构）
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        daily.to_parquet(out_path, index=False)
        print(f"[ok] {in_path.name} -> {out_path}  行: {len(daily)} 列: {len(daily.columns)}")
        return True
    except Exception as e:
        print(f"[error] 保存失败 {out_path}: {e}")
        traceback.print_exc()
        return False

def main():
    if not SRC_DIR.exists():
        print(f"[error] 源目录不存在: {SRC_DIR}")
        return

    files = sorted([p for p in SRC_DIR.rglob("*.parquet")])
    if not files:
        print(f"[warn] 源目录下没有 parquet 文件: {SRC_DIR}")
        return

    print(f"[info] 找到 {len(files)} 个 parquet 文件，开始逐个处理...")
    succeeded = 0
    for p in files:
        rel = p.relative_to(SRC_DIR)
        out_p = DST_DIR / rel
        ok = process_file(p, out_p)
        if ok:
            succeeded += 1

    print(f"[done] 处理完成：{succeeded}/{len(files)} 个文件成功 -> 输出目录: {DST_DIR}")

if __name__ == "__main__":
    main()