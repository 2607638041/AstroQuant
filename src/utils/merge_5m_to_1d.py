#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_5m_to_1d.py

从 project_root/data/merged/btc/btc_5m 读取已合并的 5 分钟 parquet 文件
按日期聚合为日线（1D），保留玄学列（当天最后非空值）。
输出到 project_root/data/merged/btc/btc_1d，保持源目录结构。
输出只保留 date 列（字符串形式），不保留 datetime。
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

SCRIPT_PATH = Path(__file__).resolve()
ROOT_DIR = SCRIPT_PATH.parents[2]
SRC_DIR = ROOT_DIR / "data" / "merged" / "btc" / "btc_5m"
DST_DIR = ROOT_DIR / "data" / "merged" / "btc" / "btc_1d"
DST_DIR.mkdir(parents=True, exist_ok=True)

OHLC = ["open", "high", "low", "close"]

def last_nonnull(series):
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.iloc[-1]

def aggregate_daily_str(df):
    if df.empty:
        return pd.DataFrame()

    if "datetime" not in df.columns:
        raise ValueError("缺少 'datetime' 列")

    df = df.copy()
    # 提取日期字符串 YYYY-MM-DD 用于 groupby
    df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame()

    agg_dict = {}
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

    # 其他列按最后非空值
    extras = [c for c in df.columns if c not in list(agg_dict.keys()) + ["datetime", "date"]]
    for c in extras:
        agg_dict[c] = lambda s, _c=c: last_nonnull(s)

    grouped = df.groupby("date", as_index=False).agg(agg_dict)

    # 列顺序整理
    ohlc_cols = [c for c in OHLC if c in grouped.columns]
    extras_after = [c for c in grouped.columns if c not in ["date"] + ohlc_cols]
    ordered = ["date"] + ohlc_cols + extras_after
    grouped = grouped[ordered]

    return grouped

def process_file(in_path: Path, out_path: Path):
    try:
        df = pd.read_parquet(in_path)
    except Exception as e:
        print(f"[warn] 读取文件失败: {in_path} -> {e}")
        return False

    if "datetime" not in df.columns:
        print(f"[warn] 文件缺少 'datetime' 列，跳过: {in_path}")
        return False

    df = df.copy()

    try:
        daily = aggregate_daily_str(df)
    except Exception as e:
        print(f"[error] 聚合失败 {in_path}: {e}")
        traceback.print_exc()
        return False

    if daily.empty:
        print(f"[info] 聚合后无数据，跳过: {in_path}")
        return False

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
