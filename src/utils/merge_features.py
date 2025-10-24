#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_features.py

功能：
- 将 BTC 5m 数据与 astro_data 的玄学特征合并
- 年文件输出到 MERGED_DIR 根（不建子文件夹）
- 月文件输出到 MERGED_DIR/<YEAR>/ 下
- 数据清洗、节气 ffill、按天对齐、保留 BTC datetime
- 无 tqdm，日志简洁
"""

import os
import re
import pandas as pd

# 参数设置
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))     # 工程根目录
BTC_DIR = os.path.join(ROOT_DIR, 'data', 'btc_data_5m')                             # BTC 数据目录
ASTRO_DIR = os.path.join(ROOT_DIR, 'data', 'astro_data')                            # 玄学数据目录
MERGED_DIR = os.path.join(ROOT_DIR, 'data', 'merged', 'btc', 'btc_5m')              # 合并后数据输出目录

# 玄学数据文件名
ASTRO_FILES = {
    '九星': '九星.parquet',
    '十二建星': '十二建星.parquet',
    '干支历': '干支历.parquet',
    '星宿': '星宿.parquet',
    '节气': '节气.parquet'
}

os.makedirs(MERGED_DIR, exist_ok=True)  # 创建输出目录

# 获取日期列名
def find_date_column(df: pd.DataFrame):
    for c in df.columns:
        if 'date' in c.lower() or '时间' in c or '日期' in c:
            return c
    return None

# 转换为 datetime 格式，容错性强
def safe_to_datetime(s, utc=True):
    try:
        return pd.to_datetime(s, errors='coerce', utc=utc)
    except Exception:
        if hasattr(s, '__len__'):
            return pd.Series([pd.NaT] * len(s))
        return pd.NaT

# 清洗 BTC 数据
def clean_btc_df(df: pd.DataFrame):
    df = df.copy()
    if 'datetime' not in df.columns:
        raise ValueError("BTC 文件缺少 datetime 列")
    df['datetime'] = safe_to_datetime(df['datetime'], utc=True)
    df = df.dropna(subset=['datetime']).drop_duplicates(subset=['datetime'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('datetime').reset_index(drop=True)

# 清洗合并后数据
def clean_merged_df(df: pd.DataFrame):
    """清洗合并后数据：保证 datetime 存在、数值列、节气 ffill、去重排序"""
    df = df.copy()
    if 'datetime' not in df.columns:
        raise ValueError("合并后缺少 datetime 列")
    df['datetime'] = safe_to_datetime(df['datetime'], utc=True)
    df = df.dropna(subset=['datetime'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if '节气' in df.columns:
        df['节气'] = df['节气'].ffill()
    df = df.drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    return df

# 转换 datetime 为字符串格式
def datetime_to_str(df: pd.DataFrame, fmt="%Y-%m-%d %H:%M"):
    """把 datetime 转为字符串格式（用于写文件）"""
    df = df.copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
        df['datetime'] = df['datetime'].dt.strftime(fmt)
    return df

# 加载玄学数据
def load_astro_master():
    print("开始加载玄学数据...")
    dfs = []
    try:
        _ = os.listdir(ASTRO_DIR)
    except Exception as e:
        print("无法访问 ASTRO_DIR:", e)
        return None

    for key, fname in ASTRO_FILES.items():
        path = os.path.join(ASTRO_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"读取玄学文件失败：{path}，跳过。错误：{e}")
            continue

        date_col = find_date_column(df)
        if not date_col:
            continue

        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
        df_vals = df.drop(columns=[date_col]).copy()
        df_vals = df_vals.dropna(subset=['datetime'])
        if df_vals.empty:
            continue

        df_vals = df_vals.set_index('datetime')
        dfs.append((key, df_vals))

    if not dfs:
        print("[错误] 未找到有效玄学文件")
        return None

    # 逐个 outer join，避免覆盖已有有效列
    master = dfs[0][1].copy()
    for key, dfv in dfs[1:]:
        conflict_cols = [c for c in dfv.columns if c in master.columns]
        for c in conflict_cols:
            if not master[c].isna().all():
                dfv = dfv.drop(columns=[c])
        master = master.join(dfv, how='outer')

    master = master.sort_index().reset_index()
    if 'datetime' not in master.columns:
        master = master.rename(columns={master.columns[0]: 'datetime'})
    master['datetime'] = pd.to_datetime(master['datetime'], errors='coerce', utc=True)
    master = master.dropna(subset=['datetime']).reset_index(drop=True)
    master['date'] = master['datetime'].dt.floor('D')

    if '节气' in master.columns:
        master['节气'] = master['节气'].ffill()

    print("玄学 master 构建完成，形状:", master.shape)
    return master

# 获取年份
def extract_year_from_filename(fname: str):
    m = re.search(r'(19|20)\d{2}', fname)
    if m:
        return m.group(0)
    return None

# 获取年月
def extract_year_month_from_filename(fname: str):
    m = re.search(r'((19|20)\d{2})[^\d]?([01]\d)', fname)
    if m:
        return m.group(1), m.group(3)
    y = extract_year_from_filename(fname)
    return (y, None) if y else (None, None)

# 合并数据
def merge_to_btc_df(btc_df: pd.DataFrame, astro_master: pd.DataFrame):
    btc_df = clean_btc_df(btc_df)
    btc_df['date'] = btc_df['datetime'].dt.floor('D')

    right = astro_master.sort_values('date').reset_index(drop=True)
    left = btc_df.sort_values('date').reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        on='date',
        direction='backward',
        suffixes=('', '_astro')
    )

    # 优先保留 BTC 的 datetime（更精确）
    merged['datetime'] = left['datetime']

    merged = clean_merged_df(merged)

    # 删除多余的 datetime_astro 列
    if 'datetime_astro' in merged.columns:
        merged = merged.drop(columns=['datetime_astro'])

    if 'date' in merged.columns:
        merged = merged.drop(columns=['date'])
    return merged

# 合并所有数据
def merge_all(datetime_fmt="%Y-%m-%d %H:%M"):
    astro_master = load_astro_master()
    if astro_master is None:
        print("[错误] 无有效玄学 master，退出")
        return

    if not os.path.exists(BTC_DIR):
        print(f"[错误] BTC 数据目录不存在: {BTC_DIR}")
        return

    items = sorted(os.listdir(BTC_DIR))

    # 1) 年文件
    year_files = [f for f in items if f.endswith('.parquet')]
    for yf in sorted(year_files):
        try:
            year_path = os.path.join(BTC_DIR, yf)
            btc_year = pd.read_parquet(year_path)
            merged = merge_to_btc_df(btc_year, astro_master)
            merged_out = datetime_to_str(merged, fmt=datetime_fmt)

            out_path = os.path.join(MERGED_DIR, yf)
            merged_out.to_parquet(out_path, index=False)

            year_str = extract_year_from_filename(yf)
            if year_str:
                print(f"{year_str} 年合并完成")
            else:
                print(f"{yf} 合并完成")
        except Exception as e:
            print(f"处理 {yf} 失败：{e}")

    # 2) 年目录下月文件
    year_dirs = [d for d in items if os.path.isdir(os.path.join(BTC_DIR, d))]
    for yd in sorted(year_dirs):
        item_path = os.path.join(BTC_DIR, yd)
        out_year_dir = os.path.join(MERGED_DIR, yd)
        os.makedirs(out_year_dir, exist_ok=True)

        month_files = sorted([f for f in os.listdir(item_path) if f.endswith('.parquet')])
        for mf in month_files:
            try:
                btc_month = pd.read_parquet(os.path.join(item_path, mf))
                merged = merge_to_btc_df(btc_month, astro_master)
                merged_out = datetime_to_str(merged, fmt=datetime_fmt)

                out_path = os.path.join(out_year_dir, mf)
                merged_out.to_parquet(out_path, index=False)

                yyyy, mm = extract_year_month_from_filename(mf)
                if yyyy and mm:
                    print(f"{yyyy} 年 {mm} 月合并完成")
                elif yyyy:
                    print(f"{yyyy} 年 合并完成（文件：{mf}）")
                else:
                    print(f"{yd} 年 {mf} 合并完成")
            except Exception as e:
                print(f"处理 {yd}/{mf} 失败：{e}")

# 主函数
if __name__ == "__main__":
    merge_all()
    # TODO: 进度条未处理