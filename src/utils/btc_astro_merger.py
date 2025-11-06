#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
btc_astro_merger.py

功能：
- 将 BTC 5m 数据与 astro_data 的玄学特征合并
- 年文件输出到 MERGED_DIR 根（不建子文件夹）
- 月文件输出到 MERGED_DIR/<YEAR>/ 下
- 数据清洗、节气 ffill、按天对齐、保留 BTC datetime
- 自动扫描玄学文件、打印文件列表、支持进度条
"""

import os
import re
import pandas as pd
from tqdm import tqdm

# ============ 参数设置 ============
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BTC_DIR = os.path.join(ROOT_DIR, 'data', 'btc_data_5m')
ASTRO_DIR = os.path.join(ROOT_DIR, 'data', 'astro_data')          # 玄学数据目录（可调整）
MERGED_DIR = os.path.join(ROOT_DIR, 'data', 'merged', 'btc', 'btc_5m')

# 是否显示进度条
SHOW_PROGRESS = True

os.makedirs(MERGED_DIR, exist_ok=True)

# ============ 工具函数 ============
def find_date_column(df: pd.DataFrame):
    """查找日期列"""
    for c in df.columns:
        if 'date' in c.lower() or '时间' in c or '日期' in c:
            return c
    return None

def safe_to_datetime(s, utc=True):
    """安全转换为 datetime"""
    try:
        return pd.to_datetime(s, errors='coerce', utc=utc)
    except Exception:
        if hasattr(s, '__len__'):
            return pd.Series([pd.NaT] * len(s))
        return pd.NaT

def clean_btc_df(df: pd.DataFrame):
    """清洗 BTC 数据"""
    df = df.copy()
    if 'datetime' not in df.columns:
        raise ValueError("BTC 文件缺少 datetime 列")
    df['datetime'] = safe_to_datetime(df['datetime'], utc=True)
    df = df.dropna(subset=['datetime']).drop_duplicates(subset=['datetime'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('datetime').reset_index(drop=True)

def clean_merged_df(df: pd.DataFrame):
    """清洗合并后数据"""
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

def datetime_to_str(df: pd.DataFrame, fmt="%Y-%m-%d %H:%M"):
    """把 datetime 转为字符串格式"""
    df = df.copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
        df['datetime'] = df['datetime'].dt.strftime(fmt)
    return df

# ============ 玄学数据自动扫描 ============
def scan_astro_files(astro_dir: str):
    """
    扫描 astro_dir 下所有 .parquet 文件
    返回: {文件名: 文件路径} 字典
    """
    astro_files = {}
    if not os.path.exists(astro_dir):
        print(f"[警告] 玄学数据目录不存在: {astro_dir}")
        return astro_files

    try:
        for fname in sorted(os.listdir(astro_dir)):
            if fname.endswith('.parquet'):
                astro_files[fname] = os.path.join(astro_dir, fname)
    except Exception as e:
        print(f"[错误] 无法扫描玄学数据目录: {e}")

    return astro_files

def load_astro_master():
    """加载并合并所有玄学数据"""
    print("=" * 60)
    print("开始加载玄学数据...")
    print("=" * 60)

    astro_files = scan_astro_files(ASTRO_DIR)

    if not astro_files:
        print("[错误] 未找到任何 .parquet 文件")
        return None

    print(f"\n找到 {len(astro_files)} 个玄学数据文件：")
    for idx, fname in enumerate(sorted(astro_files.keys()), 1):
        print(f"  {idx}. {fname}")
    print()

    dfs = []
    failed_files = []

    for fname, path in sorted(astro_files.items()):
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            failed_files.append((fname, f"读取失败: {e}"))
            continue

        date_col = find_date_column(df)
        if not date_col:
            failed_files.append((fname, "未找到日期列"))
            continue

        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
        df_vals = df.drop(columns=[date_col]).copy()
        df_vals = df_vals.dropna(subset=['datetime'])

        if df_vals.empty:
            failed_files.append((fname, "数据为空"))
            continue

        df_vals = df_vals.set_index('datetime')
        dfs.append((fname, df_vals))

    if failed_files:
        print("\n加载失败的文件：")
        for fname, reason in failed_files:
            print(f"  ✗ {fname}: {reason}")

    if not dfs:
        print("[错误] 未成功加载任何玄学文件")
        return None

    print(f"\n成功加载 {len(dfs)} 个文件\n")

    # 逐个 outer join
    master = dfs[0][1].copy()
    for fname, dfv in dfs[1:]:
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

    print(f"✓ 玄学 master 构建完成，形状: {master.shape}\n")
    return master

# ============ 数据提取函数 ============
def extract_year_from_filename(fname: str):
    """从文件名提取年份"""
    m = re.search(r'(19|20)\d{2}', fname)
    if m:
        return m.group(0)
    return None

def extract_year_month_from_filename(fname: str):
    """从文件名提取年月"""
    m = re.search(r'((19|20)\d{2})[^\d]?([01]\d)', fname)
    if m:
        return m.group(1), m.group(3)
    y = extract_year_from_filename(fname)
    return (y, None) if y else (None, None)

# ============ 数据合并函数 ============
def merge_to_btc_df(btc_df: pd.DataFrame, astro_master: pd.DataFrame):
    """合并单个 BTC 数据与玄学主数据"""
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

    merged['datetime'] = left['datetime']
    merged = clean_merged_df(merged)

    if 'datetime_astro' in merged.columns:
        merged = merged.drop(columns=['datetime_astro'])

    if 'date' in merged.columns:
        merged = merged.drop(columns=['date'])

    return merged

# ============ 主合并流程 ============
def merge_all(datetime_fmt="%Y-%m-%d %H:%M"):
    """合并所有 BTC 数据"""
    astro_master = load_astro_master()
    if astro_master is None:
        print("[错误] 无有效玄学 master，退出")
        return

    if not os.path.exists(BTC_DIR):
        print(f"[错误] BTC 数据目录不存在: {BTC_DIR}")
        return

    items = sorted(os.listdir(BTC_DIR))

    # 1) 处理年文件
    year_files = [f for f in items if f.endswith('.parquet')]
    if year_files:
        print("=" * 60)
        print(f"处理年文件（{len(year_files)} 个）")
        print("=" * 60)
        iterator = tqdm(year_files, desc="年文件", disable=not SHOW_PROGRESS)
        for yf in iterator:
            try:
                year_path = os.path.join(BTC_DIR, yf)
                btc_year = pd.read_parquet(year_path)
                merged = merge_to_btc_df(btc_year, astro_master)
                merged_out = datetime_to_str(merged, fmt=datetime_fmt)

                out_path = os.path.join(MERGED_DIR, yf)
                merged_out.to_parquet(out_path, index=False)

                year_str = extract_year_from_filename(yf) or yf
                iterator.set_postfix_str(f"✓ {year_str}", refresh=True)
            except Exception as e:
                iterator.set_postfix_str(f"✗ {yf} 失败", refresh=True)
        print()

    # 2) 处理年目录下的月文件
    year_dirs = [d for d in items if os.path.isdir(os.path.join(BTC_DIR, d))]
    if year_dirs:
        print("=" * 60)
        print(f"处理月文件（{len(year_dirs)} 个年目录）")
        print("=" * 60)

        # 收集所有月文件
        all_month_tasks = []
        for yd in year_dirs:
            item_path = os.path.join(BTC_DIR, yd)
            month_files = sorted([f for f in os.listdir(item_path) if f.endswith('.parquet')])
            for mf in month_files:
                all_month_tasks.append((yd, mf, os.path.join(item_path, mf)))

        # 使用单个进度条处理所有月文件
        iterator = tqdm(all_month_tasks, desc="月文件", disable=not SHOW_PROGRESS)
        for yd, mf, mf_path in iterator:
            try:
                out_year_dir = os.path.join(MERGED_DIR, yd)
                os.makedirs(out_year_dir, exist_ok=True)

                btc_month = pd.read_parquet(mf_path)
                merged = merge_to_btc_df(btc_month, astro_master)
                merged_out = datetime_to_str(merged, fmt=datetime_fmt)

                out_path = os.path.join(out_year_dir, mf)
                merged_out.to_parquet(out_path, index=False)

                yyyy, mm = extract_year_month_from_filename(mf)
                status = f"✓ {yyyy}-{mm}" if (yyyy and mm) else f"✓ {yd}/{mf}"
                iterator.set_postfix_str(status, refresh=True)
            except Exception as e:
                iterator.set_postfix_str(f"✗ {yd}/{mf}", refresh=True)
        print()

    print("=" * 60)
    print("✓ 合并完成")
    print("=" * 60)

# ============ 主函数 ============
if __name__ == "__main__":
    merge_all()