#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_features.py

功能：
- 将 data/btc_data_5m 下的 BTC 5m 文件（按年单文件或最新年份按月目录）与
  data/astro_data 下的玄学特征（九星、十二建星、干支历、星宿、节气）合并，输出到 data/merged/btc/btc_5m。
- 月份文件优先，如果最新年份有月份文件，则按月合并。
- 玄学 master 使用时间索引合并，避免重复列名导致错误。
- 数据清洗：时间统一、删除重复、数值列转换、节气向后填充。
- 输出前将 datetime 列格式化为字符串（默认 "%Y-%m-%d %H:%M"）。
"""

import os
import sys
import traceback
import pandas as pd

# 添加项目根目录到系统路径，以便导入astro模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 尝试导入astro模块中的各个组件
try:
    from src.astro.九星飞宫 import generate_jiuxing_df
    from src.astro.十二建星 import generate_jian_xing_data
    from src.astro.干支历 import get_ganzhi_data
    from src.astro.星宿 import generate_xiuxiu_for_period
    from src.astro.节气 import calculate_solar_terms_2017_now
    ASTRO_MODULES_AVAILABLE = True
except ImportError:
    ASTRO_MODULES_AVAILABLE = False
    print("警告：无法导入astro模块，将直接从Parquet文件读取数据")

# -----------------------
# 配置路径（基于文件位置）
# -----------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BTC_DIR = os.path.join(ROOT_DIR, 'data', 'btc_data_5m')
ASTRO_DIR = os.path.join(ROOT_DIR, 'data', 'astro_data')
MERGED_DIR = os.path.join(ROOT_DIR, 'data', 'merged', 'btc', 'btc_5m')

ASTRO_FILES = {
    '九星': '九星.parquet',
    '十二建星': '十二建星.parquet',
    '干支历': '干支历.parquet',
    '星宿': '星宿.parquet',
    '节气': '节气.parquet'
}

os.makedirs(MERGED_DIR, exist_ok=True)

# -----------------------
# 辅助函数
# -----------------------
def find_date_column(df: pd.DataFrame):
    """在 df 中猜测日期列名称，返回列名或 None"""
    for c in df.columns:
        low = c.lower()
        if 'date' in low or '时间' in c or '日期' in c:
            return c
    return None

def safe_to_datetime(s, utc=True):
    """安全的 pd.to_datetime 包装，返回 tz-aware (如果 utc=True) 或 naive datetime"""
    try:
        return pd.to_datetime(s, format='%Y-%m-%d %H:%M', utc=utc, errors='coerce')
    except Exception:
        try:
            return pd.to_datetime(s, utc=utc, errors='coerce')
        except Exception:
            # 最后退回到不带 utc 的解析（兼容极端输入）
            return pd.to_datetime(s, errors='coerce')

def clean_btc_df(df: pd.DataFrame):
    """清洗 BTC 数据，确保有 datetime 列并做基本转换"""
    df = df.copy()
    if 'datetime' not in df.columns:
        raise ValueError("BTC 文件缺少 datetime 列")
    df['datetime'] = safe_to_datetime(df['datetime'], utc=True)
    df = df.dropna(subset=['datetime'])
    df = df.drop_duplicates(subset=['datetime'])
    # 数值列转换
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('datetime')

def clean_merged_df(df: pd.DataFrame):
    """清洗合并后的 df：datetime 标准化，数值列转换，节气向后填充，去重并排序"""
    df = df.copy()
    # datetime UTC-aware
    df['datetime'] = safe_to_datetime(df['datetime'], utc=True)
    df = df.dropna(subset=['datetime'])
    # 数值列
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # 节气向后填充
    if '节气' in df.columns:
        df['节气'] = df['节气'].ffill()
    # 删除重复行（按 datetime）
    df = df.drop_duplicates(subset=['datetime'])
    df = df.sort_values('datetime')
    return df

def datetime_to_str(df: pd.DataFrame, fmt="%Y-%m-%d %H:%M"):
    """
    将 df 中的 datetime 列转换为字符串格式。
    如果 datetime 带时区（tz-aware），strftime 会把时区信息考虑进去。
    """
    df = df.copy()
    if 'datetime' in df.columns:
        # 确保为 datetime 类型
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
        except Exception:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        # 对于存在 NaT 的行，保留为 NaN（写入 parquet 时为 null）
        df['datetime'] = df['datetime'].dt.strftime(fmt)
    return df

# -----------------------
# 玄学 master 构建
# -----------------------
def load_astro_master():
    dfs = []
    for key, fname in ASTRO_FILES.items():
        path = os.path.join(ASTRO_DIR, fname)
        if not os.path.exists(path):
            print(f"[warn] 玄学文件不存在: {path}")
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"[warn] 读取玄学文件失败: {path} -> {e}")
            continue
        date_col = find_date_column(df)
        if not date_col:
            print(f"[warn] 文件 {fname} 中未找到日期列，跳过")
            continue
        dt = safe_to_datetime(df[date_col], utc=True)
        df_vals = df.drop(columns=[date_col])
        df_vals['__datetime__'] = dt
        df_vals = df_vals.dropna(subset=['__datetime__'])
        if df_vals.empty:
            continue
        df_vals = df_vals.set_index('__datetime__')
        dfs.append((key, df_vals))

    if not dfs:
        print("[error] 未找到有效玄学文件")
        return None

    # 先用第一个作为 base，再 join 其它，遇到冲突列做简单处理（优先保留 master 非空列）
    master = dfs[0][1].copy()
    for key, dfv in dfs[1:]:
        conflict_cols = [c for c in dfv.columns if c in master.columns]
        if conflict_cols:
            for c in conflict_cols:
                if not master[c].isna().all():
                    # master 有值则丢弃输入表的同名列，避免覆盖
                    dfv = dfv.drop(columns=[c])
        master = master.join(dfv, how='outer')

    master = master.sort_index().reset_index()
    master = master.rename(columns={'__datetime__': 'datetime'})
    master['datetime'] = pd.to_datetime(master['datetime'], utc=True)

    # 统一节气列名（如果有多个带 '节气' 的列）
    for c in list(master.columns):
        if '节气' in c and c != '节气':
            if '节气' not in master.columns:
                master = master.rename(columns={c: '节气'})
            else:
                master = master.drop(columns=[c])

    # 节气向后填充
    if '节气' in master.columns:
        master = master.sort_values('datetime')
        master['节气'] = master['节气'].ffill()

    return master

def merge_to_btc_df(btc_df: pd.DataFrame, astro_master: pd.DataFrame):
    btc_df = clean_btc_df(btc_df)
    astro_master = astro_master.sort_values('datetime')
    # 使用 merge_asof，direction='backward'：寻找 BTC 时间点最近且不晚于该时间的玄学行
    merged = pd.merge_asof(btc_df, astro_master, on='datetime', direction='backward')
    merged = clean_merged_df(merged)
    return merged

# -----------------------
# 主流程
# -----------------------
def merge_all(datetime_fmt="%Y-%m-%d %H:%M"):
    print("开始合并 BTC 与玄学特征（输出到 data/merged/btc/btc_5m）...")
    try:
        astro_master = load_astro_master()
        if astro_master is None:
            print("[error] 无有效玄学 master，退出")
            return
    except Exception as e:
        print("[error] 构建玄学 master 失败：", e)
        traceback.print_exc()
        return

    if not os.path.exists(BTC_DIR):
        print(f"[error] BTC 数据目录不存在: {BTC_DIR}")
        return

    items = sorted(os.listdir(BTC_DIR))
    for item in items:
        item_path = os.path.join(BTC_DIR, item)
        # 年目录（按月）
        if os.path.isdir(item_path):
            year = item
            out_year_dir = os.path.join(MERGED_DIR, year)
            os.makedirs(out_year_dir, exist_ok=True)
            month_files = sorted([f for f in os.listdir(item_path) if f.endswith('.parquet')])
            if not month_files:
                continue
            print(f"[info] 处理年份目录（按月）: {year}, {len(month_files)} 个文件")
            for mf in month_files:
                mf_path = os.path.join(item_path, mf)
                try:
                    btc_month = pd.read_parquet(mf_path)
                except Exception as e:
                    print(f"[warn] 读取 BTC 月文件失败: {mf_path} -> {e}")
                    continue
                try:
                    merged = merge_to_btc_df(btc_month, astro_master)
                    # 输出前转 datetime 为字符串
                    merged_out = datetime_to_str(merged, fmt=datetime_fmt)
                    out_path = os.path.join(out_year_dir, mf)
                    merged_out.to_parquet(out_path, index=False)
                    print(f"[ok] 已合并并保存: {mf} (rows={len(merged_out)})")
                except Exception as e:
                    print(f"[error] 合并月文件失败 {mf_path}: {e}")
                    traceback.print_exc()
            continue

        # 单文件 parquet（按年或整文件）
        if item.endswith('.parquet'):
            fpath = os.path.join(BTC_DIR, item)
            try:
                btc_df = pd.read_parquet(fpath)
            except Exception as e:
                print(f"[warn] 读取 BTC 文件失败: {fpath} -> {e}")
                continue
            try:
                merged = merge_to_btc_df(btc_df, astro_master)
                # 输出前转 datetime 为字符串
                merged_out = datetime_to_str(merged, fmt=datetime_fmt)
                out_path = os.path.join(MERGED_DIR, item)
                merged_out.to_parquet(out_path, index=False)
                print(f"[ok] 已合并并保存: {item} (rows={len(merged_out)})")
            except Exception as e:
                print(f"[error] 合并文件失败 {fpath}: {e}")
                traceback.print_exc()

    print("全部合并完成。输出目录:", MERGED_DIR)

if __name__ == "__main__":
    # 默认时间格式为 年-月-日 时:分（无秒）
    merge_all(datetime_fmt="%Y-%m-%d %H:%M")
