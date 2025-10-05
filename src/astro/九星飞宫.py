#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 日家九星 与 月家九星，范围：2017-02-06 .. (当前年份 + 2)-12-31
输出 Parquet： 项目根目录/data/astro_data/九星.parquet

关键点：
- 节气当天可能是甲子日；重置点为每个节气当天开始直到下一个节气前的所有甲子日。
- 节气列仅在六个关键节气当天显示标准名（冬至, 雨水, 谷雨, 夏至, 处暑, 霜降）。
- 日柱列只在甲子日显示“甲子”。
- 日期输出为字符串 YYYY-MM-DD，确保任何 Parquet Viewer 显示正常。
"""

from datetime import date, timedelta, datetime
import os
import pandas as pd
from tqdm import tqdm

# 外部已有模块（名称与路径按你工程保持不变）
from 节气 import calculate_solar_terms_2017_now
from 干支历 import get_ganzhi_data

# --------------------------
# 常量与映射
# --------------------------
NUM_TO_STAR = {1: "一白", 2: "二黑", 3: "三碧", 4: "四绿", 5: "五黄",
               6: "六白", 7: "七赤", 8: "八白", 9: "九紫"}

LUNAR_MONTH_ZHI_SEQ = ["寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥", "子", "丑"]
GROUP_TO_START = {"子午卯酉": 8, "寅申巳亥": 2, "辰戌丑未": 5}
ZHI_TO_GROUP = {}
for g, zhis in [("子午卯酉", "子午卯酉"),
                ("寅申巳亥", "寅申巳亥"),
                ("辰戌丑未", "辰戌丑未")]:
    for ch in zhis:
        ZHI_TO_GROUP[ch] = g

# --------------------------
# 配置
# --------------------------
START_DATE = date(2017, 2, 6)
END_DATE = date(datetime.now().year + 2, 12, 31)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "astro_data", "九星.parquet")

KEY_SOLAR_TERMS = ["冬至", "雨水", "谷雨", "夏至", "处暑", "霜降"]
SOLAR_TERM_DAY_STAR = {
    "冬至": (1, "forward"),
    "雨水": (7, "forward"),
    "谷雨": (4, "forward"),
    "夏至": (9, "backward"),
    "处暑": (3, "backward"),
    "霜降": (6, "backward"),
}

# --------------------------
# 节气处理
# --------------------------
def normalize_term_name(s: str):
    if not s:
        return ""
    s = s.strip()
    for remove in ["（首甲子）", "(首甲子)", "首甲子", "节"]:
        s = s.replace(remove, "")
    return s.strip()

solar_terms_raw = calculate_solar_terms_2017_now()
solar_terms_dict = {}
for item in solar_terms_raw:
    if not item:
        continue
    try:
        a, b = item[0], item[1]
    except Exception:
        continue
    d = None; name = None
    if isinstance(a, (datetime, date)):
        d = a.date() if isinstance(a, datetime) else a
        name = str(b)
    elif isinstance(b, (datetime, date)):
        d = b.date() if isinstance(b, datetime) else b
        name = str(a)
    else:
        try:
            dt = datetime.strptime(str(a), "%Y-%m-%d")
            d = dt.date(); name = str(b)
        except Exception:
            try:
                dt = datetime.strptime(str(b), "%Y-%m-%d")
                d = dt.date(); name = str(a)
            except Exception:
                continue
    if d is None:
        continue
    # 使用date对象而不是字符串
    solar_terms_dict[d] = normalize_term_name(name)

SOLAR_TERM_KEY_DATES = {}
for dt, nm in solar_terms_dict.items():
    if not nm:
        continue
    for key in KEY_SOLAR_TERMS:
        if key == nm or key in nm or nm in key:
            SOLAR_TERM_KEY_DATES[dt] = key
            break

# --------------------------
# 干支历处理
# --------------------------
ganzhi_raw = get_ganzhi_data()
ganzhi_dict = {}
for rec in ganzhi_raw:
    if isinstance(rec, (tuple, list)) and len(rec) >= 4:
        date_str = rec[0]; year_ganzhi = rec[1]; month_ganzhi = rec[2]; day_ganzhi = rec[3]
    elif isinstance(rec, dict):
        date_str = rec.get("date") or rec.get("date_str")
        year_ganzhi = rec.get("year_ganzhi", "")
        month_ganzhi = rec.get("month_ganzhi", "")
        day_ganzhi = rec.get("day_ganzhi", "")
    else:
        continue

    try:
        if isinstance(date_str, (datetime, date)):
            key = date_str.strftime("%Y-%m-%d")
        else:
            key = datetime.strptime(str(date_str), "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        key = str(date_str)

    def split_ganzhi(s):
        if not s or not isinstance(s, str): return "", ""
        s = s.strip()
        if len(s) >= 2: return s[0], s[1]
        return (s[0], "") if len(s) == 1 else ("", "")

    yg, yz = split_ganzhi(year_ganzhi)
    mg, mz = split_ganzhi(month_ganzhi)
    dg, dz = split_ganzhi(day_ganzhi)
    ganzhi_dict[key] = {"year_gan": yg, "year_zhi": yz,
                        "month_gan": mg, "month_zhi": mz,
                        "day_gan": dg, "day_zhi": dz}

def get_ganzhi_for_date(d: date):
    key = d.strftime("%Y-%m-%d")
    return ganzhi_dict.get(key, {"year_gan": "", "year_zhi": "",
                                 "month_gan": "", "month_zhi": "",
                                 "day_gan": "", "day_zhi": ""})

# --------------------------
# 月家九星
# --------------------------
def compute_month_star_by_ganzhi(d: date):
    """
    根据日期计算对应的月建星
    参数:
        d: date类型，表示要计算的日期
    返回:
        对应的月建星名称
    """
    # 获取日期对应的干支信息
    gj = get_ganzhi_for_date(d)
    # 从干支信息中提取年支和月支
    year_zhi = gj.get("year_zhi", "")
    month_zhi = gj.get("month_zhi", "")

    # 初始化起始编号为None
    start_num = None
    # 如果年支存在且在支组映射中，获取对应的组别和起始编号
    if year_zhi and year_zhi in ZHI_TO_GROUP:
        grp = ZHI_TO_GROUP[year_zhi]
        start_num = GROUP_TO_START.get(grp)
    # 如果起始编号仍未确定，则根据年份模3的值来确定起始编号
    if start_num is None:
        mod = d.year % 3
        if mod == 0:
            start_num = GROUP_TO_START["子午卯酉"]
        elif mod == 1:
            start_num = GROUP_TO_START["寅申巳亥"]
        else:
            start_num = GROUP_TO_START["辰戌丑未"]

    # 如果月支存在且在月支序列中，获取其索引
    if month_zhi and month_zhi in LUNAR_MONTH_ZHI_SEQ:
        m_index = LUNAR_MONTH_ZHI_SEQ.index(month_zhi)
    else:
        # 否则使用公历月份计算索引
        m_index = (d.month - 1) % 12

    # 计算星位编号
    star_num = ((start_num - m_index - 1) % 9) + 1
    # 返回对应的星位名称
    return NUM_TO_STAR[star_num]

# --------------------------
# 日家九星：节气区间内所有甲子日重置
# --------------------------
REF_DATE = date(2017, 2, 6)
REF_DAY_STAR_INDEX = 1

def compute_day_stars_range(start_date: date, end_date: date):
    term_dates_sorted = sorted([(datetime.strptime(d, "%Y-%m-%d").date() if isinstance(d, str) else d, n)
                                for d, n in SOLAR_TERM_KEY_DATES.items()
                                if n in SOLAR_TERM_DAY_STAR])
    term_dates_sorted.append((end_date + timedelta(days=1), None))

    reset_map = {}
    for i in range(len(term_dates_sorted)-1):
        term_start, term_name = term_dates_sorted[i]
        next_term_start, _ = term_dates_sorted[i+1]
        cur = term_start
        while cur < next_term_start:
            gj = get_ganzhi_for_date(cur)
            day_full = gj.get("day_gan", "") + gj.get("day_zhi", "")
            if day_full == "甲子":
                star_num, direction = SOLAR_TERM_DAY_STAR[term_name]
                # 使用date对象而不是字符串
                reset_map[cur] = (star_num, direction)
            cur += timedelta(days=1)

    day_star_map = {}
    cur = start_date
    s = REF_DAY_STAR_INDEX
    dirc = "forward"
    while cur <= end_date:
        # 使用date对象而不是字符串
        if cur in reset_map:
            s, dirc = reset_map[cur]
        day_star_map[cur] = NUM_TO_STAR[s]

        if dirc == "forward":
            s += 1
            if s > 9: s = 1
        else:
            s -= 1
            if s < 1: s = 9
        cur += timedelta(days=1)

    return day_star_map

# --------------------------
# 生成 DataFrame
# --------------------------
def generate_jiuxing_df(start_date: date, end_date: date):
    rows = []
    total = (end_date - start_date).days + 1
    day_star_map = compute_day_stars_range(start_date, end_date)
    cur = start_date
    for _ in tqdm(range(total), desc="生成九星数据"):
        # 在导出前将日期对象转换为字符串
        date_str = cur.strftime("%Y-%m-%d")
        day_star = day_star_map.get(cur, "")
        month_star = compute_month_star_by_ganzhi(cur)
        rows.append({
            "日期": date_str,
            "月家九星": month_star,
            "日家九星": day_star
        })
        cur += timedelta(days=1)
    df = pd.DataFrame(rows)
    return df

# --------------------------
# 导出 Parquet
# --------------------------
def export_parquet():
    print(f"生成范围：{START_DATE} -> {END_DATE}")
    df = generate_jiuxing_df(START_DATE, END_DATE)

    # 确保星宿列是字符串类型
    df["月家九星"] = df["月家九星"].astype("string")
    df["日家九星"] = df["日家九星"].astype("string")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, engine="pyarrow")
    print(f"已导出：{OUT_PATH} （共 {len(df)} 行）")

if __name__ == "__main__":
    export_parquet()
