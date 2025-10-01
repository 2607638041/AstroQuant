# gen_jiuxing_fixed.py
"""
生成 日家九星 与 月家九星，范围：2017-02-06 .. (当前年份 + 2)-12-31
输出 Parquet： 项目根目录/data/astro_data/九星.parquet

说明：
- 使用项目中已有的节气和干支历函数（节气：calculate_solar_terms_2017_now，干支历：get_ganzhi_data）
- 日家九星以 2017-02-06 为基准（该日为 一白），按天连续顺推（避免重置/错位）
- 月家九星按你给的歌诀分三类年份，并以 寅月 为 正月（序号0），按「逆排」即每月往前减 1（循环1..9）
"""

from datetime import date, timedelta, datetime
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm

# 导入已有模块（保持你项目中原来的名字）
from 节气 import calculate_solar_terms_2017_now
from 干支历 import get_ganzhi_data

# --------------------------
# 常量与辅助映射
# --------------------------
# 九星按数字 1..9 对应的中文名称（1 -> 一白, 2 -> 二黑, ..., 9 -> 九紫）
NUM_TO_STAR = {
    1: "一白", 2: "二黑", 3: "三碧", 4: "四绿", 5: "五黄",
    6: "六白", 7: "七赤", 8: "八白", 9: "九紫"
}

# 月支序列：寅月 为 正月（index 0）
LUNAR_MONTH_ZHI_SEQ = ["寅","卯","辰","巳","午","未","申","酉","戌","亥","子","丑"]

# 年地支所属三类与起始数（正月/寅月对应的九星数字）
# 子午卯酉年 -> 正月起始为8
# 寅申巳亥年 -> 正月起始为2
# 辰戌丑未年 -> 正月起始为5
GROUP_TO_START = {
    "子午卯酉": 8,
    "寅申巳亥": 2,
    "辰戌丑未": 5
}

# 便于根据单个年地支找到其分组
ZHI_TO_GROUP = {}
for g, zhis in [("子午卯酉", "子午卯酉"),
                ("寅申巳亥", "寅申巳亥"),
                ("辰戌丑未", "辰戌丑未")]:
    for ch in zhis:
        ZHI_TO_GROUP[ch] = g

# --------------------------
# 读取并预处理外部数据（节气与干支）
# --------------------------
# 节气（只做日期->名称快速映射，兼容返回 date 或 str）
solar_terms_raw = calculate_solar_terms_2017_now()  # 你的函数返回结构可能是 [(name, date), ...] 或 [(date, name), ...]
solar_terms_dict = {}
for item in solar_terms_raw:
    # 兼容两种元组顺序
    if len(item) >= 2:
        a, b = item[0], item[1]
        # 判断哪项是日期（datetime/date/字符串）
        if isinstance(a, (datetime, date)):
            d = a
            name = str(b)
        elif isinstance(b, (datetime, date)):
            d = b
            name = str(a)
        else:
            # 都不是日期，跳过
            continue
        date_str = d.strftime("%Y-%m-%d")
        solar_terms_dict[date_str] = name

# 干支历：期望返回 (date_str, year_ganzhi, month_ganzhi, day_ganzhi)
ganzhi_raw = get_ganzhi_data()
ganzhi_dict = {}
for rec in ganzhi_raw:
    # 支持 tuple/list 或 dict 返回
    if isinstance(rec, (tuple, list)) and len(rec) >= 4:
        date_str = rec[0]
        year_ganzhi = rec[1]
        month_ganzhi = rec[2]
        day_ganzhi = rec[3]
    elif isinstance(rec, dict):
        date_str = rec.get("date") or rec.get("date_str")
        year_ganzhi = rec.get("year_ganzhi", "")
        month_ganzhi = rec.get("month_ganzhi", "")
        day_ganzhi = rec.get("day_ganzhi", "")
    else:
        continue

    # 取干支的第一、第二字（安全检查）
    def split_ganzhi(s):
        if not s or not isinstance(s, str):
            return ("","")
        s = s.strip()
        if len(s) >= 2:
            return (s[0], s[1])
        return (s[0], "") if len(s)==1 else ("","")

    yg, yz = split_ganzhi(year_ganzhi)
    mg, mz = split_ganzhi(month_ganzhi)
    dg, dz = split_ganzhi(day_ganzhi)

    ganzhi_dict[date_str] = {
        "year_gan": yg, "year_zhi": yz,
        "month_gan": mg, "month_zhi": mz,
        "day_gan": dg, "day_zhi": dz
    }

# --------------------------
# 配置：起止日期、输出路径
# --------------------------
START_DATE = date(2017, 2, 6)
END_DATE = date(datetime.now().year + 2, 12, 31)

# 输出到项目根目录 /data/astro_data/九星.parquet
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 视文件位置，通常取上级作为项目根；如需调整请改这行
OUT_PATH = PROJECT_ROOT / "data" / "astro_data" / "九星.parquet"

# --------------------------
# 算法函数：月家九星（日家用连续日序列）
# --------------------------
def get_jieqi_for_date(d: date):
    key = d.strftime("%Y-%m-%d")
    if key in solar_terms_dict:
        return {"name": solar_terms_dict[key], "type": "solar_term"}
    return None

def get_ganzhi_for_date(d: date):
    key = d.strftime("%Y-%m-%d")
    return ganzhi_dict.get(key, {
        "year_gan":"", "year_zhi":"", "month_gan":"", "month_zhi":"", "day_gan":"", "day_zhi":""
    })

def compute_month_star_by_ganzhi(d: date):
    """
    使用你给出的歌诀按年地支分组与寅月为正月的序列来计算当月的九星。
    逻辑：
      1. 取该日对应的年地支 year_zhi（若缺失，退回使用公历月份近似）
      2. 找到该年的分组，确定正月起始数 start_num（1..9）
      3. 取该日的月支 month_zhi（应为 寅/卯/...），若缺失则根据公历月份尝试映射（兜底）
      4. 计算月序号 m_index = LUNAR_MONTH_ZHI_SEQ.index(month_zhi)  （寅->0 为正月）
      5. 月家九星数字 = ((start_num - m_index - 1) % 9) + 1   （因为是逆排：正月=start, 二月=start-1）
    返回中文九星名称（如 "八白"）
    """
    gj = get_ganzhi_for_date(d)
    year_zhi = gj.get("year_zhi", "")
    month_zhi = gj.get("month_zhi", "")

    # 1) 确定组起始数
    start_num = None
    if year_zhi and year_zhi in ZHI_TO_GROUP:
        grp = ZHI_TO_GROUP[year_zhi]
        start_num = GROUP_TO_START.get(grp)
    # 兜底：若没有年地支或找不到，则尝试用公历年的天干地支并取年最后一字（不常见）
    if start_num is None:
        # 最简单兜底：按公历年 % 3 选一个组
        mod = d.year % 3
        if mod == 0:
            start_num = GROUP_TO_START["子午卯酉"]
        elif mod == 1:
            start_num = GROUP_TO_START["寅申巳亥"]
        else:
            start_num = GROUP_TO_START["辰戌丑未"]

    # 2) 确定月支位置（寅=0）
    m_index = None
    if month_zhi and month_zhi in LUNAR_MONTH_ZHI_SEQ:
        m_index = LUNAR_MONTH_ZHI_SEQ.index(month_zhi)
    else:
        # 兜底：用公历月份近似映射：假设寅月约当农历正月，大概对应公历1-2月（不精确）
        # 我们把公历月份 1->寅(0), 2->卯(1), 3->辰(2), ... 循环映射
        m_index = (d.month - 1) % 12

    star_num = ((start_num - m_index - 1) % 9) + 1
    return NUM_TO_STAR[star_num]

# 日家九星按连续日序列固定推（避免因干支映射错位导致断裂）
REF_DATE = date(2017, 2, 6)
REF_DAY_STAR_INDEX = 1  # 参考：2017-02-06 为 一白 -> 对应数字 1
def compute_day_star_continuous(d: date):
    offset = (d - REF_DATE).days
    # day_star_num 从 1 到 9 循环
    star_num = ((REF_DAY_STAR_INDEX - 1 + offset) % 9) + 1
    return NUM_TO_STAR[star_num]

# --------------------------
# 生成 DataFrame 并导出 Parquet
# --------------------------
def generate_jiuxing_df(start_date: date, end_date: date):
    rows = []
    total = (end_date - start_date).days + 1
    cur = start_date
    for _ in tqdm(range(total), desc="生成九星数据"):
        day_star = compute_day_star_continuous(cur)
        month_star = compute_month_star_by_ganzhi(cur)
        rows.append({
            "日期": cur,
            "日家九星": day_star,
            "月家九星": month_star
        })
        cur += timedelta(days=1)
    df = pd.DataFrame(rows)
    df["日期"] = pd.to_datetime(df["日期"])
    return df

def export_parquet():
    print(f"生成范围：{START_DATE} -> {END_DATE}")
    df = generate_jiuxing_df(START_DATE, END_DATE)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, engine="pyarrow")
    print(f"已导出：{OUT_PATH} （共 {len(df)} 行）")

if __name__ == "__main__":
    export_parquet()
