#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 十二值神 数据，范围：2017-01-01 .. (当前年份 + 2)-12-31
逻辑：根据月支确定青龙起始日支，然后顺推十二值神
输出 Parquet： 项目根目录/data/astro_data/十二值神.parquet
字段：
- 日期（字符串 YYYY-MM-DD）
- 值神
"""

import datetime
import os
import sys
import pandas as pd
from tqdm import tqdm

# 处理相对导入问题
try:
    from .干支历 import get_ganzhi_data, calculate_day_ganzhi
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 十二地支映射
DI_ZHI_MAP = {
    '子': 0, '丑': 1, '寅': 2, '卯': 3, '辰': 4, '巳': 5,
    '午': 6, '未': 7, '申': 8, '酉': 9, '戌': 10, '亥': 11
}

# 逆向映射：索引到地支
INDEX_TO_DI_ZHI = {v: k for k, v in DI_ZHI_MAP.items()}

# 十二值神（固定顺序）
ZHI_SHEN = ['青龙', '明堂', '天刑', '朱雀', '金匮', '天德',
            '白虎', '玉堂', '天牢', '玄武', '司命', '勾陈']

# 月支 -> 青龙起始日支（正统规则）
MONTH_ZHI_TO_QINGLONG_START = {
    '子': '申', '午': '申',  # 子午月：从申日起青龙
    '丑': '戌', '未': '戌',  # 丑未月：从戌日起青龙
    '寅': '子', '申': '子',  # 寅申月：从子日起青龙
    '卯': '寅', '酉': '寅',  # 卯酉月：从寅日起青龙
    '辰': '辰', '戌': '辰',  # 辰戌月：从辰日起青龙
    '巳': '午', '亥': '午'   # 巳亥月：从午日起青龙
}


def get_zhi_shen_for_date(month_zhi: str, day_zhi: str) -> str:
    """
    根据月支和日支，计算该日的值神

    算法：
    1. 根据月支查表获得青龙起始日支
    2. 计算从起始日支到当前日支的偏移量（可能跨月）
    3. 偏移量对应值神序列中的位置
    """
    # 获取青龙起始日支
    qinglong_start = MONTH_ZHI_TO_QINGLONG_START.get(month_zhi, '子')

    # 获取索引
    start_index = DI_ZHI_MAP.get(qinglong_start, 0)
    day_index = DI_ZHI_MAP.get(day_zhi, 0)

    # 计算偏移量（从起始日支顺推到当前日支）
    offset = (day_index - start_index) % 12

    # 返回对应的值神
    return ZHI_SHEN[offset]


def generate_zhi_shen_data():
    """
    生成数据：日期 -> 值神
    使用干支历数据作为基准
    """
    ganzhi_data = get_ganzhi_data()

    # 构建日期到干支的映射
    ganzhi_dict = {}
    for record in ganzhi_data:
        if len(record) >= 4:
            date_str = record[0]
            month_ganzhi = record[2]
            day_ganzhi = record[3]

            # 提取月支（天干地支的第二个字符）
            month_zhi = month_ganzhi[-1] if month_ganzhi else ''
            # 提取日支
            day_zhi = day_ganzhi[-1] if day_ganzhi else ''

            date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            ganzhi_dict[date_obj] = {
                'month_zhi': month_zhi,
                'day_zhi': day_zhi
            }

    start_date = datetime.date(2017, 1, 1)
    end_date = datetime.date(datetime.datetime.now().year + 2, 12, 31)

    current_date = start_date
    data_rows = []
    total_days = (end_date - start_date).days + 1

    with tqdm(total=total_days, desc="生成十二值神数据") as pbar:
        while current_date <= end_date:
            info = ganzhi_dict.get(current_date, {})
            month_zhi = info.get('month_zhi', '子')
            day_zhi = info.get('day_zhi', '子')

            # 计算值神
            zhi_shen = get_zhi_shen_for_date(month_zhi, day_zhi)

            date_str = current_date.strftime('%Y-%m-%d')
            data_rows.append({
                '日期': date_str,
                '值神': zhi_shen
            })

            current_date += datetime.timedelta(days=1)
            pbar.update(1)

    df = pd.DataFrame(data_rows)
    df = df.astype("string")
    return df


def export_zhi_shen_to_parquet():
    """导出值神数据为 Parquet 文件"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filepath = os.path.join(project_root, "data", "astro_data", "十二值神.parquet")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df = generate_zhi_shen_data()
    df.to_parquet(filepath, index=False, engine="pyarrow")
    print(f"十二值神数据已导出到: {filepath} （共 {len(df)} 行）")


def main():
    export_zhi_shen_to_parquet()


if __name__ == "__main__":
    if '..' not in sys.path:
        sys.path.append('..')
    if '.' not in sys.path:
        sys.path.append('.')

    try:
        from 干支历 import get_ganzhi_data, calculate_day_ganzhi
    except ImportError as e:
        print(f"导入模块时出错: {e}")
        sys.exit(1)

    main()