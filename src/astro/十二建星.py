#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 十二建星 数据，范围：2017-01-01 .. (当前年份 + 2)-12-31
逻辑：建星 = ((月支索引 - 日支索引 + 12) % 12) 对应十二建星
输出 Parquet： 项目根目录/data/astro_data/十二建星.parquet
字段：
- 日期（字符串 YYYY-MM-DD）
- 月支（干支历中的月支）
- 日支（干支历中的日支）
- 建星
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
di_zhi = {
    '子': 0, '丑': 1, '寅': 2, '卯': 3, '辰': 4, '巳': 5,
    '午': 6, '未': 7, '申': 8, '酉': 9, '戌': 10, '亥': 11
}

# 十二建星
jian_xing = ['建', '除', '满', '平', '定', '执', '破', '危', '成', '收', '开', '闭']

def get_day_branch(date: datetime.date) -> str:
    """返回日支"""
    date_str = date.strftime('%Y-%m-%d')
    day_ganzhi = calculate_day_ganzhi(date_str)
    return day_ganzhi[1] if len(day_ganzhi) > 1 else '子'

def generate_jian_xing_data():
    """生成数据，列为：日期、月支、日支、建星"""
    ganzhi_data = get_ganzhi_data()

    # 创建日期映射，兼容返回 4 或 6 个元素
    ganzhi_dict = {}
    for record in ganzhi_data:
        if len(record) >= 4:
            date_str, year_ganzhi, month_ganzhi, day_ganzhi = record[:4]
            month_zhi = record[5] if len(record) > 5 else (month_ganzhi[-1] if month_ganzhi else '')
        else:
            continue
        # 使用date对象而不是字符串
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        ganzhi_dict[date_obj] = {
            'month_zhi': month_zhi,
            'day_zhi': day_ganzhi[-1] if day_ganzhi else ''
        }

    start_date = datetime.date(2017, 1, 1)
    end_date = datetime.date(datetime.datetime.now().year + 2, 12, 31)

    current_date = start_date
    data_rows = []
    total_days = (end_date - start_date).days + 1

    with tqdm(total=total_days, desc="生成十二建星数据") as pbar:
        while current_date <= end_date:
            # 使用date对象而不是字符串
            info = ganzhi_dict.get(current_date, {})
            month_zhi = info.get('month_zhi', '')
            day_zhi = info.get('day_zhi', '')

            # 建星逻辑：月支 - 日支 + 12 % 12
            month_index = di_zhi.get(month_zhi, 0)
            day_index = di_zhi.get(day_zhi, 0)
            jx_index = ( (day_index - month_index + 12) % 12 )
            jx = jian_xing[jx_index]

            # 在导出前将日期对象转换为字符串
            date_str = current_date.strftime('%Y-%m-%d')
            data_rows.append({
                '日期': date_str,
                '建星': jx
            })

            current_date += datetime.timedelta(days=1)
            pbar.update(1)

    df = pd.DataFrame(data_rows)
    df = df.astype("string")
    return df

def export_jian_xing_to_parquet():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filepath = os.path.join(project_root, "data", "astro_data", "十二建星.parquet")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df = generate_jian_xing_data()
    df.to_parquet(filepath, index=False, engine="pyarrow")
    print(f"十二建星数据已导出到: {filepath} （共 {len(df)} 行）")

def main():
    export_jian_xing_to_parquet()

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
