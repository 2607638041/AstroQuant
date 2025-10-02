#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import datetime
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from 节气 import calculate_solar_terms_2017_now, jie_only


# 天干 (甲0~癸9)
tian_gan = {
    '甲': 0, '乙': 1, '丙': 2, '丁': 3, '戊': 4,
    '己': 5, '庚': 6, '辛': 7, '壬': 8, '癸': 9
}

# 十二地支 (子0~丑11)
di_zhi = {
    '子': 0, '丑': 1, '寅': 2, '卯': 3, '辰': 4, '巳': 5,
    '午': 6, '未': 7, '申': 8, '酉': 9, '戌': 10, '亥': 11
}

tian_gan_list = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
di_zhi_list = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']

liu_shi_jia_zi = [
    '甲子', '乙丑', '丙寅', '丁卯', '戊辰', '己巳', '庚午', '辛未', '壬申', '癸酉',
    '甲戌', '乙亥', '丙子', '丁丑', '戊寅', '己卯', '庚辰', '辛巳', '壬午', '癸未',
    '甲申', '乙酉', '丙戌', '丁亥', '戊子', '己丑', '庚寅', '辛卯', '壬辰', '癸巳',
    '甲午', '乙未', '丙申', '丁酉', '戊戌', '己亥', '庚子', '辛丑', '壬寅', '癸卯',
    '甲辰', '乙巳', '丙午', '丁未', '戊申', '己酉', '庚戌', '辛亥', '壬子', '癸丑',
    '甲寅', '乙卯', '丙辰', '丁巳', '戊午', '己未', '庚申', '辛酉', '壬戌', '癸亥'
]

BASE_DATE = datetime.date(2025, 2, 24)
BASE_DATE_GANZHI_INDEX = 0

jieqi_yuezi = {
    '立春': '寅', '惊蛰': '卯', '清明': '辰', '立夏': '巳', '芒种': '午', '小暑': '未',
    '立秋': '申', '白露': '酉', '寒露': '戌', '立冬': '亥', '大雪': '子', '小寒': '丑'
}

wu_hu_dun = {
    '甲': '丙', '己': '丙', '乙': '戊', '庚': '戊', '丙': '庚',
    '辛': '庚', '丁': '壬', '壬': '壬', '戊': '甲', '癸': '甲'
}


def calculate_day_ganzhi(target_date_str):
    try:
        target_date = datetime.datetime.strptime(target_date_str, '%Y-%m-%d').date()
    except ValueError:
        target_date = datetime.datetime.strptime(target_date_str, '%Y年%m月%d日').date()
    delta_days = (target_date - BASE_DATE).days
    day_index = (BASE_DATE_GANZHI_INDEX + delta_days) % 60
    return liu_shi_jia_zi[day_index]


def calculate_year_ganzhi(year):
    gan_index = (year % 10 + 6) % 10
    zhi_index = (year + 8) % 12
    return tian_gan_list[gan_index] + di_zhi_list[zhi_index]


def get_yin_month_gan(year_gan):
    return wu_hu_dun.get(year_gan, '')


def calculate_month_gan(year_gan, month_zhi):
    yin_month_gan = get_yin_month_gan(year_gan)
    yin_month_gan_index = tian_gan[yin_month_gan]
    target_month_index = di_zhi[month_zhi]
    yin_month_index = di_zhi['寅']
    offset = (target_month_index - yin_month_index) % 12
    month_gan_index = (yin_month_gan_index + offset) % 10
    return tian_gan_list[month_gan_index] + month_zhi


def get_ganzhi_data():
    solar_terms_data = calculate_solar_terms_2017_now()
    
    # 使用更宽松的匹配方式来处理可能的字符编码问题
    jie_data = []
    unmatched_names = set()
    
    # 注意：现在solar_terms_data返回的是(date对象, 节气名称)的元组
    for date_obj, name in solar_terms_data:
        matched = False
        # 精确匹配
        if name in jie_only:
            jie_data.append((name, date_obj))
            matched = True
        else:
            # 模糊匹配
            for jie_name in jie_only:
                if name.strip() == jie_name.strip():
                    jie_data.append((jie_name, date_obj))
                    matched = True
                    break
        
        # 如果仍未匹配，尝试更宽松的匹配方式
        if not matched:
            for jie_name in jie_only:
                # 忽略空格和特殊字符
                clean_name = ''.join(c for c in name if c.isalnum())
                clean_jie_name = ''.join(c for c in jie_name if c.isalnum())
                if clean_name == clean_jie_name:
                    jie_data.append((jie_name, date_obj))
                    matched = True
                    break
        
        # 最后的后备方案：记录未匹配的名称
        if not matched:
            unmatched_names.add(name)
    
    # 如果没有匹配到任何节气数据，则使用所有节气数据进行测试
    if len(jie_data) == 0 and len(solar_terms_data) > 0:
        # 强制将前几个节气数据作为节处理
        for i, (date_obj, name) in enumerate(solar_terms_data[:min(24, len(solar_terms_data))]):
            jie_name = jie_only[i % len(jie_only)]
            jie_data.append((jie_name, date_obj))
        print(f"警告：未能匹配到节气名称，使用后备方案。未匹配的名称: {list(unmatched_names)[:5]}")

    # 构建节气字典，键为日期字符串，值为节气名称
    jieqi_dict = {}
    for name, date_obj in jie_data:
        # 确保日期格式统一，现在处理date对象
        if isinstance(date_obj, datetime.date):
            date_str = date_obj.strftime('%Y-%m-%d')
            jieqi_dict[date_str] = name
        else:
            # 如果不是date对象，尝试转换为字符串
            jieqi_dict[str(date_obj)] = name

    month_data = []
    current_year = datetime.datetime.now().year
    start_date = datetime.date(2017, 1, 1)
    end_date = datetime.date(current_year + 2, 12, 31)
    
    # 创建日期字符串到节气名称的映射
    sorted_jieqi_dates = sorted([datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in jieqi_dict.keys()])
    lichun_dict = {}
    for d in sorted_jieqi_dates:
        date_str = d.strftime('%Y-%m-%d')
        if jieqi_dict.get(date_str) == '立春':
            lichun_dict[d] = '立春'

    current_date = start_date
    while current_date <= end_date:
        current_date_str = f"{current_date.year}年{current_date.month:02d}月{current_date.day:02d}日"
        current_date_formatted = f"{current_date.year}-{current_date.month:02d}-{current_date.day:02d}"

        try:
            day_ganzhi = calculate_day_ganzhi(current_date_str)
        except:
            day_ganzhi = "未知"

        year_for_ganzhi = current_date.year
        lichun_this_year = None
        for lichun_date in lichun_dict.keys():
            if lichun_date.year == current_date.year:
                lichun_this_year = lichun_date
                break
        if lichun_this_year and current_date < lichun_this_year:
            year_for_ganzhi = current_date.year - 1
        year_ganzhi = calculate_year_ganzhi(year_for_ganzhi)

        recent_jieqi_date = None
        for jieqi_date in sorted_jieqi_dates:
            if jieqi_date <= current_date:
                recent_jieqi_date = jieqi_date
            else:
                break

        if recent_jieqi_date:
            # 获取该节气对应的月支
            recent_jieqi_date_str = recent_jieqi_date.strftime('%Y-%m-%d')
            jieqi_name = jieqi_dict.get(recent_jieqi_date_str)
            if jieqi_name and jieqi_name in jieqi_yuezi:
                yuezhi = jieqi_yuezi[jieqi_name]
                month_ganzhi = calculate_month_gan(year_ganzhi[0], yuezhi)
            else:
                month_ganzhi = "未知"
        else:
            month_ganzhi = "未知"

        # 获取当天的节气（但不再添加到结果中）
        # solar_term_today = jieqi_dict.get(current_date_formatted, "")

        # 只添加日期、年柱、月柱、日柱到结果中（移除了节气列）
        month_data.append((current_date_formatted, year_ganzhi, month_ganzhi, day_ganzhi))
        current_date += datetime.timedelta(days=1)

    return month_data


def export_month_zhi_to_parquet():
    print("开始计算干支历数据...")
    month_data = get_ganzhi_data()

    # 更新列名，移除节气列
    df = pd.DataFrame(month_data, columns=['日期', '年柱', '月柱', '日柱'])

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filename = "干支历.parquet"
    filepath = os.path.join(project_root, "data", "astro_data", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df.to_parquet(filepath, index=False)
    print("干支历数据计算完成！")
    print(f"干支历数据已导出到: {filepath}")


def main():
    export_month_zhi_to_parquet()


if __name__ == "__main__":
    main()
