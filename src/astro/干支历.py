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

# 天干列表
tian_gan_list = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']

# 地支列表
di_zhi_list = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']

# 六十甲子 (甲子0~癸亥59)
liu_shi_jia_zi = [
    '甲子', '乙丑', '丙寅', '丁卯', '戊辰', '己巳', '庚午', '辛未', '壬申', '癸酉',
    '甲戌', '乙亥', '丙子', '丁丑', '戊寅', '己卯', '庚辰', '辛巳', '壬午', '癸未',
    '甲申', '乙酉', '丙戌', '丁亥', '戊子', '己丑', '庚寅', '辛卯', '壬辰', '癸巳',
    '甲午', '乙未', '丙申', '丁酉', '戊戌', '己亥', '庚子', '辛丑', '壬寅', '癸卯',
    '甲辰', '乙巳', '丙午', '丁未', '戊申', '己酉', '庚戌', '辛亥', '壬子', '癸丑',
    '甲寅', '乙卯', '丙辰', '丁巳', '戊午', '己未', '庚申', '辛酉', '壬戌', '癸亥'
]

# 基准日期常量
BASE_DATE = datetime.date(2025, 2, 24)  # 基准日期
BASE_DATE_GANZHI_INDEX = 0  # 甲子在liu_shi_jia_zi列表中的索引

# 节气与月支对应关系
jieqi_yuezi = {
    '立春': '寅', '惊蛰': '卯', '清明': '辰', '立夏': '巳', '芒种': '午', '小暑': '未',
    '立秋': '申', '白露': '酉', '寒露': '戌', '立冬': '亥', '大雪': '子', '小寒': '丑'
}

# 五虎遁年干推月干口诀
wu_hu_dun = {
    '甲': '丙', '己': '丙', '乙': '戊', '庚': '戊', '丙': '庚',
    '辛': '庚', '丁': '壬', '壬': '壬', '戊': '甲', '癸': '甲'
}


# 干支计算相关函数
def calculate_day_ganzhi(target_date_str):
    """根据基准日期计算指定日期的日柱干支"""
    # 修改日期解析格式以适配新的日期格式
    try:
        target_date = datetime.datetime.strptime(target_date_str, '%Y-%m-%d').date()
    except ValueError:
        target_date = datetime.datetime.strptime(target_date_str, '%Y年%m月%d日').date()
    delta_days = (target_date - BASE_DATE).days
    day_index = (BASE_DATE_GANZHI_INDEX + delta_days) % 60
    day_index = (day_index + 60) % 60
    return liu_shi_jia_zi[day_index]


def calculate_year_ganzhi(year):
    """计算年柱干支"""
    gan_index = (year % 10 + 6) % 10
    zhi_index = (year + 8) % 12
    return tian_gan_list[gan_index] + di_zhi_list[zhi_index]


def get_yin_month_gan(year_gan):
    """使用五虎遁方法确定寅月（正月）的月干"""
    return wu_hu_dun.get(year_gan, '')


def calculate_month_gan(year_gan, month_zhi):
    """根据年干和月支计算月干，形成完整的月柱"""
    yin_month_gan = get_yin_month_gan(year_gan)
    yin_month_gan_index = tian_gan[yin_month_gan]
    target_month_index = di_zhi[month_zhi]
    yin_month_index = di_zhi['寅']
    offset = (target_month_index - yin_month_index) % 12
    month_gan_index = (yin_month_gan_index + offset) % 10
    month_gan = tian_gan_list[month_gan_index]
    return month_gan + month_zhi


# 数据处理相关函数
def get_ganzhi_data():
    """获取干支历数据并返回"""
    solar_terms_data = calculate_solar_terms_2017_now()
    jie_data = [(name, date) for name, date in solar_terms_data if name in jie_only]

    jieqi_dict = {}
    for name, date_str in jie_data:
        # 修改日期解析格式以适配新的日期格式
        try:
            date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            date_obj = datetime.datetime.strptime(date_str, '%Y年%m月%d日')
        # 节气变更后的第二天才开始使用新的月支
        jieqi_dict[date_obj.date() + datetime.timedelta(days=1)] = name

    month_data = []
    current_year = datetime.datetime.now().year
    start_date = datetime.date(2017, 1, 1)
    # 将截止日期改为当前日期往后推两年
    end_date = datetime.date(current_year + 2, 12, 31)
    sorted_jieqi_dates = sorted(jieqi_dict.keys())
    lichun_dict = {d: jieqi_dict[d] for d in sorted_jieqi_dates if jieqi_dict[d] == '立春'}

    current_date = start_date  # 已正确使用datetime.date类
    while current_date <= end_date:
        current_date_str = f"{current_date.year}年{current_date.month:02d}月{current_date.day:02d}日"
        current_date_formatted = f"{current_date.year}-{current_date.month:02d}-{current_date.day:02d}"

        # 1. 先计算日柱
        try:
            day_ganzhi = calculate_day_ganzhi(current_date_str)
        except:
            day_ganzhi = "未知"

        # 2. 然后计算年柱干支
        year_for_ganzhi = current_date.year
        lichun_this_year = None
        for lichun_date in lichun_dict.keys():
            if lichun_date.year == current_date.year:
                lichun_this_year = lichun_date
                break

        if lichun_this_year and current_date < lichun_this_year:
            year_for_ganzhi = current_date.year - 1

        year_ganzhi = calculate_year_ganzhi(year_for_ganzhi)

        # 3. 最后计算月柱
        recent_jieqi_date = None
        for jieqi_date in sorted_jieqi_dates:
            if jieqi_date <= current_date:
                recent_jieqi_date = jieqi_date
            else:
                break

        if recent_jieqi_date and jieqi_dict[recent_jieqi_date] in jieqi_yuezi:
            yuezhi = jieqi_yuezi[jieqi_dict[recent_jieqi_date]]
            month_ganzhi = calculate_month_gan(year_ganzhi[0], yuezhi)
        else:
            month_ganzhi = "未知"

        # 只保留日期、年柱、月柱、日柱，移除年支和月支
        month_data.append((current_date_formatted, year_ganzhi, month_ganzhi, day_ganzhi))
        current_date += datetime.timedelta(days=1)

    return month_data


# 数据导出相关函数
def export_month_zhi_to_parquet():
    """导出信息到Parquet文件"""
    print("开始计算干支历数据...")
    month_data = get_ganzhi_data()
    
    # 转换为DataFrame，只包含需要的列
    df = pd.DataFrame(month_data, columns=['日期', '年柱', '月柱', '日柱'])
    
    # 设置导出文件路径到项目根目录下的data/astro_data目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filename = f"干支历.parquet"
    filepath = os.path.join(project_root, "data", "astro_data", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存为Parquet格式
    df.to_parquet(filepath, index=False)
    
    print("干支历数据计算完成！")
    print(f"干支历数据已导出到: {filepath}")


# 程序入口
def main():
    """主函数"""
    export_month_zhi_to_parquet()


if __name__ == "__main__":
    main()