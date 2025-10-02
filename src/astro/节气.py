#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from datetime import timedelta, datetime, date
import pandas as pd
from skyfield import almanac
from skyfield import almanac_east_asia as almanac_ea
from skyfield.api import load

# 定义节气列表：节（月份分界点）
jie_only = ['立春', '惊蛰', '清明', '立夏', '芒种', '小暑', 
           '立秋', '白露', '寒露', '立冬', '大雪', '小寒']

# 定义节气列表：气（月份中点）
qi_only = ['雨水', '春分', '谷雨', '小满', '夏至', '大暑', 
          '处暑', '秋分', '霜降', '冬至', '大寒', '小雪']

def calculate_solar_terms_2017_now():
    """计算从2017年到当前日期往后推两年的二十四节气（包括节和气）"""
    # 加载天体历数据和时间尺度
    eph = load('de421.bsp')
    ts = load.timescale()
    t0 = ts.utc(2016, 12, 1)
    # 设置时间范围：从2016年12月1日到当前年份往后推两年的12月31日
    current_date = datetime.now()
    end_year = current_date.year + 2  # 包含未来两年的节气
    t1 = ts.utc(end_year, 12, 31)  # 时间范围结束于目标年份的12月31日
    # 使用skyfield库查找指定时间范围内所有离散的节气时间点
    # t: 节气时刻的时间数组，tm: 对应的节气索引数组（0-23代表24个节气）
    t, tm = almanac.find_discrete(t0, t1, almanac_ea.solar_terms(eph))
    solar_terms = almanac_ea.SOLAR_TERMS_ZHS
    result = []
    for term_index, time in zip(tm, t):
        term_name = solar_terms[term_index]
        utc_time = time.utc_datetime()
        # 转换为北京时间（UTC+8）
        beijing_time = utc_time + timedelta(hours=8)
        # 使用date对象而不是字符串
        date_obj = beijing_time.date()
        result.append((date_obj, term_name))
    return result

def classify_solar_terms(solar_terms_data):
    """将节气数据分类为"节"和"气"两类"""
    jie_data = []
    qi_data = []
    # 遍历节气数据，根据名称分类到"节"或"气"
    for name, date in solar_terms_data:
        if name in jie_only:
            jie_data.append((name, date))
        elif name in qi_only:
            qi_data.append((name, date))
    return jie_data, qi_data

def export_solar_terms_to_parquet():
    """将计算得到的二十四节气数据导出到Parquet文件，包含节气名称和对应的北京时间"""
    print("开始计算二十四节气...")
    solar_terms_2017_now = calculate_solar_terms_2017_now()
    
    # 转换为DataFrame
    df = pd.DataFrame(solar_terms_2017_now, columns=['日期', '节气名称'])
    
    # 在导出前将日期对象转换为字符串
    df['日期'] = df['日期'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, date) else x)
    
    # 设置导出文件路径到项目根目录下的data/astro_data目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filename = "节气.parquet"
    filepath = os.path.join(project_root, "data", "astro_data", filename)
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存为Parquet格式
    df.to_parquet(filepath, index=False)
    
    print("节气数据计算完成！")
    print(f"二十四节气数据已导出到: {filepath}")

def main():
    """主函数：执行节气数据导出功能"""
    export_solar_terms_to_parquet()

if __name__ == "__main__":
    main()