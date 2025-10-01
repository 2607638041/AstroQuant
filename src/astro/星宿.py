#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import pandas as pd

def get_xiuxiu_list():
    """
    返回二十八星宿列表，按顺序排列
    """
    return [
        "角", "亢", "氐", "房", "心", "尾", "箕",
        "斗", "牛", "女", "虚", "危", "室", "壁",
        "奎", "娄", "胃", "昴", "毕", "觜", "参",
        "井", "鬼", "柳", "星", "张", "翼", "轸"
    ]

def calculate_xiuxiu(target_date, reference_date=datetime.date(2025, 6, 26), reference_xiuxiu="角"):
    """
    根据基准日期和星宿计算指定日期的星宿
    
    参数:
    target_date: 目标日期 (datetime.date)
    reference_date: 基准日期 (datetime.date)，默认为2025年6月26日
    reference_xiuxiu: 基准日期的星宿，默认为"角"
    
    返回:
    指定日期的星宿名称
    """
    xiuxiu_list = get_xiuxiu_list()
    
    # 获取基准星宿在列表中的索引
    reference_index = xiuxiu_list.index(reference_xiuxiu)
    
    # 计算日期差
    delta_days = (reference_date - target_date).days
    
    # 使用公式 ((基准日-指定日)%28+28)%28 计算偏移量
    offset = ((delta_days % 28) + 28) % 28
    
    # 计算目标日期的星宿索引
    target_index = (reference_index - offset + 28) % 28
    
    return xiuxiu_list[target_index]

def generate_xiuxiu_for_period(start_date, end_date, reference_date=datetime.date(2025, 6, 26), reference_xiuxiu="角"):
    """
    生成指定时间段内每日的星宿
    
    参数:
    start_date: 开始日期 (datetime.date)
    end_date: 结束日期 (datetime.date)
    reference_date: 基准日期 (datetime.date)
    reference_xiuxiu: 基准日期的星宿
    
    返回:
    字典，键为日期，值为星宿
    """
    result = {}
    current_date = start_date
    xiuxiu_list = get_xiuxiu_list()
    
    # 获取基准星宿在列表中的索引
    reference_index = xiuxiu_list.index(reference_xiuxiu)
    
    # 计算基准日期的星宿索引
    base_date = datetime.date(2025, 6, 26)
    base_index = xiuxiu_list.index("角")
    
    while current_date <= end_date:
        # 计算日期差
        delta_days = (base_date - current_date).days
        
        # 使用公式计算偏移量
        offset = ((delta_days % 28) + 28) % 28
        
        # 计算目标日期的星宿索引
        target_index = (base_index - offset + 28) % 28
        
        result[current_date] = xiuxiu_list[target_index]
        current_date += datetime.timedelta(days=1)
    
    return result

def export_xiuxiu_to_parquet():
    """
    将星宿数据导出到Parquet文件
    """
    # 生成从2017年到当前日期再加两年的星宿数据
    start_date = datetime.date(2017, 1, 1)
    end_date = datetime.date.today() + datetime.timedelta(days=365*2)  # 当前日期加两年
    
    print("开始计算每日星宿数据...")
    xiuxiu_data = generate_xiuxiu_for_period(start_date, end_date)
    
    # 转换为DataFrame
    df = pd.DataFrame([(date.strftime('%Y-%m-%d'), xiuxiu + "宿") for date, xiuxiu in xiuxiu_data.items()], 
                      columns=['日期', '星宿'])
    
    # 设置导出文件路径到项目根目录下的data/astro_data目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filename = "星宿.parquet"
    filepath = os.path.join(project_root, "data", "astro_data", filename)
    
    # 创建导出目录
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存为Parquet格式
    df.to_parquet(filepath, index=False)
    
    print("每日星宿数据计算完成！")
    print(f"每日星宿数据已导出到: {filepath}")

# 主程序部分
if __name__ == "__main__":
    export_xiuxiu_to_parquet()