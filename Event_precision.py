#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件统计脚本
统计shangao_result文件夹中各事件类型的时间>0s的比例
"""

import os
import re
from collections import defaultdict

def parse_event_file(file_path):
    """
    解析事件文件，提取各事件类型的时间
    
    Args:
        file_path (str): 事件文件路径
        
    Returns:
        dict: 包含各事件类型时间的字典
    """
    event_times = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 使用正则表达式提取事件时间
        # 匹配格式: "Event Name: X.XXs"，排除标题行
        pattern = r'^([^:=\n]+):\s*(\d+\.?\d*)s$'
        matches = re.findall(pattern, content, re.MULTILINE)
        
        for event_name, time_str in matches:
            event_name = event_name.strip()
            time_value = float(time_str)
            event_times[event_name] = time_value
            
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")
        
    return event_times

def analyze_events(base_dir):
    """
    分析所有事件文件，按文件夹分别统计对应事件类型的时间>0s的比例
    
    Args:
        base_dir (str): shangao_result文件夹路径
        
    Returns:
        dict: 统计结果
    """
    # 存储每个文件夹的统计结果
    folder_results = {}
    total_files = 0
    
    # 遍历所有子文件夹
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        print(f"正在处理文件夹: {subdir}")
        
        # 确定该文件夹对应的事件类型
        event_type = None
        if subdir == "parking":
            event_type = "Parking"
        elif subdir == "People":
            event_type = "People"
        elif subdir == "TrafficJam":
            event_type = "Traffic Jam"
        
        print(f"  对应事件类型: {event_type}")
        
        if event_type is None:
            print(f"  跳过文件夹 {subdir}，未找到对应的事件类型")
            continue
        
        # 存储该文件夹中对应事件类型的时间数据
        event_times = []
        folder_file_count = 0
        
        # 遍历子文件夹中的所有txt文件
        for filename in os.listdir(subdir_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(subdir_path, filename)
                parsed_events = parse_event_file(file_path)
                
                if parsed_events and event_type in parsed_events:
                    folder_file_count += 1
                    total_files += 1
                    event_times.append(parsed_events[event_type])
        
        # 计算该文件夹的统计结果
        if event_times:
            positive_count = sum(1 for t in event_times if t > 0)
            ratio = positive_count / len(event_times)
            
            print(f"  {event_type}: {len(event_times)} 个文件, {positive_count} 个时间>0s, 比例: {ratio:.4f}")
            
            folder_results[event_type] = {
                'folder_name': subdir,
                'total_files': len(event_times),
                'positive_time_files': positive_count,
                'ratio': ratio
            }
        else:
            print(f"  {event_type}: 没有找到有效数据")
    
    return folder_results, total_files

def main():
    """
    主函数
    """
    base_dir = "shangao_result"
    
    if not os.path.exists(base_dir):
        print(f"错误: 找不到文件夹 {base_dir}")
        return
    
    print("开始分析事件数据...")
    results, total_files = analyze_events(base_dir)
    
    print(f"\n总共处理了 {total_files} 个文件")
    print("\n=== 各文件夹对应事件类型统计结果 ===")
    
    # 计算平均比例
    total_ratio = 0
    event_count = 0
    
    for event_name, stats in results.items():
        ratio = stats['ratio']
        total_ratio += ratio
        event_count += 1
        
        print(f"{event_name} (来自 {stats['folder_name']} 文件夹):")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  时间>0s的文件数: {stats['positive_time_files']}")
        print(f"  比例: {ratio:.4f} ({ratio*100:.2f}%)")
        print()
    
    # 计算并显示平均比例
    if event_count > 0:
        average_ratio = total_ratio / event_count
        print(f"=== 平均比例 ===")
        print(f"所有事件类型的平均比例: {average_ratio:.4f} ({average_ratio*100:.2f}%)")
    
    # 保存结果到文件
    with open("event_statistics_result.txt", "w", encoding="utf-8") as f:
        f.write("事件统计结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"总共处理文件数: {total_files}\n\n")
        
        for event_name, stats in results.items():
            ratio = stats['ratio']
            f.write(f"{event_name} (来自 {stats['folder_name']} 文件夹):\n")
            f.write(f"  总文件数: {stats['total_files']}\n")
            f.write(f"  时间>0s的文件数: {stats['positive_time_files']}\n")
            f.write(f"  比例: {ratio:.4f} ({ratio*100:.2f}%)\n\n")
        
        if event_count > 0:
            average_ratio = total_ratio / event_count
            f.write(f"平均比例: {average_ratio:.4f} ({average_ratio*100:.2f}%)\n")
    
    print("结果已保存到 event_statistics_result.txt")

if __name__ == "__main__":
    main()
