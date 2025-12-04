#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频文件拷贝脚本
从指定目录的所有子目录中查找MP4视频文件，并拷贝到目标目录，使用指定前缀和序号重命名
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List


def find_mp4_files(source_dir: Path, dir_prefix: str = None) -> List[Path]:
    """
    递归查找指定目录下所有子目录中的MP4文件
    
    @param {Path} source_dir - 源目录路径
    @param {str} dir_prefix - 可选的子目录名前缀，只查找位于以此前缀开头的子目录中的MP4文件
    @returns {List[Path]} - MP4文件路径列表
    """
    mp4_files = []
    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        # 如果指定了目录前缀，检查当前目录是否位于以指定前缀开头的子目录中
        if dir_prefix is not None:
            # 获取相对于源目录的路径
            try:
                relative_path = root_path.relative_to(source_dir)
                # 如果是在源目录本身，跳过（只查找子目录中的文件）
                if relative_path == Path('.'):
                    continue
                # 获取第一级子目录名（相对于源目录的第一层目录）
                first_dir = relative_path.parts[0] if relative_path.parts else None
                # 只包含第一级子目录名以指定前缀开头的目录中的文件
                if first_dir is None or not first_dir.startswith(dir_prefix):
                    continue
            except ValueError:
                # 如果无法计算相对路径，跳过
                continue
        
        for file in files:
            if file.lower().endswith('.mp4'):
                mp4_files.append(root_path / file)
    return sorted(mp4_files)


def copy_videos_with_prefix(source_dir: str, target_dir: str, prefix: str, dir_prefix: str = None):
    """
    拷贝源目录下所有子目录中的MP4文件到目标目录，使用指定前缀和序号重命名
    
    @param {str} source_dir - 源目录路径
    @param {str} target_dir - 目标目录路径
    @param {str} prefix - 文件命名前缀
    @param {str} dir_prefix - 可选的子目录名前缀，只查找位于以此前缀开头的子目录中的MP4文件
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 检查源目录是否存在
    if not source_path.exists():
        raise ValueError(f"源目录不存在: {source_dir}")
    
    if not source_path.is_dir():
        raise ValueError(f"源路径不是目录: {source_dir}")
    
    # 创建目标目录（如果不存在）
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有MP4文件
    if dir_prefix:
        print(f"正在搜索 {source_dir} 下位于以 '{dir_prefix}' 开头的子目录中的MP4文件...")
    else:
        print(f"正在搜索 {source_dir} 下的所有MP4文件...")
    mp4_files = find_mp4_files(source_path, dir_prefix)
    
    if not mp4_files:
        if dir_prefix:
            print(f"未找到任何位于以 '{dir_prefix}' 开头的子目录中的MP4文件")
        else:
            print("未找到任何MP4文件")
        return
    
    print(f"找到 {len(mp4_files)} 个MP4文件")
    
    # 拷贝文件并重命名
    copied_count = 0
    for idx, mp4_file in enumerate(mp4_files, start=1):
        # 生成新文件名：前缀_序号.mp4
        new_filename = f"{prefix}_{idx}.mp4"
        target_file = target_path / new_filename
        
        try:
            # 拷贝文件
            shutil.copy2(mp4_file, target_file)
            print(f"[{idx}/{len(mp4_files)}] 已拷贝: {mp4_file.name} -> {new_filename}")
            copied_count += 1
        except Exception as e:
            print(f"错误: 拷贝 {mp4_file} 时出错: {e}")
    
    print(f"\n完成! 成功拷贝 {copied_count}/{len(mp4_files)} 个文件到 {target_dir}")


def main():
    """
    主函数，解析命令行参数并执行拷贝操作
    """
    parser = argparse.ArgumentParser(
        description='从指定目录的所有子目录中查找MP4视频文件，并拷贝到目标目录，使用指定前缀和序号重命名',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python copy_videos.py -s ./data/test -t ./output/videos -p video
  python copy_videos.py --source /path/to/source --target /path/to/target --prefix myvideo
  python copy_videos.py -s ./data/test -t ./output/videos -p video -f 停车
  (只查找位于以"停车"开头的子目录中的MP4文件)
        """
    )
    
    parser.add_argument(
        '-s', '--source',
        type=str,
        required=True,
        help='源目录路径（将从此目录的所有子目录中查找MP4文件）'
    )
    
    parser.add_argument(
        '-t', '--target',
        type=str,
        required=True,
        help='目标目录路径（MP4文件将被拷贝到此目录）'
    )
    
    parser.add_argument(
        '-p', '--prefix',
        type=str,
        required=True,
        help='文件命名前缀（生成的文件名格式：前缀_1.mp4, 前缀_2.mp4, ...）'
    )
    
    parser.add_argument(
        '-f', '--dir-prefix',
        type=str,
        default=None,
        help='可选的子目录名前缀过滤，只查找位于以此前缀开头的子目录中的MP4文件（例如：停车、拥堵等）'
    )
    
    args = parser.parse_args()
    
    try:
        copy_videos_with_prefix(args.source, args.target, args.prefix, args.dir_prefix)
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

