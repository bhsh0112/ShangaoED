#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频裁剪脚本
将输入的视频文件裁剪成指定时长的小段，保存到指定目录
"""

import os
import sys
import argparse
import cv2
from pathlib import Path


def get_video_info(video_path):
    """
    获取视频信息
    
    Args:
        video_path (str): 视频文件路径
        
    Returns:
        tuple: (fps, width, height, total_frames, duration) 或 None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    return fps, width, height, total_frames, duration


def split_video(video_path, output_dir, segment_duration=20.0):
    """
    将视频分割成多个片段
    
    Args:
        video_path (str): 原始视频文件路径
        output_dir (str): 输出目录
        segment_duration (float): 每个片段的时长（秒），默认20秒
        
    Returns:
        list: 分割后的视频文件路径列表
    """
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return []
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            print("❌ 无法获取视频帧率")
            cap.release()
            return []
        
        # 计算片段数量
        frames_per_segment = int(fps * segment_duration)
        segment_count = (total_frames + frames_per_segment - 1) // frames_per_segment
        total_duration = total_frames / fps
        
        print(f"\n📹 视频信息:")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.2f} fps")
        print(f"   总帧数: {total_frames}")
        print(f"   总时长: {total_duration:.2f} 秒")
        print(f"   片段时长: {segment_duration} 秒")
        print(f"   将生成: {segment_count} 个片段\n")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取视频文件名和扩展名
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_ext = os.path.splitext(video_path)[1]
        
        segment_files = []
        # 使用H.264编码器（更好的兼容性）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        print(f"🔄 开始分割视频...")
        
        for segment_idx in range(segment_count):
            output_filename = f"{video_name}_segment_{segment_idx+1:03d}{video_ext}"
            output_path = os.path.join(output_dir, output_filename)
            segment_files.append(output_path)
            
            # 创建视频写入器
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"❌ 无法创建输出文件: {output_path}")
                continue
            
            frames_written = 0
            while frames_written < frames_per_segment:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frames_written += 1
            
            out.release()
            
            # 计算进度
            progress = (segment_idx + 1) / segment_count * 100
            print(f"   ✅ [{progress:5.1f}%] 片段 {segment_idx+1}/{segment_count}: {output_filename}")
            
            if not ret:
                break
        
        cap.release()
        
        print(f"\n✨ 视频分割完成！共生成 {len(segment_files)} 个片段")
        print(f"📁 输出目录: {os.path.abspath(output_dir)}\n")
        
        return segment_files
        
    except Exception as e:
        print(f"❌ 分割视频时出错: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """
    主函数
    解析命令行参数并执行视频分割
    """
    parser = argparse.ArgumentParser(
        description='将视频文件裁剪成指定时长的小段',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 将视频裁剪成20秒的小段（默认）
  python video_splitter.py input.mp4 output_dir/
  
  # 指定片段时长为30秒
  python video_splitter.py input.mp4 output_dir/ --duration 30
  
  # 使用绝对路径
  python video_splitter.py /path/to/video.mp4 /path/to/output/
        """
    )
    
    parser.add_argument('input', type=str, help='输入视频文件路径')
    parser.add_argument('output_dir', type=str, help='输出目录路径')
    parser.add_argument('--duration', type=float, default=20.0,
                       help='每个片段的时长（秒），默认: 20.0')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入视频文件不存在: {args.input}")
        sys.exit(1)
    
    # 检查是否为视频文件
    video_ext = os.path.splitext(args.input)[1].lower()
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v']
    if video_ext not in supported_formats:
        print(f"⚠️  警告: 文件扩展名 {video_ext} 可能不是常见的视频格式")
        print(f"   支持的格式: {', '.join(supported_formats)}")
        response = input("   是否继续? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # 获取视频信息（用于验证）
    video_info = get_video_info(args.input)
    if video_info is None:
        print("❌ 错误: 无法读取视频文件，请检查文件是否有效")
        sys.exit(1)
    
    fps, width, height, total_frames, duration = video_info
    
    # 检查片段时长是否合理
    if args.duration <= 0:
        print("❌ 错误: 片段时长必须大于0")
        sys.exit(1)
    
    if args.duration > duration:
        print(f"⚠️  警告: 片段时长 ({args.duration}秒) 大于视频总时长 ({duration:.2f}秒)")
        print("   将只生成1个片段（完整视频）")
    
    # 执行视频分割
    segment_files = split_video(args.input, args.output_dir, args.duration)
    
    if not segment_files:
        print("❌ 视频分割失败")
        sys.exit(1)
    
    print("🎉 完成！")


if __name__ == '__main__':
    main()

