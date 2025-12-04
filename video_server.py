#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频服务器脚本
输入视频文件路径和端口号，启动HTTP服务器提供视频播放服务
"""

import os
import sys
import argparse
import re
import socket
import cv2
from flask import Flask, send_file, Response, request
from werkzeug.serving import run_simple


# 视频传输块大小（字节），用于优化大视频传输
# 默认 256KB，可根据网络带宽调整
DEFAULT_CHUNK_SIZE = 256 * 1024  # 256KB


def get_local_ip():
    """
    获取本机IP地址
    
    Returns:
        str: 本机IP地址
    """
    try:
        # 连接到一个远程地址（不实际发送数据）来获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_video_duration(video_path):
    """
    获取视频时长（秒）
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        float: 视频时长（秒），如果获取失败返回0
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
            return duration
        return 0.0
    except Exception as e:
        print(f"获取视频时长失败: {e}")
        return 0.0


def create_app(video_path, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    创建Flask应用
    
    Args:
        video_path: 视频文件路径
        chunk_size: 视频传输块大小（字节），用于优化大视频传输
        
    Returns:
        Flask应用实例
    """
    app = Flask(__name__)
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_filename = os.path.basename(video_path)
    
    @app.route('/')
    def index():
        """主页面，显示视频播放器"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>视频播放器</title>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    background-color: #1a1a1a;
                    color: #fff;
                    font-family: Arial, sans-serif;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                }}
                h1 {{
                    margin-bottom: 20px;
                }}
                video {{
                    max-width: 90%;
                    max-height: 80vh;
                    border: 2px solid #444;
                    border-radius: 8px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                }}
                .info {{
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #2a2a2a;
                    border-radius: 5px;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <h1>视频播放器</h1>
            <video controls autoplay>
                <source src="/video/{video_filename}" type="video/mp4">
                您的浏览器不支持视频播放。
            </video>
            <div class="info">
                <p>视频文件: {video_filename}</p>
                <p>时长: {get_video_duration(video_path):.1f}秒</p>
            </div>
        </body>
        </html>
        """
        return html
    
    @app.route('/video/<filename>')
    def serve_video(filename):
        """
        提供视频文件服务，支持范围请求（Range requests）用于视频流
        优化大视频传输：使用更大的块大小和缓存头
        
        Args:
            filename: 视频文件名
            
        Returns:
            Response: 视频文件响应
        """
        # 检查文件名是否匹配
        if filename != video_filename:
            return "File not found", 404
        
        file_path = os.path.join(video_dir, filename)
        if not os.path.exists(file_path):
            return "File not found", 404
        
        size = os.path.getsize(file_path)
        
        # 支持HTTP Range请求，用于视频流播放
        range_header = request.headers.get('Range', None)
        if not range_header:
            # 非Range请求，直接发送文件（使用send_file优化）
            response = send_file(file_path, mimetype='video/mp4')
            # 添加缓存和优化头
            response.headers.add('Accept-Ranges', 'bytes')
            response.headers.add('Cache-Control', 'public, max-age=3600')
            response.headers.add('Content-Length', str(size))
            return response
        
        # 解析Range请求
        byte_start = 0
        byte_end = size - 1
        
        match = re.search(r'(\d+)-(\d*)', range_header)
        if match:
            byte_start = int(match.group(1))
            if match.group(2):
                byte_end = int(match.group(2))
            else:
                # 如果只指定了起始位置，设置合理的结束位置
                # 允许客户端请求更大的块
                byte_end = min(byte_start + chunk_size - 1, size - 1)
        
        # 确保范围有效
        if byte_start >= size or byte_start < 0:
            return "Range Not Satisfiable", 416
        
        byte_end = min(byte_end, size - 1)
        length = byte_end - byte_start + 1
        
        def generate():
            """生成器函数，使用更大的块大小传输数据"""
            with open(file_path, 'rb') as f:
                f.seek(byte_start)
                remaining = length
                while remaining:
                    # 使用配置的chunk_size，但不超过剩余数据量
                    read_size = min(chunk_size, remaining)
                    chunk = f.read(read_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        rv = Response(generate(), 206, mimetype='video/mp4', direct_passthrough=True)
        rv.headers.add('Content-Range', f'bytes {byte_start}-{byte_end}/{size}')
        rv.headers.add('Accept-Ranges', 'bytes')
        rv.headers.add('Content-Length', str(length))
        # 添加缓存控制，提高传输效率
        rv.headers.add('Cache-Control', 'public, max-age=3600')
        return rv
    
    return app


def main():
    """
    主函数
    解析命令行参数，启动视频服务器
    """
    parser = argparse.ArgumentParser(
        description='启动视频服务器，提供视频播放服务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python video_server.py /path/to/video.mp4 8080
  python video_server.py data/test/test.mp4 5000
        """
    )
    parser.add_argument('video', type=str, help='视频文件路径')
    parser.add_argument('port', type=int, help='服务器端口号')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                       help='服务器主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                       help=f'视频传输块大小（字节，默认: {DEFAULT_CHUNK_SIZE // 1024}KB，建议范围: 64KB-1MB）')
    
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在: {args.video}")
        sys.exit(1)
    
    # 检查是否为视频文件
    video_ext = os.path.splitext(args.video)[1].lower()
    if video_ext not in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
        print(f"警告: 文件扩展名 {video_ext} 可能不是视频格式")
    
    # 获取视频时长
    duration = get_video_duration(args.video)
    if duration <= 0:
        print("错误: 无法获取视频时长，请检查视频文件是否有效")
        sys.exit(1)
    
    print(f"视频时长: {duration:.2f}秒")
    
    # 验证并调整chunk_size
    if args.chunk_size < 1024:
        print(f"警告: chunk_size ({args.chunk_size}) 太小，已调整为 64KB")
        args.chunk_size = 64 * 1024
    elif args.chunk_size > 10 * 1024 * 1024:
        print(f"警告: chunk_size ({args.chunk_size // 1024 // 1024}MB) 太大，已调整为 1MB")
        args.chunk_size = 1024 * 1024
    
    print(f"视频传输块大小: {args.chunk_size // 1024}KB")
    
    # 创建Flask应用
    app = create_app(args.video, args.chunk_size)
    
    # 获取本机IP
    local_ip = get_local_ip()
    
    # 输出访问URL
    print("\n" + "="*60)
    print("视频服务器已启动!")
    print("="*60)
    print(f"视频文件: {os.path.abspath(args.video)}")
    print(f"视频时长: {duration:.2f}秒")
    print(f"\n视频播放URL:")
    print(f"  本地: http://127.0.0.1:{args.port}/")
    print(f"  局域网: http://{local_ip}:{args.port}/")
    print("="*60)
    print("\n按 Ctrl+C 停止服务器\n")
    
    # 启动服务器
    try:
        run_simple(
            hostname=args.host,
            port=args.port,
            application=app,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
