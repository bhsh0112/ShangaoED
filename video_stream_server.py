#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大视频流式服务器
使用 FFmpeg 进行实时转码，支持大视频文件的流式播放
适用于服务器无可视化界面、无法下载视频的场景
"""

import os
import sys
import subprocess
import threading
import socket
import argparse
import signal
from flask import Flask, Response, request
from werkzeug.serving import run_simple


def get_local_ip():
    """
    获取本机IP地址
    
    Returns:
        str: 本机IP地址
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def check_ffmpeg():
    """
    检查 FFmpeg 是否可用
    
    Returns:
        bool: FFmpeg 是否可用
    """
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_video_info(video_path):
    """
    使用 FFprobe 获取视频信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        dict: 包含视频信息的字典，如果失败返回 None
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format',
            '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        info = json.loads(result.stdout)
        
        # 提取视频流信息
        video_stream = None
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        format_info = info.get('format', {})
        duration = float(format_info.get('duration', 0))
        size = int(format_info.get('size', 0))
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        fps = eval(video_stream.get('r_frame_rate', '0/1'))
        
        return {
            'duration': duration,
            'size': size,
            'width': width,
            'height': height,
            'fps': fps,
            'codec': video_stream.get('codec_name', 'unknown')
        }
    except Exception as e:
        print(f"获取视频信息失败: {e}")
        return None


def build_speed_filters(speed):
    """
    构建FFmpeg速度调整滤镜
    
    Args:
        speed: 播放速度倍数 (例如: 0.5, 1.0, 2.0, 4.0)
        
    Returns:
        tuple: (视频滤镜字符串, 音频滤镜字符串)
    """
    # 视频速度调整：使用 setpts 滤镜
    # setpts=1/speed*PTS 表示速度倍数
    video_filter = f"setpts={1.0/speed}*PTS"
    
    # 音频速度调整：使用 atempo 滤镜
    # atempo 范围是 0.5-2.0，超过2.0需要链式使用
    audio_filters = []
    remaining_speed = speed
    
    while remaining_speed > 2.0:
        audio_filters.append("atempo=2.0")
        remaining_speed /= 2.0
    
    while remaining_speed < 0.5:
        audio_filters.append("atempo=0.5")
        remaining_speed *= 2.0
    
    if remaining_speed != 1.0:
        audio_filters.append(f"atempo={remaining_speed:.6f}")
    
    audio_filter = ",".join(audio_filters) if audio_filters else None
    
    return video_filter, audio_filter


def create_app(video_path, quality='medium', target_fps=None, original_fps=None):
    """
    创建Flask应用，提供视频流服务
    
    Args:
        video_path: 视频文件路径
        quality: 转码质量 ('low', 'medium', 'high')
        target_fps: 目标FPS (如果为None，则使用原始FPS)
        original_fps: 原始视频FPS (用于计算速度倍数)
        
    Returns:
        Flask应用实例
    """
    app = Flask(__name__)
    video_path = os.path.abspath(video_path)
    video_filename = os.path.basename(video_path)
    
    # 转码质量设置
    quality_settings = {
        'low': {
            'video_bitrate': '500k',
            'resolution': '640x360',
            'preset': 'ultrafast'
        },
        'medium': {
            'video_bitrate': '1500k',
            'resolution': '1280x720',
            'preset': 'fast'
        },
        'high': {
            'video_bitrate': '3000k',
            'resolution': '1920x1080',
            'preset': 'medium'
        }
    }
    
    settings = quality_settings.get(quality, quality_settings['medium'])
    
    # 计算速度倍数
    if target_fps is not None and original_fps is not None and original_fps > 0:
        speed = target_fps / original_fps
    else:
        speed = 1.0
    
    # 构建速度调整滤镜
    video_speed_filter, audio_speed_filter = build_speed_filters(speed)
    
    @app.route('/')
    def index():
        """主页面，显示视频播放器"""
        video_info = get_video_info(video_path)
        duration_str = f"{video_info['duration']:.1f}秒" if video_info else "未知"
        size_str = f"{video_info['size'] / (1024*1024):.1f}MB" if video_info else "未知"
        
        # 显示FPS信息
        if target_fps is not None and original_fps is not None:
            fps_str = f"{original_fps:.2f} → {target_fps:.2f} fps"
            speed_str = f"{speed:.2f}x"
        else:
            fps_str = f"{original_fps:.2f} fps" if original_fps else "未知"
            speed_str = "1.0x"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>视频流播放器 - {video_filename}</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #fff;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    min-height: 100vh;
                    padding: 20px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                }}
                .container {{
                    background: rgba(26, 26, 26, 0.95);
                    border-radius: 20px;
                    padding: 30px;
                    max-width: 1200px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
                }}
                h1 {{
                    text-align: center;
                    margin-bottom: 30px;
                    font-size: 2em;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                .video-wrapper {{
                    position: relative;
                    width: 100%;
                    background: #000;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                }}
                video {{
                    width: 100%;
                    height: auto;
                    display: block;
                }}
                .info-panel {{
                    margin-top: 20px;
                    padding: 20px;
                    background: rgba(42, 42, 42, 0.8);
                    border-radius: 10px;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                }}
                .info-item {{
                    text-align: center;
                }}
                .info-label {{
                    font-size: 0.9em;
                    color: #aaa;
                    margin-bottom: 5px;
                }}
                .info-value {{
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #fff;
                }}
                .quality-selector {{
                    margin-top: 20px;
                    text-align: center;
                }}
                .quality-selector select {{
                    padding: 10px 20px;
                    background: #667eea;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 1em;
                    cursor: pointer;
                }}
                .quality-selector select:hover {{
                    background: #5568d3;
                }}
                .status {{
                    margin-top: 15px;
                    padding: 10px;
                    background: rgba(76, 175, 80, 0.2);
                    border-left: 4px solid #4CAF50;
                    border-radius: 5px;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎬 视频流播放器</h1>
                <div class="video-wrapper">
                    <video id="videoPlayer" controls autoplay>
                        <source src="/stream" type="video/mp4">
                        您的浏览器不支持视频播放。
                    </video>
                </div>
                <div class="info-panel">
                    <div class="info-item">
                        <div class="info-label">文件名</div>
                        <div class="info-value">{video_filename}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">时长</div>
                        <div class="info-value">{duration_str}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">文件大小</div>
                        <div class="info-value">{size_str}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">质量</div>
                        <div class="info-value">{quality}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">帧率 (FPS)</div>
                        <div class="info-value">{fps_str}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">播放速度</div>
                        <div class="info-value">{speed_str}</div>
                    </div>
                </div>
                <div class="status">
                    ✅ 实时转码流式传输 | 支持大视频文件播放 | 帧率: {fps_str} | 速度: {speed_str}
                </div>
            </div>
            <script>
                // 视频加载错误处理
                document.getElementById('videoPlayer').addEventListener('error', function(e) {{
                    console.error('视频加载错误:', e);
                }});
                
                // 视频加载成功
                document.getElementById('videoPlayer').addEventListener('loadeddata', function() {{
                    console.log('视频加载成功');
                }});
            </script>
        </body>
        </html>
        """
        return html
    
    @app.route('/stream')
    def stream_video():
        """
        视频流端点，使用 FFmpeg 实时转码
        支持 HTTP Range 请求，实现视频拖拽播放
        """
        # 获取请求的 Range 头
        range_header = request.headers.get('Range', None)
        
        # 构建视频滤镜（包含缩放和速度调整）
        scale_filter = f"scale={settings['resolution']}:force_original_aspect_ratio=decrease"
        if speed != 1.0:
            # 如果速度不是1.0，组合缩放和速度滤镜
            video_filter = f"{scale_filter},{video_speed_filter}"
        else:
            video_filter = scale_filter
        
        # 构建 FFmpeg 命令
        # 使用 libx264 编码，适合流式传输
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-c:v', 'libx264',
            '-preset', settings['preset'],
            '-b:v', settings['video_bitrate'],
            '-maxrate', settings['video_bitrate'],
            '-bufsize', str(int(settings['video_bitrate'].replace('k', '')) * 2) + 'k',
            '-vf', video_filter,
            '-c:a', 'aac',
            '-b:a', '128k',
            '-f', 'mp4',
            '-movflags', 'frag_keyframe+empty_moov+default_base_moof',  # 流式输出
            '-threads', '0',  # 使用所有可用线程
            '-'  # 输出到 stdout
        ]
        
        # 如果指定了目标FPS，添加输出帧率参数
        if target_fps is not None:
            # 在 -vf 之后插入 -r 参数
            vf_index = ffmpeg_cmd.index('-vf')
            ffmpeg_cmd.insert(vf_index + 2, '-r')
            ffmpeg_cmd.insert(vf_index + 3, str(target_fps))
        
        # 如果速度不是1.0，添加音频速度调整滤镜
        if speed != 1.0 and audio_speed_filter:
            # 在 -c:a 之前插入 -af 参数
            aac_index = ffmpeg_cmd.index('-c:a')
            ffmpeg_cmd.insert(aac_index, '-af')
            ffmpeg_cmd.insert(aac_index + 1, audio_speed_filter)
        
        # 如果请求包含 Range，需要从指定位置开始
        if range_header:
            # 解析 Range 头 (格式: bytes=start-end)
            import re
            match = re.search(r'bytes=(\d+)-(\d*)', range_header)
            if match:
                start_byte = int(match.group(1))
                # 对于视频流，我们从头开始，但可以优化
                # 这里简化处理，从头开始转码
                pass
        
        def generate():
            """生成器函数，执行 FFmpeg 并流式输出"""
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10**8  # 大缓冲区
                )
                
                # 在后台线程中读取 stderr（避免阻塞）
                def read_stderr():
                    for line in iter(process.stderr.readline, b''):
                        pass  # 可以在这里处理 FFmpeg 输出
                
                stderr_thread = threading.Thread(target=read_stderr, daemon=True)
                stderr_thread.start()
                
                # 流式输出视频数据
                try:
                    while True:
                        chunk = process.stdout.read(1024 * 64)  # 64KB 块
                        if not chunk:
                            break
                        yield chunk
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            except Exception as e:
                print(f"FFmpeg 转码错误: {e}")
                return
        
        # 创建响应
        response = Response(
            generate(),
            mimetype='video/mp4',
            direct_passthrough=True
        )
        
        # 设置响应头
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
        response.headers.add('Pragma', 'no-cache')
        response.headers.add('Expires', '0')
        response.headers.add('Content-Type', 'video/mp4')
        
        return response
    
    return app


def main():
    """
    主函数
    解析命令行参数，启动视频流服务器
    """
    parser = argparse.ArgumentParser(
        description='启动大视频流式服务器，使用 FFmpeg 实时转码',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python video_stream_server.py /path/to/large_video.mp4 8080
  python video_stream_server.py data/test/test.mp4 5000 --quality high
  python video_stream_server.py video.mp4 8080 --quality low --host 0.0.0.0
  python video_stream_server.py video.mp4 8080 --fps 60
  python video_stream_server.py video.mp4 8080 --fps 15 --quality high

质量选项:
  low    - 低质量 (500k, 640x360) - 适合慢速网络
  medium - 中等质量 (1500k, 1280x720) - 默认，平衡质量和速度
  high   - 高质量 (3000k, 1920x1080) - 适合快速网络

帧率设置:
  --fps 15    - 设置为15fps (慢速播放，适合慢速网络)
  --fps 30    - 设置为30fps (标准帧率)
  --fps 60    - 设置为60fps (流畅播放)
  如果不指定 --fps，将使用原始视频的帧率
        """
    )
    parser.add_argument('video', type=str, help='视频文件路径')
    parser.add_argument('port', type=int, help='服务器端口号')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务器主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--quality', type=str, default='medium',
                       choices=['low', 'medium', 'high'],
                       help='转码质量 (默认: medium)')
    parser.add_argument('--fps', type=float, default=None,
                       help='目标帧率FPS (如果不指定，将使用原始视频的帧率)')
    
    args = parser.parse_args()
    
    # 验证FPS参数
    if args.fps is not None:
        if args.fps <= 0:
            print(f"错误: FPS必须大于0，当前值: {args.fps}")
            sys.exit(1)
        if args.fps > 120:
            print(f"警告: FPS值 {args.fps} 可能过高，建议范围: 1-60")
    
    # 检查 FFmpeg
    if not check_ffmpeg():
        print("错误: 未找到 FFmpeg")
        print("请安装 FFmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  CentOS/RHEL: sudo yum install ffmpeg")
        sys.exit(1)
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在: {args.video}")
        sys.exit(1)
    
    # 获取视频信息
    video_info = get_video_info(args.video)
    if not video_info:
        print("错误: 无法获取视频信息，请检查视频文件是否有效")
        sys.exit(1)
    
    print(f"\n视频信息:")
    print(f"  文件: {os.path.abspath(args.video)}")
    print(f"  大小: {video_info['size'] / (1024*1024):.2f} MB")
    print(f"  分辨率: {video_info['width']}x{video_info['height']}")
    print(f"  时长: {video_info['duration']:.2f} 秒")
    original_fps = video_info['fps']
    print(f"  原始帧率: {original_fps:.2f} fps")
    print(f"  编码: {video_info['codec']}")
    print(f"  转码质量: {args.quality}")
    
    # 计算目标FPS和速度倍数
    if args.fps is not None:
        target_fps = args.fps
        speed = target_fps / original_fps if original_fps > 0 else 1.0
        print(f"  目标帧率: {target_fps:.2f} fps")
        print(f"  播放速度: {speed:.2f}x")
        
        # 验证速度倍数是否在合理范围内
        if speed < 0.25 or speed > 4.0:
            print(f"警告: 计算出的播放速度 {speed:.2f}x 超出推荐范围 (0.25-4.0)")
            print(f"      原始FPS: {original_fps:.2f}, 目标FPS: {target_fps:.2f}")
    else:
        target_fps = None
        speed = 1.0
        print(f"  使用原始帧率: {original_fps:.2f} fps")
    
    # 创建Flask应用
    app = create_app(args.video, args.quality, target_fps, original_fps)
    
    # 获取本机IP
    local_ip = get_local_ip()
    
    # 输出访问URL
    print("\n" + "="*70)
    print("🎬 视频流服务器已启动!")
    print("="*70)
    print(f"\n📺 访问地址:")
    print(f"  本地访问: http://127.0.0.1:{args.port}/")
    print(f"  局域网访问: http://{local_ip}:{args.port}/")
    print(f"\n💡 提示:")
    print(f"  - 如果服务器在远程，可以使用 SSH 端口转发:")
    print(f"    ssh -L {args.port}:localhost:{args.port} user@server_ip")
    print(f"  - 然后在本地浏览器访问: http://127.0.0.1:{args.port}/")
    print("="*70)
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

