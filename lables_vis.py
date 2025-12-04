import cv2
import os
import numpy as np
import glob
import argparse

def find_image_file(image_dir, base_name):
    """
    在图像目录中查找与基础名称匹配的图像文件
    支持常见图像格式: jpg, jpeg, png, bmp, tiff
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']
    
    # 尝试直接匹配常见扩展名
    for ext in image_extensions:
        image_path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(image_path):
            return image_path
    
    # 尝试匹配大小写变体
    pattern = os.path.join(image_dir, base_name + ".*")
    matches = glob.glob(pattern)
    for match in matches:
        ext = os.path.splitext(match)[1].lower()
        if ext in [ext.lower() for ext in image_extensions]:
            return match
    
    # 尝试在文件名中查找匹配（忽略扩展名）
    for filename in os.listdir(image_dir):
        file_base = os.path.splitext(filename)[0]
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in [ext.lower() for ext in image_extensions] and file_base == base_name:
            return os.path.join(image_dir, filename)
    
    return None

def visualize_annotations(annotation_path, image_dir, output_dir, output_suffix="_annotated", class_names=None):
    """
    在原始图像上可视化标注边界框
    
    参数:
        annotation_path: 标注文件路径
        image_dir: 图像文件目录
        output_dir: 输出目录
        output_suffix: 输出图像文件名后缀
        class_names: 类别名称列表（可选）
    """
    # 获取标注文件的基础名称（不含扩展名）
    base_name = os.path.splitext(os.path.basename(annotation_path))[0]
    
    # 在图像目录中查找对应的图像文件
    image_path = find_image_file(image_dir, base_name)
    if not image_path:
        print(f"警告: 在 {image_dir} 中找不到与 {base_name} 对应的图像文件")
        return None
    
    print(f"找到图像文件: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像文件 {image_path}")
        return None
    
    # 获取图像尺寸
    img_height, img_width = image.shape[:2]
    
    # 读取并解析标注文件
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    # 解析归一化坐标 (类别, x_center, y_center, width, height)
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height))
                except ValueError:
                    continue
    
    # 如果没有找到有效标注
    if not annotations:
        print(f"警告: 在 {annotation_path} 中未找到有效标注")
    
    # 设置颜色和字体
    colors = {
        0: (0, 0, 255),    # 红色
        1: (0, 255, 0),    # 绿色
        2: (255, 0, 0),    # 蓝色
        3: (0, 255, 255),  # 黄色
        4: (255, 0, 255),  # 紫色
        5: (255, 255, 0)   # 青色
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # 绘制所有标注
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        
        # 转换为绝对坐标
        x_center_abs = int(x_center * img_width)
        y_center_abs = int(y_center * img_height)
        width_abs = int(width * img_width)
        height_abs = int(height * img_height)
        
        # 计算边界框坐标
        x1 = max(0, int(x_center_abs - width_abs / 2))
        y1 = max(0, int(y_center_abs - height_abs / 2))
        x2 = min(img_width - 1, int(x_center_abs + width_abs / 2))
        y2 = min(img_height - 1, int(y_center_abs + height_abs / 2))
        
        # 获取类别颜色（如果类别ID超出预设范围，使用随机颜色）
        color = colors.get(class_id % len(colors), tuple(np.random.randint(0, 255, 3).tolist()))
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 创建标签文本
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}({class_id})"
        else:
            label = str(class_id)
        
        # 计算文本大小和位置
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(15, y1 - 5)
        
        # 绘制文本背景
        cv2.rectangle(image, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        
        # 绘制标签文本
        cv2.putText(image, label, (x1, y1 - 5), 
                   font, font_scale, (0, 0, 0), thickness)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构造输出路径
    output_filename = base_name + output_suffix + os.path.splitext(image_path)[1]
    output_path = os.path.join(output_dir, output_filename)
    
    # 保存结果图像
    cv2.imwrite(output_path, image)
    print(f"标注可视化结果已保存至: {output_path}")
    return output_path

def process_all_annotations(annotations_dir, images_dir, output_dir, output_suffix="_annotated", class_names=None):
    """
    处理目录中的所有标注文件
    
    参数:
        annotations_dir: 标注文件目录
        images_dir: 图像文件目录
        output_dir: 输出目录
        output_suffix: 输出图像文件名后缀
        class_names: 类别名称列表（可选）
    """
    # 获取所有标注文件
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
    
    if not annotation_files:
        print(f"错误: 在 {annotations_dir} 中没有找到任何标注文件 (.txt)")
        return
    
    print(f"找到 {len(annotation_files)} 个标注文件，开始处理...")
    
    processed_count = 0
    skipped_count = 0
    
    # 处理每个标注文件
    for annotation_file in annotation_files:
        annotation_path = os.path.join(annotations_dir, annotation_file)
        result = visualize_annotations(
            annotation_path, 
            images_dir, 
            output_dir, 
            output_suffix, 
            class_names
        )
        
        if result:
            processed_count += 1
        else:
            skipped_count += 1
    
    print("\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"跳过处理: {skipped_count} 个文件")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='在图像上可视化标注边界框')
    parser.add_argument('--annotations', type=str, default="./data/dataset/labels", 
                        help='标注文件目录路径')
    parser.add_argument('--images', type=str, default="./data/dataset/images", 
                        help='图像文件目录路径')
    parser.add_argument('--output', type=str, default="label_output", 
                        help='输出目录路径')
    parser.add_argument('--suffix', type=str, default="_annotated", 
                        help='输出文件后缀 (默认: "_annotated")')
    parser.add_argument('--class_names', type=str, nargs='+', default="0 1",
                        help='类别名称列表 (空格分隔)')
    
    args = parser.parse_args()
    
    # 可选：定义类别名称（根据实际类别修改）
    # 如果通过命令行提供了类别名称，则使用它们
    class_names = args.class_names
    
    # 处理所有标注
    process_all_annotations(
        annotations_dir=args.annotations,
        images_dir=args.images,
        output_dir=args.output,
        output_suffix=args.suffix,
        class_names=class_names
    )