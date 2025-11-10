from gradio_client import Client, handle_file
import shutil
import os
import argparse
import cv2
from tqdm import tqdm
from PIL import Image

# 初始化Qwen-Image-Edit API客户端
client = Client("http://10.59.67.2:5012/")

def find_video_files(root_dir):
    """查找目录下所有视频文件"""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    video_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(video_extensions):
                video_files.append(os.path.join(dirpath, filename))
    return video_files

def extract_first_frame(video_path, output_dir):
    """从视频中提取首帧并保存"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return None, None
        
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"错误：无法读取视频首帧 {video_path}")
            return None, None
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_path = os.path.join(output_dir, f"{base_name}_first_frame.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path, (width, height)
    except Exception as e:
        print(f"提取首帧失败 {video_path}：{str(e)}")
        return None, None

def extract_last_frame(video_path, output_dir):
    """从视频中提取尾帧并保存"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return None, None
        
        # 获取视频总帧数并定位到最后一帧
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"错误：无法读取视频尾帧 {video_path}")
            return None, None
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_path = os.path.join(output_dir, f"{base_name}_last_frame.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path, (width, height)
    except Exception as e:
        print(f"提取尾帧失败 {video_path}：{str(e)}")
        return None, None

def generate_augmented_frame(client, image_path, prompt, prompt_id, output_path, target_width=1280, target_height=720):
    """调用API生成增强帧（默认输出720p）"""
    try:
        width, height = target_width, target_height
        
        result = client.predict(
            image1=handle_file(image_path),
            image2=None,
            image3=None,
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            true_guidance_scale=1,
            num_inference_steps=4,
            rewrite_prompt=False,
            height=height,
            width=width,
            api_name="/infer"
        )
        src_path = result[0]

        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        # 命名规则：原帧名_aug_prompt{id}（确保首尾帧同prompt_id可配对）
        dst_path = os.path.join(output_path, f"{name}_aug_prompt{prompt_id}{ext}")
        shutil.move(src_path, dst_path)
        return True, (width, height)
    except FileNotFoundError:
        print(f"错误：源图像不存在 {image_path}")
    except Exception as e:
        print(f"增强失败 {image_path}：{str(e)}")
    return False, None

def process_videos(source, output_root, target_width=None, target_height=None):
    """处理视频：提取首尾帧→生成匹配的增强首尾帧对"""
    # 核心增强Prompt列表（优化：增加衣着/体型多样性 + 新增车间攀爬物类型 + 调整弱光描述）
    prompt_list = [
        # 一、基础属性：衣着+体型+年龄（新增5种衣着、2种体型）
        "工人穿着蓝色工装，体型中等，30-40岁男性，攀爬动作保持不变，其他场景元素不变",
        "工人穿着红色安全服，体型偏瘦，20-30岁女性，攀爬动作保持不变，其他场景元素不变",
        "工人穿着黄色马甲，体型偏胖，50-60岁男性，攀爬动作保持不变，其他场景元素不变",
        "工人穿着绿色反光条工装，体型健壮，35-45岁男性，攀爬动作保持不变，其他场景元素不变",
        "工人穿着深蓝色安全服，体型匀称，25-35岁女性，攀爬动作保持不变，其他场景元素不变",
        "工人穿着浅灰色工装，体型偏瘦，40-50岁男性，攀爬动作保持不变，其他场景元素不变",
        "工人穿着橙色反光工装，体型中等，20-30岁男性，攀爬动作保持不变，其他场景元素不变",
        "工人穿着藏青色安全服，体型偏胖，45-55岁女性，攀爬动作保持不变，其他场景元素不变",
        # 二、环境属性：光线+视角（调整弱光描述，去除夜间模式）
        "工人衣着不变，正常光线改为强光（镜头有轻微光晕），摄像头俯拍视角，攀爬动作保持不变",
        "工人衣着不变，正常光线改为弱光（环境偏暗，无夜间滤镜），摄像头侧拍视角，攀爬动作保持不变",
        "工人衣着不变，正常光线改为逆光（轮廓增强），摄像头45°斜拍视角，攀爬动作保持不变",
        "工人衣着不变，正常光线改为侧光（明暗对比增强），摄像头平视视角，攀爬动作保持不变",
        # 三、复合属性：衣着+光线+体型+攀爬物（综合维度增强）
        "工人穿着绿色工装，体型中等，40-50岁女性，弱光环境（偏暗无夜间滤镜），攀爬动作保持不变",
        "工人穿着橙色安全服，体型偏瘦，30-40岁男性，逆光环境，攀爬动作保持不变",
        "工人穿着蓝色反光工装，体型健壮，35-45岁男性，强光环境，攀爬动作保持不变",
        "工人穿着红色工装，体型匀称，25-35岁女性，侧光环境，攀爬动作保持不变"
    ]

    # 确定处理对象
    if os.path.isdir(source):
        video_files = find_video_files(source)
        print(f"发现 {len(video_files)} 个视频文件，开始处理...")
    elif os.path.isfile(source) and source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        video_files = [source]
        print(f"开始处理单个视频：{source}")
    else:
        print(f"错误：无效的视频源 {source}")
        return

    # 输出目录结构（区分原始帧和增强帧，便于管理）
    original_first_dir = os.path.join(output_root, "original_first_frames")
    original_last_dir = os.path.join(output_root, "original_last_frames")
    augmented_first_dir = os.path.join(output_root, "augmented_first_frames")
    augmented_last_dir = os.path.join(output_root, "augmented_last_frames")

    # 批量处理视频
    for video_path in tqdm(video_files, desc="视频处理进度"):
        # 提取同一视频的首帧和尾帧
        first_frame, first_size = extract_first_frame(video_path, original_first_dir)
        last_frame, last_size = extract_last_frame(video_path, original_last_dir)
        if not first_frame or not last_frame:
            print(f"跳过视频 {video_path}（首帧或尾帧提取失败）")
            continue
        if first_size != last_size:
            print(f"警告：视频 {video_path} 首尾帧尺寸不一致（首帧：{first_size}，尾帧：{last_size}），跳过增强")
            continue
        
        # 用相同的prompt和ID增强首尾帧（确保配对）
        for prompt_id, prompt in enumerate(prompt_list):
            # 增强首帧（使用原始尺寸或指定尺寸）
            first_success, first_aug_size = generate_augmented_frame(
                client, first_frame, prompt, prompt_id, augmented_first_dir, target_width, target_height
            )
            # 增强尾帧（复用首帧的增强尺寸，确保配对尺寸一致）
            if first_success:
                generate_augmented_frame(
                    client, last_frame, prompt, prompt_id, augmented_last_dir,
                    target_width=first_aug_size[0], target_height=first_aug_size[1]
                )

def main():
    parser = argparse.ArgumentParser(description='异常攀高视频首尾帧提取与匹配增强工具')
    parser.add_argument('--source', required=True, help='视频源（单个视频路径或视频目录）')
    parser.add_argument('--output', required=True, help='输出根目录（自动创建子目录存储原始/增强帧）')
    parser.add_argument('--width', type=int, default=1280, help='增强图片宽度（默认1280），None则使用原始尺寸')
    parser.add_argument('--height', type=int, default=720, help='增强图片高度（默认720）')
    
    args = parser.parse_args()

    process_videos(
        args.source,
        args.output,
        target_width=args.width,
        target_height=args.height
    )

if __name__ == "__main__":
    main()
