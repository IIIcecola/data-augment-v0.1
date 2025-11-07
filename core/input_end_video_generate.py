from gradio_client import Client, handle_file
import os
import argparse
import re
from tqdm import tqdm
import traceback

# -------------------------- 核心配置 --------------------------
API_URL = "your_actual_pusa_ti2v_api_url"  # 替换为实际API地址
# 视频生成Prompt列表（与增强首尾帧时的属性保持一致）
VIDEO_PROMPT_LIST = [
    "摄像机视角稳定，延续首帧背景，工人攀爬动作连贯自然，保持与首尾帧一致的衣着、性别、体型、场景和光线，符合工业摄像机拍摄质感"
]
# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
# 文件名解析正则（匹配增强帧命名规则：{原始视频名}_first/last_frame_aug_prompt{id}.ext）
FRAME_PATTERN = re.compile(r"^(.+)_(first|last)_frame_aug_prompt(\d+)\.(.+)$")

# -------------------------- 工具函数 --------------------------
def parse_frame_filename(filename):
    """解析增强帧文件名，提取原始视频名、帧类型（首/尾）、prompt_id"""
    match = FRAME_PATTERN.match(filename)
    if not match:
        return None  # 不符合命名规则的文件跳过
    base_name = match.group(1)  # 原始视频名
    frame_type = match.group(2)  # "first" 或 "last"
    prompt_id = int(match.group(3))  # 增强参数ID
    ext = match.group(4)  # 文件扩展名
    return {
        "base_name": base_name,
        "frame_type": frame_type,
        "prompt_id": prompt_id,
        "ext": ext
    }

def find_matched_frames(aug_first_dir, aug_last_dir):
    """查找所有配对的增强首帧和尾帧"""
    # 先收集所有首帧信息（按base_name+prompt_id分组）
    first_frames = {}
    for filename in os.listdir(aug_first_dir):
        parsed = parse_frame_filename(filename)
        if not parsed or parsed["frame_type"] != "first":
            continue
        key = (parsed["base_name"], parsed["prompt_id"])
        first_frames[key] = os.path.join(aug_first_dir, filename)
    
    # 匹配尾帧
    matched_pairs = []
    for filename in os.listdir(aug_last_dir):
        parsed = parse_frame_filename(filename)
        if not parsed or parsed["frame_type"] != "last":
            continue
        key = (parsed["base_name"], parsed["prompt_id"])
        if key in first_frames:
            # 找到配对，添加到结果列表
            matched_pairs.append({
                "first_frame": first_frames[key],
                "last_frame": os.path.join(aug_last_dir, filename),
                "base_name": parsed["base_name"],
                "prompt_id": parsed["prompt_id"]
            })
            # 移除已匹配的首帧（避免重复匹配）
            del first_frames[key]
    
    # 打印未匹配的首帧（可选）
    for key in first_frames:
        print(f"警告：未找到匹配的尾帧 - 视频名: {key[0]}, prompt_id: {key[1]}")
    
    return matched_pairs

def generate_video(client, first_frame, last_frame, video_prompt, output_dir, base_name, prompt_id):
    """调用API生成视频（使用配对的首尾帧）"""
    try:
        result = client.predict(
            prompt=video_prompt,
            negative_prompt='画面断层, 人物突变, 场景不一致, 动作跳跃',
            seed=1,
            steps=4,
            input_image=handle_file(first_frame),
            end_image=handle_file(last_frame),  # 新增尾帧参数
            mode_selector="图生视频",
            fps_slider=24,
            input_video=None,
            prompt_refiner=True,
            lora_selector=[],
            height=720,
            width=1280,
            frame_num=81,
            api_name="/generate_video"
        )

        # 解析API返回的视频路径
        video_temp_path = result.get("video")
        if not video_temp_path or not os.path.exists(video_temp_path):
            print(f"警告：API未返回有效视频路径 - {base_name}_prompt{prompt_id}")
            return False

        # 构建输出视频路径（包含原始视频名和prompt_id，明确配对关系）
        output_video_name = f"{base_name}_aug{prompt_id}.mp4"
        output_video_path = os.path.join(output_dir, output_video_name)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 保存视频
        with open(video_temp_path, "rb") as f_in, open(output_video_path, "wb") as f_out:
            f_out.write(f_in.read())

        print(f"成功生成视频：{output_video_path}")
        return True

    except Exception as e:
        print(f"\n错误：生成视频失败 - {base_name}_prompt{prompt_id}")
        print(f"错误详情：{str(e)}")
        traceback.print_exc()
        return False

# -------------------------- 主流程 --------------------------
def main():
    parser = argparse.ArgumentParser(description='基于配对首尾帧生成视频工具（确保人物/场景一致性）')
    parser.add_argument('--aug-first-dir', required=True, 
                      help='增强首帧目录（如：augmented_frames/augmented_first_frames）')
    parser.add_argument('--aug-last-dir', required=True, 
                      help='增强尾帧目录（如：augmented_frames/augmented_last_frames）')
    parser.add_argument('--output', required=True, help='输出视频目录')
    parser.add_argument('--api-url', default=API_URL, help=f'视频生成API地址（默认：{API_URL}）')

    args = parser.parse_args()

    # 验证输入目录
    for dir_path in [args.aug_first_dir, args.aug_last_dir]:
        if not os.path.isdir(dir_path):
            print(f"错误：目录不存在 - {dir_path}")
            return

    # 初始化API客户端
    print(f"连接API：{args.api_url}")
    try:
        client = Client(args.api_url, timeout=300)  # 5分钟超时
    except Exception as e:
        print(f"错误：无法连接API {args.api_url}")
        traceback.print_exc()
        return

    # 查找所有配对的首尾帧
    print("正在匹配增强首尾帧对...")
    matched_pairs = find_matched_frames(args.aug_first_dir, args.aug_last_dir)
    if not matched_pairs:
        print("错误：未找到任何配对的首尾帧")
        return
    print(f"共发现 {len(matched_pairs)} 对匹配的首尾帧")

    # 批量生成视频
    print("\n开始批量生成视频...")

    for pair in tqdm(matched_pairs, desc="视频生成进度"):
      for video_prompt in VIDEO_PROMPT_LIST:
        try:
          generate_video(
            client,
            first_frame=pair["first_frame"],
            last_frame=pair["last_frame"],
            video_prompt=video_prompt,
            output_dir=args.output,
            base_name=pair["base_name"],
            prompt_id=pair["prompt_id"]
          )
        except Exception as e:
          print(f"error: {str(e)}")
          traceback.print_exc()


    # 输出统计结果
    print("\n" + "="*50)
    print(f"批量处理完成！")
    print(f"总配对数：{len(matched_pairs)} 对")
    print(f"输出目录：{os.path.abspath(args.output)}")
    print("="*50)

if __name__ == "__main__":
    main()
