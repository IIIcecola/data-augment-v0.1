from gradio_client import Client, handle_file
import shutil
import os
import argparse
from tqdm import tqdm
from PIL import Image

# 初始化API客户端（根据仓库实际API地址调整）
client = Client("http://10.59.67.2:5012/")

client = Client("http://10.59.67.2:5016/")
client.predict(
  prompt=prompt,
  input_image=handle_file(image_path),
  mask={"background":handle_file(image_path),"layers":[handle_file(image_path)],"composite":handle_file(image_path)},
  mode="扩展图像",
  reso="16:9",
  x_slider=0,
  y=slider=0,
  num_inference_steps=10,
  api_name="/generate_image"
)



def find_image_files(root_dir):
    """查找目录下所有图片文件"""
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
    image_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                image_files.append(os.path.join(dirpath, filename))
    return image_files

def get_image_size(image_path):
    """获取图片原始尺寸"""
    with Image.open(image_path) as img:
        return img.size

def edit_one_image(client, image_path, prompt, prompt_id, output_path, use_original_size=True, target_width=1280, target_height=720):
    """编辑单张图片（新增prompt_id参数用于命名）"""
    try:
        # 尺寸处理：支持原尺寸或统一尺寸
        width, height = get_image_size(image_path) if use_original_size else (target_width, target_height)

        # 调用API生成增强图片
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

        # 保存结果（使用prompt_id命名）
        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        # 新命名规则：原文件名_prompt{id}扩展名
        dst_path = os.path.join(output_path, f"{name}_prompt{prompt_id}{ext}")
        shutil.move(src_path, dst_path)
        return True
    except FileNotFoundError:
        print(f"错误: 源文件不存在 - {image_path}")
    except Exception as e:
        print(f"处理失败 {image_path}: {str(e)}")
    return False

def process_images(source, output_dir, use_original_size=True, target_width=1280, target_height=720):
    """处理图片（支持批量/单张）"""
    # 核心Prompt列表：围绕“角度+姿态+光线+监控特效+遮挡”设计
    prompt_list = [
        "摄像头俯拍角度，人员呈俯卧倒地状态，头部与上半身完全贴合地面，画面为正常光线，无遮挡，带轻微监控噪点，边缘轻微模糊（还原摄像头焦距效果）",
        "摄像头侧拍角度，人员呈侧卧倒地状态（左侧身体贴地），头部枕于地面，画面为弱光环境（模拟夜间监控），被栏杆边缘轻微遮挡腿部，带轻微监控噪点，边缘轻微模糊（还原摄像头焦距效果）",
        "摄像头45°斜拍角度，人员呈仰面倒地状态，背部与头部完全贴地，画面为逆光环境（镜头朝向光源），被监控文字轻微遮挡衣角，带轻微监控噪点，边缘轻微模糊（还原摄像头焦距效果）",
        "摄像头平视角度，人员呈半坐式倒地状态（上半身贴地，腿部微曲），头部偏向一侧贴地，画面为正常光线，被地面线缆轻微遮挡脚部，带轻微监控噪点，边缘轻微模糊（还原摄像头焦距效果）",
        "摄像头高机位俯拍角度，人员呈蜷缩式倒地状态（身体弯曲，上半身贴地），头部埋于臂弯贴地，画面为弱光环境，无遮挡，带轻微监控噪点，边缘轻微模糊（还原摄像头焦距效果）",
        "摄像头低机位侧拍角度，人员呈前扑倒地状态（手部撑地但上半身贴地），头部贴近地面，画面为正常光线，被掉落的安全帽轻微遮挡背部，带轻微监控噪点，边缘轻微模糊（还原摄像头焦距效果）",
        "摄像头30°斜拍角度，人员呈右侧卧倒地状态（手臂伸展），头部完全贴地，画面为逆光环境，被栏杆轻微遮挡臀部，带轻微监控噪点，边缘轻微模糊（还原摄像头焦距效果）",
        "摄像头平视角度，人员呈平躺倒地状态（四肢自然展开，上半身贴地），头部居中贴地，画面为弱光环境，被小型纸箱轻微遮挡腰部，带轻微监控噪点，边缘轻微模糊（还原摄像头焦距效果）",
        "摄像头俯拍角度，人员呈单膝跪地后倾倒地状态（单侧膝盖着地，上半身贴地），头部偏向膝盖侧贴地，画面为正常光线，无遮挡，带轻微监控噪点，边缘轻微模糊（还原摄像头焦距效果）",
        "摄像头高机位侧拍角度，人员呈坐姿倾倒倒地状态（原坐姿向一侧倾倒，上半身完全贴地），头部贴地，画面为逆光环境，被地面抹布轻微遮挡手臂，带轻微监控噪点，边缘轻微模糊（还原摄像头焦距效果）"
    ]

    # 确定处理对象（批量目录或单张图片）
    if os.path.isdir(source):
        image_files = find_image_files(source)
        print(f"发现 {len(image_files)} 张图片，开始批量处理...")
    elif os.path.isfile(source) and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        image_files = [source]
        print(f"开始处理单张图片: {source}")
    else:
        print(f"错误: 无效的图片源 - {source}")
        return

    # 批量/单张处理逻辑（通过enumerate获取prompt_id）
    for image_path in tqdm(image_files, desc="处理进度"):
        for prompt_id, prompt in enumerate(prompt_list):  # 新增prompt_id（从0开始）
            edit_one_image(
                client,
                image_path,
                prompt,
                prompt_id,  # 传递prompt_id到编辑函数
                output_dir,
                use_original_size,
                target_width,
                target_height
            )
      # 二次处理，确保未佩戴护目镜
      generated_images = find_image_files(output_dir)
      for img_path in tqdm(generated_images, desc="修正护目镜"):
        edit_one_image(
          client,
          img_path,
          prompt="如果人物佩戴防护面具或者眼部护目镜，保持场景和工具不变，仅移除人物防护面具或者眼部护目镜，确保眼部裸露；否则不做处理",
          prompt_id=999,  # 用特殊ID标记二次修正样本
          output_path=corrected_dir
        )

def main():
    parser = argparse.ArgumentParser(description='人员倒地数据增强工具（支持批量/单张处理）')
    parser.add_argument('source', help='图片源（单张图片路径或图片目录）')
    parser.add_argument('output', help='处理结果输出目录')
    parser.add_argument('--size', choices=['original', 'uniform'], default='original',
                      help='尺寸处理方式: original(保持原尺寸) 或 uniform(统一尺寸，默认1280x720)')
    parser.add_argument('--width', type=int, default=1280, help='统一尺寸时的宽度（仅--size=uniform生效）')
    parser.add_argument('--height', type=int, default=720, help='统一尺寸时的高度（仅--size=uniform生效）')
    
    args = parser.parse_args()

    # 执行处理
    process_images(
        args.source,
        args.output,
        use_original_size=(args.size == 'original'),
        target_width=args.width,
        target_height=args.height
    )

if __name__ == "__main__":
    main()
