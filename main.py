import os

from mode import WDTagger, ImageProcessing, Txt2JsonProcessing, Json2TxtProcessing, YoloCropper

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像加速示例
os.environ.setdefault("HF_HOME", "./data/hf_cache")  # 缓存目录

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="数据集预处理工具")
    parser.add_argument("--mode", type=str, choices=["pro_image", "wd-tagged", "pro_tagger", "pro_json", "yolo_cropper"], help="运行模式")
    parser.add_argument("--input_dir", type=str, default="./in", help="输入文件路径")
    parser.add_argument("--output_dir", type=str, default="./out", help="输出文件路径")
    parser.add_argument("--confidence_threshold", type=float, default=0.3, help="置信度")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou 阈值")
    parser.add_argument("--thread_count", type=int, default=1, help="多线程数量")
    parser.add_argument("--wd_model", type=str, default="wd-eva02-large-tagger-v3", help="wd-tagged 的模型名称")
    parser.add_argument("--image_size", type=int, default=1024, help="pro_image 的图片大小")
    parser.add_argument("--ratio", type=bool, default=True, help="pro_image 是否裁剪比例")
    parser.add_argument("--processsing_py", type=str, default="./default_pro_json.py", help="pro_json 的处理文件路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mode = args.mode

    if mode == "wd-tagged":
        model_name = args.wd_model
        wd_tagger = WDTagger(model_name, (args.input_dir, args.output_dir), args.confidence_threshold, args.thread_count)
        wd_tagger.run()
    elif mode == "pro_image":
        pro_images = ImageProcessing(args.input_dir, args.output_dir, args.image_size, args.ratio)
        pro_images.run()
    elif mode == "pro_tagger":
        txt2json = Txt2JsonProcessing(args.input_dir, args.output_dir)
        txt2json.run()
    elif mode == "pro_json":
        json2txt = Json2TxtProcessing(args.input_dir, args.output_dir, args.processsing_py)
        json2txt.run()
    elif mode == "yolo_cropper":
        yolo_cropper = YoloCropper(args.input_dir, args.output_dir)
        yolo_cropper.run()

    print(f"运行模式: {args.mode}")
    print(f"输入目录: {os.path.abspath(args.input_dir)}")
    print(f"输出目录: {os.path.abspath(args.output_dir)}")
