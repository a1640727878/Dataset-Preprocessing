import os

from numpy.strings import endswith

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像加速示例
os.environ.setdefault("HF_HOME", "./data/hf_cache")  # 缓存目录

import argparse
import threading
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="数据集预处理工具")
    parser.add_argument("--mode", type=str, choices=["pro_image", "wd-tagged", "gui"], help="运行模式")
    parser.add_argument("--input_dir", type=str, default="./in", help="输入文件路径")
    parser.add_argument("--output_dir", type=str, default="./out", help="输出文件路径")
    parser.add_argument("--confidence_threshold", type=float, default=0.3, help="wd-tagged 的置信度")
    parser.add_argument("--thread_count", type=int, default=1, help="多线程数量")
    parser.add_argument("--wd_model", type=str, default="wd-eva02-large-tagger-v3", help="wd-tagged 的模型名称")
    parser.add_argument("--image_size", type=int, default=1024, help="pro_image 的图片大小")
    parser.add_argument("--ratio", type=bool, default=True, help="pro_image 是否裁剪比例")
    return parser.parse_args()


def thread_wd_tagged(wd_model: str, input_dir: str, output_dir: str, confidence_threshold: float = 0.3, thread_count: int = 1):
    from tools.wd_tagger import WDTagger

    wd_tagger_moadl = WDTagger(wd_model)
    images = wd_tagged_images(input_dir, output_dir, count=thread_count)
    start_time = time.time()
    threads = []
    for i in range(thread_count):
        thread = threading.Thread(target=wd_tagged_worker, args=(start_time, wd_tagger_moadl, i, images[i], confidence_threshold))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    end_time = time.time()
    print(f"所有线程执行完毕, 共耗时{end_time - start_time:.2f}秒")


def wd_tagged_worker(start_time: float, wd_tagger_moadl, task_id, images: list[tuple[str, str]], confidence_threshold: float = 0.3):
    print(f"线程 {task_id} 开始执行")
    wd_tagged(wd_tagger_moadl, images, confidence_threshold)
    end_time = time.time()
    print(f"线程 {task_id} 执行完毕, 耗时{end_time - start_time:.2f}秒")


def wd_tagged_images(input_dir: str, output_dir: str, count: int = 1) -> list[list[tuple[str, str]]]:
    image_list = []
    image_endswith = [".jpg", ".jpeg", ".png"]
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                endswith_str = image_path.split(".")[-1]
                out_file = image_path.replace("\\", "/").replace(input_dir, output_dir)
                out_file = out_file[: -len(endswith_str)] + "txt"
                image_list.append((image_path, out_file))
    k, m = divmod(len(image_list), count)
    return [image_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(count)]


def wd_tagged(wd_tagger_moadl, images: list[tuple[str, str]], confidence_threshold: float = 0.3):
    for image_path, out_file in images:
        data = wd_tagger_moadl.predict_confidence(image_path, confidence_threshold)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(",".join(data))
        print(f"{image_path} -> {out_file}")


def pro_image_images(input_dir: str, output_dir: str) -> list[tuple[str, str]]:
    image_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                endswith_str = image_path.split(".")[-1]
                out_file = image_path.replace("\\", "/").replace(input_dir, output_dir)
                end_int = len(endswith_str) + 1
                out_file = out_file[:-end_int] + "_[ratio_name].png"
                image_list.append((image_path, out_file))
    return image_list


def pro_image(input_dir: str, output_dir: str, max_size=1024, ratio=True):
    from tools.image_processing import Image_Processing

    image_processing = Image_Processing()
    images = pro_image_images(input_dir, output_dir)
    for image_path, out_file in images:
        image, ratio_name = image_processing.pro_image(image_path, max_size, ratio)
        if not ratio:
            ratio_name = ""
        out_file = out_file.replace("[ratio_name]", ratio_name)
        image.save(out_file)
        print(f"{image_path} -> {out_file}")


if __name__ == "__main__":
    args = parse_args()
    mode = args.mode
    if mode == "wd-tagged":
        model_name = args.wd_model
        thread_wd_tagged(model_name, args.input_dir, args.output_dir, args.confidence_threshold, args.thread_count)
    if mode == "pro_image":
        pro_image(args.input_dir, args.output_dir, args.image_size, args.ratio)

    print(f"运行模式: {args.mode}")
    print(f"输入目录: {os.path.abspath(args.input_dir)}")
    print(f"输出目录: {os.path.abspath(args.output_dir)}")
