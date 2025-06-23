import argparse
from mode_utils import ImageProcessing


def parse_args():
    parser = argparse.ArgumentParser(description="数据集预处理工具")
    parser.add_argument("--input_dir", type=str, default="./in", help="输入文件路径")
    parser.add_argument("--output_dir", type=str, default="./out", help="输出文件路径")
    parser.add_argument("--image_size", type=int, default=1024, help="pro_image 的图片大小")
    parser.add_argument("--ratio", type=bool, default=True, help="pro_image 是否裁剪比例")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    image_size = args.image_size
    ratio = args.ratio
    pro_images = ImageProcessing(input_dir, output_dir, image_size, ratio)
    pro_images.run()

if __name__ == "__main__":
    main()