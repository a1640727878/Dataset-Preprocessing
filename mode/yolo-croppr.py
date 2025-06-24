import argparse
from mode_utils import YoloCroppr


def parse_args():
    parser = argparse.ArgumentParser(description="数据集预处理工具")
    parser.add_argument("--input_dir", type=str, default="./in", help="输入文件路径")
    parser.add_argument("--output_dir", type=str, default="./out", help="输出文件路径")
    parser.add_argument("--confidence_threshold", type=float, default=0.35, help="置信度阈值")
    parser.add_argument("--iou_threshold", type=float, default=0.7, help="iou阈值")
    parser.add_argument("--cropers_names", type=str, default="person,halfbody,head,face", help="裁剪器名称")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    confidence_threshold = args.confidence_threshold
    iou_threshold = args.iou_threshold
    cropers_names = args.cropers_names.split(",")

    yolo_cropper = YoloCroppr(input_dir, output_dir, cropers_names, confidence_threshold, iou_threshold)
    yolo_cropper.run()


if __name__ == "__main__":
    main()
