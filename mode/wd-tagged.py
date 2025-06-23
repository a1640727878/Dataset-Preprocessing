import argparse
from mode_utils import WDTagger


def parse_args():
    parser = argparse.ArgumentParser(description="数据集预处理工具")
    parser.add_argument("--input_dir", type=str, default="./in", help="输入文件路径")
    parser.add_argument("--output_dir", type=str, default="./out", help="输出文件路径")
    parser.add_argument("--confidence_threshold", type=float, default=0.3, help="置信度")
    parser.add_argument("--thread_count", type=int, default=1, help="多线程数量")
    parser.add_argument("--wd_model", type=str, default="wd-eva02-large-tagger-v3", help="wd-tagged 的模型名称")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    confidence_threshold = args.confidence_threshold
    thread_count = args.thread_count
    wd_model = args.wd_model
    wd_tagger = WDTagger(wd_model, (input_dir, output_dir), confidence_threshold, thread_count)
    wd_tagger.run()

if __name__ == "__main__":
    main()