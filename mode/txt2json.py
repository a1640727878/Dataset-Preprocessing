import argparse
from mode_utils import Txt2JsonProcessing


def parse_args():
    parser = argparse.ArgumentParser(description="数据集预处理工具")
    parser.add_argument("--input_dir", type=str, default="./in", help="输入文件路径")
    parser.add_argument("--output_dir", type=str, default="./out", help="输出文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    txt2json = Txt2JsonProcessing(input_dir, output_dir)
    txt2json.run()

if __name__ == "__main__":
    main()