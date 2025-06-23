import argparse
from mode_utils import Json2TxtProcessing


def parse_args():
    parser = argparse.ArgumentParser(description="数据集预处理工具")
    parser.add_argument("--input_dir", type=str, default="./in", help="输入文件路径")
    parser.add_argument("--output_dir", type=str, default="./out", help="输出文件路径")
    parser.add_argument("--processsing_py", type=str, default="./default_pro_json.py", help="pro_json 的处理文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    processing_py = args.processsing_py
    json2txt = Json2TxtProcessing(input_dir, output_dir, processing_py)
    json2txt.run()

if __name__ == "__main__":
    main()