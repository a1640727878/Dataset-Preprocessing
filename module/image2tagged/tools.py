from pathlib import Path
from ai_tools.tagger import Image2Tagger, models_source, SmilingWolf_models
from PIL import Image

IMAGE2TAGGER = None


def get_model_source():
    return models_source


def get_model_name():
    return SmilingWolf_models


def get_image2tagger():
    global IMAGE2TAGGER
    if IMAGE2TAGGER is None:
        IMAGE2TAGGER = Image2Tagger()
    return IMAGE2TAGGER


def run(model_source: str, model_name: str, confidence: float, input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"输入路径 {input_path} 不存在")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_image_paths = [p for p in input_path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    for image_path in input_image_paths:
        input_image = Image.open(str(image_path))
        output_tagger = get_image2tagger().run_models([input_image], model_source, model_name)[0]
        out_str = ""
        for tag, category, conf in output_tagger:
            if category >= 9:
                continue
            if category == 4:
                continue
            if conf < confidence:
                continue
            out_str = f"{out_str}{tag}, "

        output_tagger_path = output_path / f"{image_path.stem}.txt"
        if out_str:
            out_str = out_str.rstrip(", ")
            with open(output_tagger_path, "w", encoding="utf-8") as f:
                f.write(out_str)
        print(f"成功将标签文件保存至: {output_tagger_path}")
    print("所有打标任务已完成")
