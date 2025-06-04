import gradio as gr
from pathlib import Path

from .tools import get_model_name, get_model_source, run


def __validate_folder_path(path):
    if not path:
        return False
    return Path(path).is_dir()


def get_tag():
    return "图片打标-tagger"


def get_default_info():
    return {
        "model_source": get_model_source()[0],
        "model_name": get_model_name()[0],
        "confidence": 0.35,
        "input_dir": "./img_data/image",
        "output_dir": "./img_data/image",
    }


def __is_source(model_source):
    return model_source == get_model_source()[0]


def get_gui(info):
    data = gr.State(info)

    def update_data(data, key, value):
        data[key] = value
        return data

    def updata_model_source(data, key, value):
        data[key] = value
        return data, gr.update(visible=__is_source(value))

    def update_input_data(data, key, value):
        data[key] = value
        return data, gr.update(visible=not __validate_folder_path(value))

    with gr.Column():
        with gr.Row():
            model_source = gr.Dropdown(get_model_source(), value=info["model_source"], scale=5, label="模型来源")
            with gr.Row(scale=5):
                model_name = gr.Dropdown(get_model_name(), value=info["model_name"], label="模型", visible=__is_source(model_source.value))

        confidence = gr.Slider(label="置信度", minimum=0.0, maximum=1.0, step=0.01, value=info["confidence"], info="标签置信度")

        input_dir = gr.Textbox(label="输入目录", value=info["input_dir"], info="设置图片输入的目录路径", placeholder="请输入图片输入目录路径")
        input_warning = gr.Markdown("⚠️ 输入目录不存在，请检查路径是否正确。", visible=not __validate_folder_path(info["input_dir"]))

        output_dir = gr.Textbox(label="输出目录", value=info["output_dir"], info="设置图片输出的目录路径", placeholder="请输入图片输出目录路径")

        model_source.change(updata_model_source, inputs=[data, gr.State("model_source"), model_source], outputs=[data, model_name])
        model_name.change(update_data, inputs=[data, gr.State("model_name"), model_name], outputs=[data])
        confidence.change(update_data, inputs=[data, gr.State("confidence"), confidence], outputs=[data])
        input_dir.change(update_input_data, inputs=[data, gr.State("input_dir"), input_dir], outputs=[data, input_warning])
        output_dir.change(update_data, inputs=[data, gr.State("output_dir"), output_dir], outputs=[data])

    return data


def run_tools(info):
    print(info)
    run(info["model_source"], info["model_name"], info["confidence"], info["input_dir"], info["output_dir"])
