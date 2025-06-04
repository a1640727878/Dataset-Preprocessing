import gradio as gr
from pathlib import Path

from .tools import run


def __validate_folder_path(path):
    if not path:
        return False
    return Path(path).is_dir()


def get_tag():
    return "规整图片"


def get_default_info():
    return {
        "image_size": 1024,
        "input_dir": "./img_data/image",
        "output_dir": "./img_data/organize_image",
    }


def get_gui(info):
    data = gr.State(info)

    def update_data(data, key, value):
        data[key] = value
        return data

    def update_input_data(data, key, value):
        data[key] = value
        return data, gr.update(visible=not __validate_folder_path(value))

    with gr.Column():
        image_size = gr.Slider(label="图片尺寸", minimum=512, maximum=2048, step=1, value=info["image_size"], info="设置图片的尺寸大小")

        input_dir = gr.Textbox(label="输入目录", value=info["input_dir"], info="设置图片输入的目录路径", placeholder="请输入图片输入目录路径")
        input_warning = gr.Markdown("⚠️ 输入目录不存在，请检查路径是否正确。", visible=not __validate_folder_path(info["input_dir"]))

        output_dir = gr.Textbox(label="输出目录", value=info["output_dir"], info="设置图片输出的目录路径", placeholder="请输入图片输出目录路径")
 
        image_size.change(update_data, inputs=[data, gr.State("image_size"), image_size], outputs=[data])
        input_dir.change(update_input_data, inputs=[data, gr.State("input_dir"), input_dir], outputs=[data, input_warning])
        output_dir.change(update_data, inputs=[data, gr.State("output_dir"), output_dir], outputs=[data])
    return data


def run_tools(info):
    print(info)
    run(info["image_size"], info["input_dir"], info["output_dir"])
