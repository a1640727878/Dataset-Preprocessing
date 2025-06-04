import gradio as gr
from pathlib import Path


def get_tag():
    return "创建文件夹"


def get_default_info():
    return {
        "dir": "./img_data/image",
    }


def get_gui(info):
    data = gr.State(info)

    def update_data(data, key, value):
        data[key] = value
        return data

    with gr.Column():
        creat_dir = gr.Textbox(label="创建目录", value=info["dir"], info="输入要创建的目录", placeholder="请输入需要创建的目录")
        creat_dir.change(update_data, inputs=[data, gr.State("dir"), creat_dir], outputs=[data])

    return data


def run_tools(info):
    print(info)
    path = Path(info["dir"])
    path.mkdir(parents=True, exist_ok=True)
