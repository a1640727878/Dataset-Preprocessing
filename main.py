from pathlib import Path
from tkinter import S
import webview
import gradio as gr

import threading
import time

import uuid
import glob
import json
import datetime

_task_options = ["None"]
_task_run = {}
_task_run["None"] = {
    "get_gui": lambda info: None,
    "run_tools": lambda info: None,
    "get_default_info": lambda: {},
}
for ui_file in glob.glob("./module/*/run.py"):
    try:
        module_path = ui_file.replace("\\", "/").lstrip("./").replace("/", ".").removesuffix(".py")
        module = __import__(module_path, fromlist=["get_tag", "get_gui", "run_tools", "get_default_info"])
        if (get_tag := getattr(module, "get_tag", None)) is not None:
            tag = get_tag()
            _task_options.append(tag)

            if tag not in _task_run:
                _task_run[tag] = {}
            if (ui_func := getattr(module, "get_gui", None)) is not None:
                _task_run[tag]["get_gui"] = ui_func
            if (run_func := getattr(module, "run_tools", None)) is not None:
                _task_run[tag]["run_tools"] = run_func
            if (run_func := getattr(module, "get_default_info", None)) is not None:
                _task_run[tag]["get_default_info"] = run_func
    except Exception:
        print(f"加载模块 {ui_file} 时出错:", exc_info=True)
        continue

_PRESETS = None


def get_presets():
    global _PRESETS
    if not _PRESETS:
        config_dir = Path("./config/presets")
        config_dir.mkdir(parents=True, exist_ok=True)
        _PRESETS = {path.stem: path for path in Path(config_dir).glob("*.json")}
    return _PRESETS


CUSTOM_CSS = """
body {
    width: 980px;
    overflow-x: hidden;
    overflow-y: scroll;
}
#task_hand {
    height: 100px;
}
"""
tasks_info_state = {}
with gr.Blocks(css=CUSTOM_CSS) as demo:

    gr.HTML(
        """
    <div style="font-family: 'Microsoft YaHei', sans-serif;">
        <span style="font-size: 2.5em;">兔兔</span>
        <span style="font-size: 1.5em;">的妙妙的小工具</span>
    </div>
    """
    )

    tasks_type_state = gr.State({})
    presets_keys_state = gr.State(list(get_presets().keys()))

    def add_task(type: dict):
        while (task_id := f"task_{uuid.uuid4()}") in type:
            continue
        type[task_id] = _task_options[0]
        tasks_info_state[task_id] = {
            "type_name": _task_options[0],
            "info": {},
        }
        return type

    def del_task(type: dict, task_id):
        del type[task_id]
        del tasks_info_state[task_id]
        return type

    def dropdown_set_type(type: dict, task_id, type_name):
        type[task_id] = type_name
        tasks_info_state[task_id] = {"task_type": type_name, "info": _task_run[type_name]["get_default_info"]()}
        return type

    def updata_data(data: dict, task_id):
        tasks_info_state[task_id]["info"] = data

    def clear_task():
        tasks_info_state.clear()
        return {}

    def run_tools():
        info = tasks_info_state.copy()
        for id, value in info.items():
            type = value["task_type"]
            if type == "None":
                continue
            info_data = value["info"]
            if type in _task_run:
                _task_run[type]["run_tools"](info_data)

    def set_presets(preset):
        json_path = get_presets().get(preset, None)
        if not json_path:
            return {}
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data: dict = json.load(f)
                tasks_info_state.clear()
                tasks_info_state.update(json_data)
                tasks_type = {}
                for task_id, task_dict in tasks_info_state.items():
                    tasks_type[task_id] = task_dict["task_type"]
                return tasks_type
        except Exception as e:
            print(f"加载预设文件 {json_path} 时出错: {e}")
            return {}

    def sava_presets():
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = Path(f"./config/presets/preset_{current_time}.json")
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(tasks_info_state, f, ensure_ascii=False, indent=4)
            print(f"预设文件已保存至 {json_path}")
        except Exception as e:
            print(f"保存预设文件 {json_path} 时出错: {e}")

    def refresh_presets():
        config_dir = Path("./config/presets")
        config_dir.mkdir(parents=True, exist_ok=True)
        global _PRESETS
        _PRESETS = {path.stem: path for path in Path(config_dir).glob("*.json")}
        return list(_PRESETS.keys())

    @gr.render(inputs=[presets_keys_state])
    def render_presets(presets_keys):
        with gr.Group():
            with gr.Column():
                presets = gr.Dropdown(presets_keys, value=presets_keys[0], label="选择预设")
                with gr.Row():
                    set_presets_button = gr.Button("加载预设")
                    set_presets_button.click(set_presets, inputs=[presets], outputs=[tasks_type_state])
                    sava_presets_button = gr.Button("保存预设")
                    sava_presets_button.click(sava_presets)
                    refresh_presets_button = gr.Button("刷新预设")
                    refresh_presets_button.click(refresh_presets, outputs=[presets_keys_state])

    with gr.Row():
        add_task_button = gr.Button("添加任务")
        add_task_button.click(add_task, inputs=[tasks_type_state], outputs=[tasks_type_state])
        clear_task_button = gr.Button("清空任务")
        clear_task_button.click(clear_task, outputs=[tasks_type_state])
        run_tools_button = gr.Button("运行任务")
        run_tools_button.click(run_tools)

    @gr.render(inputs=[tasks_type_state])
    def render_task(state_data: dict):
        with gr.Column():
            for task_id, task_type in state_data.items():
                with gr.Group():
                    with gr.Row():
                        dropdown = gr.Dropdown(_task_options, value=task_type, scale=9, elem_id="task_hand", label="类型选择")
                        dropdown.input(dropdown_set_type, inputs=[tasks_type_state, gr.State(task_id), dropdown], outputs=[tasks_type_state])
                        button_remove_task = gr.Button("🗑️", variant="stop", scale=1, min_width=0, elem_id="task_hand")
                        button_remove_task.click(del_task, inputs=[tasks_type_state, gr.State(task_id)], outputs=[tasks_type_state])
                    if task_type != "None":
                        info_data = tasks_info_state[task_id]["info"]
                        ui_data = _task_run[task_type]["get_gui"](info_data)
                        if isinstance(ui_data, gr.State):
                            ui_data.change(updata_data, inputs=[ui_data, gr.State(task_id)])


def run_gradio():
    # 启动服务器，返回服务器信息（包含地址和端口）
    return demo.launch(
        server_name="127.0.0.1",  # 本地地址
        server_port=7860,  # 固定端口
        share=False,  # 不生成公网链接
        debug=False,  # 非调试模式
        quiet=False,  # 减少控制台输出
    )


if __name__ == "__main__":
    # 在单独线程中启动 Gradio 服务器
    gradio_thread = threading.Thread(target=run_gradio, daemon=True)
    gradio_thread.start()

    # 等待服务器启动
    time.sleep(2)

    # 用 pywebview 创建窗口加载 Gradio 应用
    webview.create_window(
        title="兔兔的妙妙小工具",
        url="http://127.0.0.1:7860",  # 对应 Gradio 服务器地址
        width=1000,
        height=700,
        resizable=True,
    )

    # 启动 pywebview 事件循环
    webview.start()
