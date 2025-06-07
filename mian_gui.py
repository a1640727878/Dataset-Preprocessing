import eel

# 初始化eel并指定前端文件夹
eel.init("gui")


# 暴露给前端调用的示例函数
@eel.expose
def say_hello(name):
    return f"你好，{name}!"


# 主程序入口
def main():
    # 加载首页
    eel.start("index.html", size=(800, 600))


if __name__ == "__main__":
    main()
