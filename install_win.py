import shutil
import subprocess
import sys
import os


def download_file(url: str, to_path: str):
    try:
        from pySmartDL import SmartDL
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pySmartDL"])
        from pySmartDL import SmartDL

    file_obj = SmartDL(url, to_path)
    if file_obj.isFinished():
        print("文件已存在，无需下载")
        return True
    file_obj.start()
    if file_obj.isSuccessful():
        print("下载已成功完成")
        file_obj.wait()
        return True
    else:
        print("下载失败")
        return False


def __set_python_embed_url(url_header: str, python_version: str, sys_platform: str):
    return f"{url_header}{python_version}/python-{python_version}-embed-{sys_platform}.zip"


def __get_sys_platform():
    sys_platform = "amd64" if sys.platform == "win32" else "win-amd64"
    return sys_platform


def download_python_embed_for_origin(python_version="3.10.9"):
    url = __set_python_embed_url("https://www.python.org/ftp/python/", python_version, __get_sys_platform())
    to_path = os.path.join(os.getcwd(), "python-embed.zip")
    return download_file(url, to_path)


def download_python_embed_for_hawei(python_version="3.10.9"):
    url = __set_python_embed_url("https://mirrors.huaweicloud.com/python/", python_version, __get_sys_platform())
    to_path = os.path.join(os.getcwd(), "python-embed.zip")
    return download_file(url, to_path)


def download_python_embed_for_ali(python_version="3.10.9"):
    url = f"https://mirrors.aliyun.com/python-release/windows/python-{python_version}-embed-{__get_sys_platform()}.zip"
    to_path = os.path.join(os.getcwd(), "python-embed.zip")
    return download_file(url, to_path)


def download_python_embed(python_version="3.10.9"):
    if download_python_embed_for_hawei(python_version):
        return True
    if download_python_embed_for_ali(python_version):
        return True
    if download_python_embed_for_origin(python_version):
        return True
    return False


pth_str = """
./Lib/site-packages
./Scripts
../
./
import site
"""


def get_python_embed_data_paths(venv_dir: str):
    python_zip = None
    python_pth = None
    for root, dirs, files in os.walk(venv_dir):
        for file in files:
            if file.startswith("python"):
                if file.endswith("._pth"):
                    python_pth = os.path.join(root, file)
                if file.endswith(".zip"):
                    python_zip = os.path.join(root, file)
    return python_zip, python_pth


def install_python_embed_venv():
    python_embed_path = os.path.join(os.getcwd(), "python-embed.zip")
    if not os.path.exists(python_embed_path):
        download_is_success = download_python_embed()
        if not download_is_success:
            return False
    venv_path = os.path.join(os.getcwd(), "venv")
    if os.path.isdir(venv_path):
        if os.path.isfile(os.path.join(venv_path, "python.exe")):
            print("虚拟环境已存在，无需创建")
            return True
    import zipfile

    with zipfile.ZipFile(python_embed_path, "r") as zip_ref:
        zip_ref.extractall(venv_path)

    python_zip, python_pth = get_python_embed_data_paths(venv_path)
    with zipfile.ZipFile(python_zip, "r") as zip_ref:
        zip_ref.extractall(os.path.join(venv_path, "Lib", "site-packages"))

    with open(python_pth, "w") as f:
        f.write(pth_str)

    python_exe_path = os.path.join(venv_path, "python.exe")
    subprocess.run([python_exe_path, "get-pip.py"])

    pip_exe_path = os.path.join(venv_path, "Scripts", "pip.exe")
    subprocess.run([pip_exe_path, "install", "--upgrade", "pip"])
    subprocess.run([pip_exe_path, "install", "-r", "requirements.txt"])


if __name__ == "__main__":
    try:
        install_python_embed_venv()
    except Exception as e:
        print("安装失败")
        venv_path = os.path.join(os.getcwd(), "venv")
        if os.path.isdir(venv_path):
            shutil.rmtree(venv_path)
