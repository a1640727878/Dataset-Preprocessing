import os
from pySmartDL import SmartDL

_HUGGINGFACE_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")


class huggingface_url_data:
    def __init__(self, repo_id: str, repo_file_path: str, commit: str = "main"):
        """初始化 huggingface_url_data 类"""
        global _HUGGINGFACE_ENDPOINT
        self.huggingface_url = _HUGGINGFACE_ENDPOINT
        self.repo_id = repo_id
        self.repo_file_path = repo_file_path
        self.commit = commit

    def set_huggingface_url(self, url: str) -> "huggingface_url_data":
        """设置 huggingface_url"""
        self.huggingface_url = url
        return self

    def get_huggingface_url(self) -> str:
        """获取 huggingface_url"""
        return self.huggingface_url

    def get_repo_id(self) -> str:
        """获取 repo_id"""
        return self.repo_id

    def get_repo_file_path(self) -> str:
        """获取 repo_file_path"""
        return self.repo_file_path

    def get_commit(self) -> str:
        """获取 commit"""
        return self.commit

    def get_fine_url(self) -> str:
        """获取 Hugging Face 文件的下载 URL"""
        return f"{self.huggingface_url}/{self.repo_id}/resolve/{self.commit}/{self.repo_file_path}?download=true"


def download(hf_data: huggingface_url_data, output_dir: str = "./download_data", output_path: str = None):
    if output_path is None:
        output_dir = output_dir.replace("\\", "/").rstrip("/")
        output_path = f"{output_dir}/{hf_data.get_repo_id()}/{hf_data.get_repo_file_path()}"
    output_path = output_path.replace("\\", "/")

    if os.path.exists(output_path):
        return True, output_path

    print(f"下载 {hf_data.get_fine_url()} 到 {output_path}")

    download_obj = SmartDL(
        urls=hf_data.get_fine_url(),
        dest=output_path,
    )

    if download_obj.isFinished():
        return True, output_path

    try:
        download_obj.start()
    except Exception as e:
        print(f"下载失败, 报错: {e}")
        return False, None
    return download_obj.isSuccessful(), output_path
