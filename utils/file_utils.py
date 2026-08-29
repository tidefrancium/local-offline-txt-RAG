import hashlib
import os

def get_file_md5(file_path: str) -> str:
    """计算单个文件的MD5值，用于增量更新判断"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_all_txt_files(folder_path: str) -> list:
    """获取文件夹下所有txt文件路径"""
    txt_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files
