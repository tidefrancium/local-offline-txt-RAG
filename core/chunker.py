# core/chunker.py
import os
import chardet
from config import RAW_TXT_PATH, CHUNK_MAX_CHAR, CHUNK_STEP

# 可配置常量（统一管理阈值，不用硬编码数字）
MIN_CHUNK_LEN = 10    # 最小有效切片长度
SAMPLE_BYTES = 10000  # 编码检测仅读取前10k字节，降低内存占用

class SemanticChunker:
    def __init__(self):
        self.file_dir = RAW_TXT_PATH

    def detect_encoding(self, file_path: str) -> str:
        """优化：只读取文件头部样本字节，避免大文件全量读；兜底utf-8"""
        with open(file_path, "rb") as f:
            raw_data = f.read(SAMPLE_BYTES)
        if not raw_data:
            return "utf-8"
        res = chardet.detect(raw_data)
        enc = res.get("encoding")
        # 编码为空/识别失败，强制兜底utf-8
        if enc is None or enc.lower() in ["ascii", "None"]:
            return "utf-8"
        return enc

    def split_text(self, text: str, source_name: str):
        """固定窗口滑动切块，过滤过短片段，返回切片字典列表"""
        slice_list = []
        text_len = len(text)
        start = 0
        while start < text_len:
            end = start + CHUNK_MAX_CHAR
            seg = text[start:end].strip()
            # 过滤过短无效切片
            if len(seg) >= MIN_CHUNK_LEN:
                slice_dict = {
                    "content": seg,
                    "source": source_name,
                    "start_pos": start,
                    "end_pos": end
                }
                slice_list.append(slice_dict)
            start += CHUNK_STEP
        return slice_list

    def _scan_all_txt(self, root_dir: str):
        """递归遍历所有子文件夹，兼容 .txt / .TXT"""
        file_list = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.lower().endswith(".txt"):
                    full_path = os.path.join(dirpath, fn)
                    file_list.append(full_path)
        return file_list

    def load_all_documents(self):
        all_slice_dicts = []
        if not os.path.exists(self.file_dir):
            print(f"【警告】文档目录不存在 {self.file_dir}")
            return []
        
        # 递归获取全部txt文件
        txt_paths = self._scan_all_txt(self.file_dir)
        if not txt_paths:
            print("【提示】raw_txt目录下未找到任何txt文档")
            return []

        for file_path in txt_paths:
            filename = os.path.basename(file_path)
            try:
                enc = self.detect_encoding(file_path)
                with open(file_path, "r", encoding=enc) as f:
                    full_text = f.read().strip()
                # 跳过完全空白文件
                if not full_text:
                    print(f"【跳过】空白文件：{filename}")
                    continue
                slice_dicts = self.split_text(full_text, filename)
                all_slice_dicts.extend(slice_dicts)
            except Exception as e:
                print(f"【读取失败】{filename}，跳过该文件，错误信息：{str(e)}")
                continue
        
        print(f"文档加载完成，总切片字典数量：{len(all_slice_dicts)}")
        return all_slice_dicts
