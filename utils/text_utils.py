import re

def clean_text(text: str) -> str:
    """清洗文本，去除多余空白、特殊符号"""
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊符号（保留中文、英文、数字、常用标点）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。、；：？！《》【】『』（）\-\s]', '', text)
    return text.strip()
