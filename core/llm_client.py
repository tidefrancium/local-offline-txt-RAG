import ollama
from config import OLLAMA_BASE_URL, EMBED_MODEL, CHAT_MODEL
from utils.logger import logger

class OllamaClient:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.retry_times = 3

    def embed_text(self, text: str):
        for i in range(self.retry_times):
            try:
                resp = self.client.embed(model=EMBED_MODEL, input=[text])
                return resp["embeddings"][0]
            except Exception as e:
                logger.warning(f"嵌入请求失败，第{i+1}次重试: {str(e)}")
        logger.error("嵌入请求重试全部失败")
        raise Exception("嵌入模型调用失败")

    def chat(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        return self.client.chat(model=CHAT_MODEL, messages=messages)["message"]["content"]

    # 新增：流式对话生成器，仅新增，不改动原有任何代码
    def chat_stream(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        stream = self.client.chat(
            model=CHAT_MODEL,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            yield chunk["message"]["content"]

# 全局单例
ollama_client = OllamaClient()
