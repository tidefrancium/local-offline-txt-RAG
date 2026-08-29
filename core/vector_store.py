import chromadb
from config import VECTOR_DB_PATH
from utils.logger import logger

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.coll = self.client.get_or_create_collection("txt_kb")

    def add_batch(self, embeds, docs, metas, ids):
        self.coll.add(
            embeddings=embeds,
            documents=docs,
            metadatas=metas,
            ids=ids
        )

    def query(self, embed, top_n: int):
        # 关键：chroma query_embeddings必须套一层列表 [embed]
        return self.coll.query(
            query_embeddings=[embed],
            n_results=top_n
        )

    def get_all_docs(self):
        return self.coll.get()["documents"]

    def clear(self):
        self.client.delete_collection("txt_kb")
        self.coll = self.client.get_or_create_collection("txt_kb")
        logger.info("向量库已清空")

vector_store = VectorStore()
