import os
import pickle
import jieba
from rank_bm25 import BM25Okapi

from core.chunker import SemanticChunker
from core.llm_client import ollama_client
from core.vector_store import vector_store
from utils.logger import logger
from utils.file_utils import get_all_txt_files, get_file_md5
from utils.text_utils import clean_text
from config import RAW_TXT_PATH, BATCH_INSERT_SIZE, BM25_CACHE_PATH, FILE_MD5_CACHE_PATH, TOP_N, SIMILAR_THRESHOLD

chunker = SemanticChunker()

def load_md5_cache():
    if os.path.exists(FILE_MD5_CACHE_PATH):
        with open(FILE_MD5_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_md5_cache(cache):
    with open(FILE_MD5_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

def build_kb(incremental: bool = True):
    """构建知识库，支持增量更新"""
    # 1. 获取所有txt文件
    txt_files = get_all_txt_files(RAW_TXT_PATH)
    if not txt_files:
        return "没有找到txt文件"

    # 2. 计算文件MD5，判断是否需要更新
    old_md5_cache = load_md5_cache()
    new_md5_cache = {}
    files_to_process = []

    for file_path in txt_files:
        file_md5 = get_file_md5(file_path)
        new_md5_cache[file_path] = file_md5
        if not incremental or old_md5_cache.get(file_path) != file_md5:
            files_to_process.append(file_path)

    if not files_to_process and incremental:
        logger.info("所有文件无变化，无需更新知识库")
        return "✅知识库无更新（所有文件无变化）"

    # 3. 读取并清洗文件，生成切片

    slice_list = []
    for file_path in files_to_process:
        src_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        cleaned_text = clean_text(raw_text)
        # 传入两个位置参数：文本、source_name（chunker强制需要）
        file_slices = chunker.split_text(cleaned_text, src_name)
        slice_list.extend(file_slices)


    # 4. 分批入库
    total = len(slice_list)
    logger.info(f"开始处理 {len(files_to_process)} 个文件，生成 {total} 个切片")

    if not incremental:
        vector_store.clear()

    for batch_start in range(0, total, BATCH_INSERT_SIZE):
        batch = slice_list[batch_start:batch_start + BATCH_INSERT_SIZE]
        embeds, docs, metas, ids = [], [], [], []

        for idx, item in enumerate(batch):
            try:
                embed = ollama_client.embed_text(item["content"])
                embeds.append(embed)
                docs.append(item["content"])
                metas.append({
                    "source": item["source"],
                    "start_pos": item["start_pos"],
                    "end_pos": item["end_pos"]
                })
                ids.append(f"id_{batch_start + idx}")
            except Exception as e:
                logger.error(f"切片 {batch_start+idx+1} 处理失败: {str(e)}")
                continue

        if embeds:
            vector_store.add_batch(embeds, docs, metas, ids)

    # 5. 更新MD5缓存和BM25缓存
    save_md5_cache(new_md5_cache)
    # 生成BM25语料缓存
    all_docs = vector_store.get_all_docs()
    token_corpus = [jieba.lcut(doc) for doc in all_docs]
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump({"docs": all_docs, "corpus": token_corpus}, f)

    return f"✅知识库构建完成，新增/更新 {len(slice_list)} 个切片"

def hybrid_search(query: str):
    """向量+BM25混合检索，带相似度过滤"""
    # 加载BM25缓存
    if not os.path.exists(BM25_CACHE_PATH):
        return "【提示】请先构建知识库"
    with open(BM25_CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    all_docs, token_corpus = cache["docs"], cache["corpus"]

    # 向量检索
    query_embed = ollama_client.embed_text(query)
    vec_res = vector_store.query(query_embed, TOP_N * 2)
    vec_chunks = vec_res["documents"][0]
    vec_distances = vec_res["distances"][0]

    # 过滤低相似度结果
    valid_chunks = []
    for chunk, dist in zip(vec_chunks, vec_distances):
        if (1 - dist) >= SIMILAR_THRESHOLD:
            valid_chunks.append(chunk)

    # BM25检索
    bm25 = BM25Okapi(token_corpus)
    token_q = jieba.lcut(query)
    bm25_scores = bm25.get_scores(token_q)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)[:TOP_N]
    bm25_chunks = [all_docs[i] for i in top_bm25_idx]

    # 合并去重
    final_chunks = list(dict.fromkeys(valid_chunks + bm25_chunks))[:TOP_N]
    return "\n---\n".join(final_chunks)
