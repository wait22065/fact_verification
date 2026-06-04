"""
FEVER事实验证系统 - 证据检索模块 (多模式分流离线版)
"""
import os
import re
import json
import glob
import pickle
import logging
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional

import nltk
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.config import (
    DUMP_DIR,
    INDEX_DIR,
    HOVER_INDEX_DIR,
    BM25_TOP_N_DOCS,
    SBERT_TOP_K_SENTENCES,
    MAX_SENTENCES_PER_DOC,
    SBERT_MODEL_NAME,
    CE_MODEL_NAME,
    USE_CROSS_ENCODER,
    CE_RERANK_TOP_N,
)

logger = logging.getLogger("FEVER_Retriever")

# ---------------------------------------------------------------------------
# NLTK 资源初始化
# ---------------------------------------------------------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# ---------------------------------------------------------------------------
# 自动下载维基百科 Dump 数据机制
# ---------------------------------------------------------------------------
def check_and_download_wiki_dump(dump_dir: str = DUMP_DIR) -> None:
    """检查本地是否存在维基数据。如果缺失，自动从官方链接下载并解压。"""
    jsonl_files = glob.glob(os.path.join(dump_dir, "wiki-*.jsonl"))
    if jsonl_files:
        return

    logger.warning(f"检测到本地维基数据目录 {dump_dir} 缺失或为空。")
    logger.warning("准备从 FEVER 官网自动下载全量维基百科 Dump (约 1.6 GB)...")

    parent_dir = os.path.dirname(os.path.abspath(dump_dir))
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    zip_path = os.path.join(parent_dir, "wiki-pages.zip")
    url = "https://s3-eu-west-1.amazonaws.com/fever.ai/wiki-pages.zip"

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if not os.path.exists(zip_path):
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="📥 下载 wiki-pages.zip") as t:
                urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
            print("\n下载完成，准备解压文件...")
        except Exception as e:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise RuntimeError(f"自动下载数据失败：{e}。请尝试手动下载：\n  {url}")

    try:
        logger.info("正在解压数据，请稍候...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(parent_dir)
        logger.info("🎉 数据集解压完成！")
    except Exception as e:
        raise RuntimeError(f"解压失败：{e}")
    finally:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except:
                pass

# ---------------------------------------------------------------------------
# Sentence-BERT 单例加载 (仅在相似度检索模式下才会被激活载入)
# ---------------------------------------------------------------------------
_ST_MODEL: Optional[SentenceTransformer] = None

def get_st_model() -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_models', SBERT_MODEL_NAME)
        if os.path.exists(local_path):
            logger.info(f"使用本地 Sentence-BERT 模型: {local_path}")
            _ST_MODEL = SentenceTransformer(local_path)
        else:
            logger.info(f"未找到本地模型，自动拉取 Sentence-BERT ({SBERT_MODEL_NAME})...")
            _ST_MODEL = SentenceTransformer(SBERT_MODEL_NAME)
    return _ST_MODEL


_CE_MODEL = None
def get_ce_model():
    """懒加载 CrossEncoder 精排模型，优先使用本地模型。"""
    global _CE_MODEL

    if _CE_MODEL is None:
        from sentence_transformers import CrossEncoder

        local_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "local_models",
            "cross-encoder",
            "ms-marco-MiniLM-L-6-v2"
        )

        if os.path.exists(local_path):
            logger.info(f"使用本地 CrossEncoder 模型: {local_path}")
            _CE_MODEL = CrossEncoder(local_path)
        else:
            logger.info(f"未找到本地 CrossEncoder，尝试从 Hugging Face 加载: {CE_MODEL_NAME}")
            _CE_MODEL = CrossEncoder(CE_MODEL_NAME)

    return _CE_MODEL


# BM25 相关缓存
# 【修改】按 index_dir 分开缓存，支持 FEVER 和 HoVer 两套索引并存
# ---------------------------------------------------------------------------
_BM25_CACHE: dict = {}

def _clean_text(text: str) -> str:
    return text.replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]").strip()

def _parse_lines_field(lines_str: str) -> dict:
    """
    【修改】解析 FEVER dump 的 lines 字段。
    dump 中一行可能是：
      sent_id \t sentence_text \t link_text \t link_target ...
    所以这里只取前两列，后面的链接信息丢弃。
    """
    result = {}

    if not isinstance(lines_str, str):
        return result

    for line in lines_str.split("\n"):
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        sent_id_str = parts[0]
        sent_text = parts[1]
        sent_text = _clean_text(sent_text)

        if not sent_text:
            continue

        try:
            result[int(sent_id_str)] = sent_text
        except ValueError:
            continue

    return result

def _tokenize(text: str) -> list:
    return text.lower().split()

# ---------------------------------------------------------------------------
# 构建过滤版 BM25 索引
# ---------------------------------------------------------------------------
def build_bm25_index_filtered(
    dump_dir:  str = DUMP_DIR,
    index_dir: str = INDEX_DIR,
    data_file: str = None,
) -> None:
    from src.config import FEVER_SPLIT
    check_and_download_wiki_dump(dump_dir)

    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f"fever_{FEVER_SPLIT}.json")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"找不到数据缓存：{data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    target_pages = set()
    for item in data_list:
        for page in item.get('evidence_pages', []):
            if page:
                target_pages.add(page)

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    doc_ids_path   = os.path.join(index_dir, "doc_ids.pkl")
    sentences_path = os.path.join(index_dir, "sentences.pkl")
    bm25_path      = os.path.join(index_dir, "bm25.pkl")

    jsonl_files = sorted(glob.glob(os.path.join(dump_dir, "wiki-*.jsonl")))
    
    doc_ids = []
    tokenized_corpus = []
    sentences_store = {}
    found_pages = set()

    logger.info("开始扫描本地维基 dump 文件构建索引...")
    for file_idx, jsonl_path in enumerate(jsonl_files):
        if found_pages == target_pages:
            break
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_id = record.get("id", "").strip()
                if doc_id not in target_pages:
                    continue

                text = record.get("text", "").strip()
                lines = record.get("lines", "").strip()

                sent_dict = _parse_lines_field(lines)
                if not sent_dict:
                    sent_dict = {0: _clean_text(text)}

                sentences_store[doc_id] = sent_dict
                tokens = _tokenize(_clean_text(text))
                
                doc_ids.append(doc_id)
                tokenized_corpus.append(tokens)
                found_pages.add(doc_id)

    bm25 = BM25Okapi(tokenized_corpus)
    with open(doc_ids_path, 'wb') as f:
        pickle.dump(doc_ids, f)
    with open(sentences_path, 'wb') as f:
        pickle.dump(sentences_store, f)
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info("过滤版索引构建完成。")


def build_bm25_index_hover(
    dump_dir: str = DUMP_DIR,
    index_dir: str = HOVER_INDEX_DIR,
    data_file: str = None,
) -> None:
    """
    【新增】HoVer 专用 BM25 索引构建函数。

    HoVer 的证据页面来自 supporting_facts，
    而不是 FEVER 的 evidence_pages。
    所以需要单独构建一套 HoVer 索引。
    """
    if data_file is None:
        data_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "cache",
            "hover_dev.json"
        )

    check_and_download_wiki_dump(dump_dir)

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"找不到 HoVer 数据文件：{data_file}\n"
            "请确认你已经将 HoVer dev 文件放到 data/cache/hover_dev.json"
        )

    print(f"从 HoVer 数据文件提取目标页面：{data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    target_pages = set()
    for item in raw_data:
        for fact in item.get("supporting_facts", []):
            if fact and len(fact) >= 1:
                page = str(fact[0]).replace(" ", "_")
                if page:
                    target_pages.add(page)

    print(f"HoVer 数据集共 {len(raw_data)} 条 claim，涉及 {len(target_pages)} 个页面。")

    if not target_pages:
        raise ValueError(
            "未从 HoVer 数据中读取到 supporting_facts 页面。\n"
            "请检查 hover_dev.json 的格式。"
        )

    Path(index_dir).mkdir(parents=True, exist_ok=True)

    doc_ids_path   = os.path.join(index_dir, "doc_ids.pkl")
    sentences_path = os.path.join(index_dir, "sentences.pkl")
    bm25_path      = os.path.join(index_dir, "bm25.pkl")

    if all(os.path.exists(p) for p in [doc_ids_path, sentences_path, bm25_path]):
        print(f"HoVer BM25 索引已存在于 {index_dir}，跳过构建。")
        return

    jsonl_files = sorted(glob.glob(os.path.join(dump_dir, "wiki-*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(
            f"在 {dump_dir} 下未找到 wiki-*.jsonl 文件。\n"
            "请确认 FEVER wiki-pages 已经解压到 data/wiki-pages/。"
        )

    doc_ids = []
    tokenized_corpus = []
    sentences_store = {}
    found_pages = set()

    print("开始扫描本地 wiki dump，构建 HoVer 过滤索引...")

    for file_idx, jsonl_path in enumerate(jsonl_files):
        if found_pages == target_pages:
            break

        if file_idx == 0 or (file_idx + 1) % 10 == 0:
            print(
                f"  扫描中：{os.path.basename(jsonl_path)} "
                f"({file_idx + 1}/{len(jsonl_files)})，"
                f"已找到 {len(found_pages)}/{len(target_pages)}"
            )

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_id = record.get("id", "").strip()
                if doc_id not in target_pages:
                    continue

                text = record.get("text", "").strip()
                lines = record.get("lines", "").strip()

                if not doc_id or not text:
                    continue

                sent_dict = _parse_lines_field(lines)
                if not sent_dict:
                    sent_dict = {0: _clean_text(text)}

                tokens = _tokenize(_clean_text(text))
                if not tokens:
                    continue

                sentences_store[doc_id] = sent_dict
                doc_ids.append(doc_id)
                tokenized_corpus.append(tokens)
                found_pages.add(doc_id)

    missing_pages = target_pages - found_pages

    print("\nHoVer 索引扫描完成：")
    print(f"  目标页面数：{len(target_pages)}")
    print(f"  成功加载：{len(doc_ids)} 篇")
    print(f"  未找到：{len(missing_pages)} 个")

    if len(doc_ids) == 0:
        raise RuntimeError(
            "HoVer 索引构建失败：未加载到任何页面。\n"
            "请检查 supporting_facts 页面名是否和 wiki dump 的 id 对齐。"
        )

    bm25 = BM25Okapi(tokenized_corpus)

    with open(doc_ids_path, "wb") as f:
        pickle.dump(doc_ids, f)

    with open(sentences_path, "wb") as f:
        pickle.dump(sentences_store, f)

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"HoVer BM25 索引构建完成，保存到：{index_dir}")

# ---------------------------------------------------------------------------
# 加载索引
# ---------------------------------------------------------------------------
def load_bm25_index(index_dir: str = INDEX_DIR):
    """
    【修改】加载 BM25 索引。
    按 index_dir 分别缓存，支持：
      - FEVER: config.INDEX_DIR
      - HoVer: config.HOVER_INDEX_DIR
    """
    global _BM25_CACHE

    if index_dir in _BM25_CACHE:
        return _BM25_CACHE[index_dir]

    doc_ids_path   = os.path.join(index_dir, "doc_ids.pkl")
    sentences_path = os.path.join(index_dir, "sentences.pkl")
    bm25_path      = os.path.join(index_dir, "bm25.pkl")

    # 【保留你的优点】如果 FEVER 默认索引不存在，自动构建
    if not all(os.path.exists(p) for p in [doc_ids_path, sentences_path, bm25_path]):
        if index_dir == INDEX_DIR:
            build_bm25_index_filtered(dump_dir=DUMP_DIR, index_dir=index_dir)
        else:
            raise FileNotFoundError(
                f"找不到索引文件：{index_dir}\n"
                "如果你在跑 HoVer，请先运行：\n"
                "from src.retriever import build_bm25_index_hover\n"
                "build_bm25_index_hover()"
            )

    with open(doc_ids_path, 'rb') as f:
        doc_ids = pickle.load(f)

    with open(sentences_path, 'rb') as f:
        sentences_store = pickle.load(f)

    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)

    _BM25_CACHE[index_dir] = (bm25, doc_ids, sentences_store)
    return bm25, doc_ids, sentences_store

# ---------------------------------------------------------------------------
# 模式 A: 超高速字典查找（用于 RAG / RAG_COT 模式，无相似度检索）
# ---------------------------------------------------------------------------
def retrieve_evidence_by_title(entity_query: str, num_sentences: int = 4) -> str:
    """
    通过实体名，在内存字典中进行 O(1) 的超高速匹配。
    完全不使用 SBERT 相似度模型。
    """
    if not entity_query:
        return "【检索失败：未提供有效实体名】"
        
    formatted_entity = entity_query.strip().replace(" ", "_")
    
    try:
        _, _, sentences_store = load_bm25_index()
    except Exception as e:
        return f"RETRIEVAL_ERROR: {str(e)}"
    
    # 查找内存字典
    sent_dict = sentences_store.get(formatted_entity)
    
    # 模糊兜底：尝试首字母大写
    if not sent_dict:
        formatted_entity_title = entity_query.strip().title().replace(" ", "_")
        sent_dict = sentences_store.get(formatted_entity_title)
        
    if not sent_dict:
        return f"【未在本地维基中找到匹配词条: {entity_query}】"
        
    # 获取前 N 句
    evidence_sents = []
    for sent_id in sorted(sent_dict.keys())[:num_sentences]:
        evidence_sents.append(sent_dict[sent_id])
        
    readable_title = entity_query.strip()
    return f"[Source: {readable_title}]\n" + " ".join(evidence_sents)

# ---------------------------------------------------------------------------
# 模式 B: 两阶段相似度检索（仅用于 EXTENDED_PIPELINE / HoVer 多跳优化模式）
# ---------------------------------------------------------------------------
def retrieve_evidence_from_dump(
    claim:       str,
    bm25_top_n:  int = BM25_TOP_N_DOCS,
    sbert_top_k: int = SBERT_TOP_K_SENTENCES,
    index_dir:   str = INDEX_DIR,
    use_cross_encoder: bool | None = None,
) -> str:
    """
    两阶段检索流程：
      1. BM25 粗筛召回相关文档。
      2. Sentence-BERT 提取语义最相似的句子（相似度检索）。
    """
    try:
        bm25, doc_ids, sentences_store = load_bm25_index(index_dir)

        query_tokens = _tokenize(claim)
        if not query_tokens:
            return "RETRIEVAL_ERROR: claim 无法分词。"

        top_doc_ids = bm25.get_top_n(query_tokens, doc_ids, n=bm25_top_n)
        if not top_doc_ids:
            return "RETRIEVAL_ERROR: BM25 未能召回任何页面。"

        candidate_sentences = []
        for doc_id in top_doc_ids:
            sent_dict = sentences_store.get(doc_id, {})
            if not sent_dict:
                continue
            doc_count = 0
            for sent_id in sorted(sent_dict.keys()):
                sent_text = sent_dict[sent_id]
                if len(sent_text) < 15:
                    continue
                candidate_sentences.append((doc_id, sent_id, sent_text))
                doc_count += 1
                if doc_count >= MAX_SENTENCES_PER_DOC:
                    break

        if not candidate_sentences:
            return "RETRIEVAL_ERROR: 召回文档中没有找到有效的句子。"

        # 语义向量比对
        sent_texts = [s[2] for s in candidate_sentences]
        model = get_st_model()

        claim_embedding  = model.encode(claim, convert_to_tensor=True)
        corpus_embedding = model.encode(sent_texts, convert_to_tensor=True)

        cos_scores = util.cos_sim(claim_embedding, corpus_embedding)[0]

        # 是否启用 CrossEncoder
        if use_cross_encoder is None:
            use_cross_encoder = USE_CROSS_ENCODER

        actual_k = min(sbert_top_k, len(candidate_sentences))

        if use_cross_encoder:
            # -------------------------------------------------------
            # CrossEncoder 精排：
            # 1. 先用 SBERT 取前 CE_RERANK_TOP_N 个候选
            # 2. 再用 CrossEncoder 对 claim-sentence pair 精排
            # 3. 返回最终 Top-k
            # -------------------------------------------------------
            from src import config

            pre_k = min(
                getattr(config, "CE_RERANK_TOP_N", 20),
                len(candidate_sentences)
            )

            pre_results = torch.topk(cos_scores, k=pre_k)
            pre_indices = pre_results.indices.tolist()

            ce_model = get_ce_model()

            pairs = [
                [claim, candidate_sentences[idx][2]]
                for idx in pre_indices
            ]

            ce_scores = ce_model.predict(pairs)

            ranked = sorted(
                zip(ce_scores, pre_indices),
                key=lambda x: x[0],
                reverse=True
            )

            top_indices = [idx for _, idx in ranked[:actual_k]]

        else:
            # 原始 SBERT 排序
            top_results = torch.topk(cos_scores, k=actual_k)
            top_indices = top_results.indices.tolist()

        # 格式化证据输出
        evidence_lines = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_id, sent_id, sent_text = candidate_sentences[idx]
            readable_title = doc_id.replace("_", " ")
            evidence_lines.append(f"[{rank}] ({readable_title}, 句#{sent_id}) {sent_text}")

        return "\n".join(evidence_lines)

    except Exception as e:
        logger.exception("retrieve_evidence_from_dump 异常")
        return f"RETRIEVAL_ERROR: {str(e)}"

def retrieve_evidence_local_hop(
    article_title: str,
    claim: str,
    top_k: int = 2,
    index_dir: str = HOVER_INDEX_DIR,
) -> str:
    """
    【新增】IRCoT 多跳检索的一跳。
    输入一个 Wikipedia 词条名，在本地 HoVer/FEVER 索引中找最相关句子。
    """

    try:
        bm25, doc_ids, sentences_store = load_bm25_index(index_dir)
        model = get_st_model()

        normalized = article_title.strip().replace(" ", "_")
        normalized_lower = normalized.lower()

        matched_id = None
        for doc_id in sentences_store:
            if doc_id.lower() == normalized_lower:
                matched_id = doc_id
                break

        # 情况1：精确命中词条
        if matched_id:
            sent_dict = sentences_store[matched_id]
            candidate_items = [
                (sid, text)
                for sid, text in sent_dict.items()
                if len(text) >= 15
            ]

            if not candidate_items:
                return f"ERROR: 文章 '{article_title}' 存在但没有有效句子"

            sent_texts = [text for _, text in candidate_items]

            claim_embedding = model.encode(claim, convert_to_tensor=True)
            corpus_embedding = model.encode(sent_texts, convert_to_tensor=True)
            scores = util.cos_sim(claim_embedding, corpus_embedding)[0]

            top_indices = torch.topk(
                scores,
                k=min(top_k, len(sent_texts))
            ).indices.tolist()

            return "\n".join(
                f"• {sent_texts[i]}"
                for i in top_indices
            )

        # 情况2：没有精确命中，用 BM25 兜底找相近页面
        query_tokens = _tokenize(article_title)
        if not query_tokens:
            return f"ERROR: 无法解析搜索词 '{article_title}'"

        top_doc_ids = bm25.get_top_n(query_tokens, doc_ids, n=3)
        if not top_doc_ids:
            return f"ERROR: 找不到与 '{article_title}' 相关的页面"

        candidate_sentences = []
        for doc_id in top_doc_ids:
            sent_dict = sentences_store.get(doc_id, {})
            doc_count = 0

            for sid in sorted(sent_dict.keys()):
                text = sent_dict[sid]
                if len(text) < 15:
                    continue

                candidate_sentences.append(text)
                doc_count += 1

                if doc_count >= MAX_SENTENCES_PER_DOC:
                    break

        if not candidate_sentences:
            return f"ERROR: 找不到与 '{article_title}' 相关的有效句子"

        claim_embedding = model.encode(claim, convert_to_tensor=True)
        corpus_embedding = model.encode(candidate_sentences, convert_to_tensor=True)
        scores = util.cos_sim(claim_embedding, corpus_embedding)[0]

        top_indices = torch.topk(
            scores,
            k=min(top_k, len(candidate_sentences))
        ).indices.tolist()

        return "\n".join(
            f"• {candidate_sentences[i]}"
            for i in top_indices
        )

    except FileNotFoundError as e:
        return f"RETRIEVAL_ERROR: 索引未找到。{e}"

    except Exception as e:
        logger.exception("retrieve_evidence_local_hop 发生异常")
        return f"RETRIEVAL_ERROR: {e}"