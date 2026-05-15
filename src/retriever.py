# ====== src/retriever.py ======
"""
FEVER事实验证系统 - 证据检索模块

包含两套检索方案：
  1. 原有方案（保留不动）：实时调用 Wikipedia API
     - retrieve_evidence()           用于 RAG / RAG_COT 模式
     - retrieve_evidence_pipeline()  用于 EXTENDED_PIPELINE 模式

  2. 新方案：从本地 FEVER Wikipedia dump 检索
     - build_bm25_index()             全量构建（540万篇，需大内存，已废弃）
     - build_bm25_index_filtered()    【推荐】只索引测试集 evidence 涉及的页面，
                                      内存友好，构建速度快
     - load_bm25_index()              加载已有索引（进程内单例）
     - retrieve_evidence_from_dump()  标准两阶段检索
         阶段1：BM25 文档检索（从过滤后索引中召回 Top-N 篇文档）
         阶段2：Sentence-BERT 句子筛选（从召回文档中选 Top-K 句）

所有可调参数统一在 src/config.py 中设置：
  DUMP_DIR              - wiki-*.jsonl 所在目录
  INDEX_DIR             - 索引持久化目录
  BM25_TOP_N_DOCS       - BM25 召回的候选文档数
  SBERT_TOP_K_SENTENCES - 最终保留的证据句子数
  MAX_SENTENCES_PER_DOC - 每篇文档最多进入候选池的句子数
  SBERT_MODEL_NAME      - Sentence-BERT 模型名称

dump 文件格式：
  每行：{"id": "Page_Title", "text": "...", "lines": "0\t句子0\n1\t句子1\n..."}
  - id     与 FEVER evidence 的 wikipedia_page 字段完全对应（下划线格式）
  - lines  的句子编号与 FEVER evidence 的 sentence_id 对应
  - -LRB- / -RRB- 等是 dump 对括号的转义，读取时统一还原
"""

import os
import re
import json
import glob
import pickle
import logging
from pathlib import Path
from typing import Optional

import wikipedia
import nltk
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

from src.config import (
    DUMP_DIR,
    INDEX_DIR,
    BM25_TOP_N_DOCS,
    SBERT_TOP_K_SENTENCES,
    MAX_SENTENCES_PER_DOC,
    SBERT_MODEL_NAME,
)

# ---------------------------------------------------------------------------
# 代理 & Wikipedia 初始化
# ---------------------------------------------------------------------------

# ⚠️ 本地代理：部署到海外服务器时注释掉这两行
os.environ['http_proxy']  = 'http://127.0.0.1:17890'
os.environ['https_proxy'] = 'http://127.0.0.1:17890'

wikipedia.set_lang("en")
wikipedia.set_user_agent("FeverFactChecker/1.0 (Student_Project)")

logger = logging.getLogger("FEVER_Retriever")

# ---------------------------------------------------------------------------
# NLTK 资源（实时 API 方案分词用）
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
# Sentence-BERT 单例
# ---------------------------------------------------------------------------
_ST_MODEL: Optional[SentenceTransformer] = None

def get_st_model() -> SentenceTransformer:
    """
    懒加载 Sentence-BERT 模型，全进程只加载一次。
    模型名称从 config.SBERT_MODEL_NAME 读取，方便切换。
    """
    global _ST_MODEL
    if _ST_MODEL is None:
        logger.info(f"正在加载 Sentence-BERT 模型 ({SBERT_MODEL_NAME})...")
        _ST_MODEL = SentenceTransformer(SBERT_MODEL_NAME)
        logger.info("Sentence-BERT 模型加载完成")
    return _ST_MODEL

# ---------------------------------------------------------------------------
# BM25 相关单例（三个对象绑定在一起，由 load_bm25_index 统一写入）
# ---------------------------------------------------------------------------
_BM25_INDEX:      Optional[BM25Okapi] = None
_BM25_DOC_IDS:    Optional[list]      = None
_SENTENCES_STORE: Optional[dict]      = None  # {doc_id: {sent_id: text}}

# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    还原 FEVER dump 对特殊字符的转义：
      -LRB- / -RRB-  →  ( / )
      -LSB- / -RSB-  →  [ / ]
      -LCB- / -RCB-  →  { / }
    """
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("-LSB-", "[").replace("-RSB-", "]")
    text = text.replace("-LCB-", "{").replace("-RCB-", "}")
    return text.strip()


def _parse_lines_field(lines_str: str) -> dict:
    """
    将 dump 的 lines 字段解析为 {句子编号(int): 句子文本(str)} 的字典。

    lines 字段格式示例：
      "0\t句子0\n1\t句子1\n2\t"

    注意：
      - split("\t", 1) 最多分割一次，防止句子本身含制表符被截断
      - 空句子（\t 后面没有内容）直接跳过
      - 句子编号从 0 开始，与 FEVER evidence 的 sentence_id 对应
    """
    result = {}
    for line in lines_str.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        sent_id_str, sent_text = parts
        sent_text = _clean_text(sent_text)
        if not sent_text:
            continue
        try:
            result[int(sent_id_str)] = sent_text
        except ValueError:
            continue
    return result


def _tokenize(text: str) -> list:
    """
    BM25 分词：转小写 + 按空白切分。

    必须与建索引时保持完全一致，否则查询词和文档词表不匹配，召回率会极差。
    FEVER dump 已经是预处理过的纯文本，空白切分已足够精确。
    """
    return text.lower().split()

# ---------------------------------------------------------------------------
# 全量索引构建（原始版本，540万篇，需大内存，保留备用）
# ---------------------------------------------------------------------------

def build_bm25_index(
    dump_dir:  str = DUMP_DIR,
    index_dir: str = INDEX_DIR,
) -> None:
    """
    扫描全部 wiki-*.jsonl 文件，构建全量 BM25 索引并持久化到磁盘。

    ⚠️  FEVER dump 共约 540 万篇文章，构建时需要约 20GB+ 内存，
        普通机器会 MemoryError。推荐改用 build_bm25_index_filtered()。

    持久化三个文件：
      doc_ids.pkl    - 文档 id 列表，顺序与 BM25 内部索引对齐
      sentences.pkl  - {doc_id: {sent_id: text}}，供第二阶段句子筛选使用
      bm25.pkl       - BM25Okapi 对象，供文档级检索使用
    """
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    doc_ids_path   = os.path.join(index_dir, "doc_ids.pkl")
    sentences_path = os.path.join(index_dir, "sentences.pkl")
    bm25_path      = os.path.join(index_dir, "bm25.pkl")

    if all(os.path.exists(p) for p in [doc_ids_path, sentences_path, bm25_path]):
        print(f"BM25 索引已存在于 {index_dir}，跳过构建。")
        print("如需重建，请手动删除该目录后再运行。")
        return

    jsonl_files = sorted(glob.glob(os.path.join(dump_dir, "wiki-*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(
            f"在 {dump_dir} 下未找到 wiki-*.jsonl 文件。\n"
            "请先将解压后的 wiki-pages/ 目录放到 data/ 下。"
        )

    print(f"找到 {len(jsonl_files)} 个 jsonl 文件，开始构建全量 BM25 索引...")
    print("预计耗时 5-15 分钟（取决于 CPU 性能），请耐心等待。")

    doc_ids          = []
    tokenized_corpus = []
    sentences_store  = {}
    total_docs = 0
    skipped    = 0

    for file_idx, jsonl_path in enumerate(jsonl_files):
        file_name = os.path.basename(jsonl_path)
        if (file_idx + 1) % 10 == 0 or file_idx == 0:
            print(f"  处理中：{file_name} ({file_idx + 1}/{len(jsonl_files)})，"
                  f"已加载 {total_docs} 篇文档...")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                doc_id = record.get("id",    "").strip()
                text   = record.get("text",  "").strip()
                lines  = record.get("lines", "").strip()

                if not doc_id or not text:
                    skipped += 1
                    continue

                sent_dict = _parse_lines_field(lines)
                if not sent_dict:
                    sent_dict = {0: _clean_text(text)}

                sentences_store[doc_id] = sent_dict

                tokens = _tokenize(_clean_text(text))
                if not tokens:
                    skipped += 1
                    continue

                doc_ids.append(doc_id)
                tokenized_corpus.append(tokens)
                total_docs += 1

    print(f"文档加载完成：{total_docs} 篇有效文档，{skipped} 条记录跳过。")
    print("正在构建 BM25Okapi 索引...")

    bm25 = BM25Okapi(tokenized_corpus)

    print("索引构建完成，正在持久化到磁盘...")
    with open(doc_ids_path, 'wb') as f:
        pickle.dump(doc_ids, f)
    print(f"  doc_ids.pkl 已保存（{len(doc_ids)} 个 id）")

    with open(sentences_path, 'wb') as f:
        pickle.dump(sentences_store, f)
    print(f"  sentences.pkl 已保存（{len(sentences_store)} 篇文档）")

    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"  bm25.pkl 已保存")

    print(f"\nBM25 索引构建完成！文件存放于：{index_dir}")


# ---------------------------------------------------------------------------
# 过滤索引构建（推荐）：只索引测试集 evidence 涉及的页面
# ---------------------------------------------------------------------------

def build_bm25_index_filtered(
    dump_dir:  str = DUMP_DIR,
    index_dir: str = INDEX_DIR,
    data_file: str = None,
) -> None:
    """
    只对测试集 evidence_pages 涉及的 Wikipedia 页面构建 BM25 索引。

    相比全量版本（build_bm25_index）的优势：
      - 内存：全量约 540 万篇需要 20GB+，过滤后通常只有数万篇，几百 MB 以内
      - 速度：构建时间从数小时缩短到数分钟
      - 召回质量：检索范围限定在 ground-truth 相关页面，BM25 噪声更低

    页面列表来源：
      从 data_file（fever_*.json）中读取每条 claim 的 evidence_pages 字段，
      该字段由 data_loader.py 的 load_fever_data() 在首次下载时一并保存。
      evidence_pages 中的页面名是下划线格式，与 dump 的 id 字段直接对应，
      不需要任何额外的格式转换。

    持久化三个文件（与全量版本格式完全相同，load_bm25_index 可直接加载）：
      doc_ids.pkl    - 过滤后的文档 id 列表
      sentences.pkl  - {doc_id: {sent_id: text}}
      bm25.pkl       - BM25Okapi 对象

    Args:
        dump_dir:  wiki-*.jsonl 所在目录（默认读 config.DUMP_DIR）
        index_dir: 索引保存目录（默认读 config.INDEX_DIR，覆盖原索引）
        data_file: fever_*.json 路径；为 None 时自动推断为
                   data/fever_{FEVER_SPLIT}.json
    """
    from src.config import FEVER_SPLIT  # 避免循环导入，局部引入

    # ------------------------------------------------------------------
    # 步骤1：从 JSON 数据文件中读取需要索引的页面名集合
    # ------------------------------------------------------------------
    if data_file is None:
        data_file = f"data/fever_{FEVER_SPLIT}.json"

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"找不到数据文件：{data_file}\n"
            "请先运行 load_fever_data() 生成本地缓存，再构建索引。"
        )

    print(f"从数据文件读取 evidence 页面列表：{data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 收集所有 claim 的 evidence_pages，合并去重
    target_pages: set = set()
    for item in data_list:
        for page in item.get('evidence_pages', []):
            if page:
                target_pages.add(page)

    print(f"数据集共 {len(data_list)} 条 claim，"
          f"涉及 {len(target_pages)} 个不重复 Wikipedia 页面。")

    if not target_pages:
        raise ValueError(
            "未从数据文件中读取到任何 evidence_pages。\n"
            "请确认 data_loader.py 已更新并重新运行 load_fever_data() 刷新缓存。\n"
            f"（删除 {data_file} 后重新运行即可重新下载并保存 evidence_pages）"
        )

    # ------------------------------------------------------------------
    # 步骤2：检查索引是否已存在，避免重复构建
    # ------------------------------------------------------------------
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    doc_ids_path   = os.path.join(index_dir, "doc_ids.pkl")
    sentences_path = os.path.join(index_dir, "sentences.pkl")
    bm25_path      = os.path.join(index_dir, "bm25.pkl")

    if all(os.path.exists(p) for p in [doc_ids_path, sentences_path, bm25_path]):
        print(f"BM25 索引已存在于 {index_dir}，跳过构建。")
        print("如需重建（例如更换了数据集），请手动删除该目录后再运行。")
        return

    # ------------------------------------------------------------------
    # 步骤3：扫描 dump 文件，只保留 target_pages 中的文章
    # ------------------------------------------------------------------
    jsonl_files = sorted(glob.glob(os.path.join(dump_dir, "wiki-*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(
            f"在 {dump_dir} 下未找到 wiki-*.jsonl 文件。\n"
            "请先将解压后的 wiki-pages/ 目录放到 data/ 下。"
        )

    print(f"找到 {len(jsonl_files)} 个 jsonl 文件，开始扫描（只加载目标页面）...")

    doc_ids          = []   # 与 tokenized_corpus 下标严格对齐
    tokenized_corpus = []   # BM25 语料：每篇文档的 token 列表
    sentences_store  = {}   # {doc_id: {sent_id: text}}

    total_docs  = 0   # 成功加入索引的文档数
    skipped     = 0   # 跳过的行数（格式错误 / 不在目标列表）
    found_pages = set()  # 已找到的页面，用于进度提示和最终统计

    for file_idx, jsonl_path in enumerate(jsonl_files):
        # 每处理10个文件打印一次进度
        if (file_idx + 1) % 10 == 0 or file_idx == 0:
            print(f"  扫描中：{os.path.basename(jsonl_path)} "
                  f"({file_idx + 1}/{len(jsonl_files)})，"
                  f"已找到 {len(found_pages)}/{len(target_pages)} 个目标页面...")

        # 如果所有目标页面都已找到，可以提前退出，避免扫描剩余文件
        if found_pages == target_pages:
            print(f"  所有目标页面已找到，提前结束扫描（节省时间）。")
            break

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                doc_id = record.get("id",    "").strip()
                text   = record.get("text",  "").strip()
                lines  = record.get("lines", "").strip()

                # 不在目标页面集合中，直接跳过（这是过滤的核心逻辑）
                if doc_id not in target_pages:
                    continue

                if not doc_id or not text:
                    skipped += 1
                    continue

                # 解析 lines 字段为句子字典，供第二阶段 SBERT 筛选使用
                sent_dict = _parse_lines_field(lines)
                if not sent_dict:
                    # lines 为空时回退到 text 字段，视为第0句
                    sent_dict = {0: _clean_text(text)}

                sentences_store[doc_id] = sent_dict

                # BM25 文档级表示：对全文 text 分词
                tokens = _tokenize(_clean_text(text))
                if not tokens:
                    skipped += 1
                    continue

                doc_ids.append(doc_id)
                tokenized_corpus.append(tokens)
                found_pages.add(doc_id)
                total_docs += 1

    # ------------------------------------------------------------------
    # 步骤4：报告未找到的页面（dump 中确实不存在的页面）
    # ------------------------------------------------------------------
    missing_pages = target_pages - found_pages
    print(f"\n扫描完成：")
    print(f"  目标页面数：{len(target_pages)}")
    print(f"  成功加载：  {total_docs} 篇文档")
    print(f"  未找到：    {len(missing_pages)} 个页面（这些页面在 dump 中不存在，属正常现象）")
    if missing_pages and len(missing_pages) <= 20:
        # 未找到页面数量较少时打印出来，方便排查
        print(f"  未找到的页面：{sorted(missing_pages)}")

    if total_docs == 0:
        raise RuntimeError(
            "未能加载任何文档！请检查：\n"
            "  1. dump_dir 路径是否正确\n"
            "  2. dump 文件是否已完整解压\n"
            "  3. evidence_pages 中的页面名格式是否与 dump id 匹配（均为下划线格式）"
        )

    # ------------------------------------------------------------------
    # 步骤5：构建 BM25 索引并持久化
    # ------------------------------------------------------------------
    print(f"\n正在构建 BM25Okapi 索引（{total_docs} 篇文档）...")
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 索引构建完成。")

    print("正在持久化到磁盘...")
    with open(doc_ids_path, 'wb') as f:
        pickle.dump(doc_ids, f)
    print(f"  doc_ids.pkl 已保存（{len(doc_ids)} 个 id）")

    with open(sentences_path, 'wb') as f:
        pickle.dump(sentences_store, f)
    print(f"  sentences.pkl 已保存（{len(sentences_store)} 篇文档）")

    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"  bm25.pkl 已保存")

    print(f"\nBM25 过滤索引构建完成！文件存放于：{index_dir}")


# ---------------------------------------------------------------------------
# 索引加载（进程内单例，只加载一次）
# ---------------------------------------------------------------------------

def load_bm25_index(index_dir: str = INDEX_DIR):
    """
    加载持久化的 BM25 索引到内存。
    全进程单例：第一次调用时从磁盘加载，后续调用直接返回缓存对象。

    返回：
      (bm25, doc_ids, sentences_store)
        bm25:            BM25Okapi 对象
        doc_ids:         文档 id 列表（顺序与 bm25 内部对齐）
        sentences_store: {doc_id: {sent_id: text}}
    """
    global _BM25_INDEX, _BM25_DOC_IDS, _SENTENCES_STORE

    if _BM25_INDEX is not None:
        return _BM25_INDEX, _BM25_DOC_IDS, _SENTENCES_STORE

    doc_ids_path   = os.path.join(index_dir, "doc_ids.pkl")
    sentences_path = os.path.join(index_dir, "sentences.pkl")
    bm25_path      = os.path.join(index_dir, "bm25.pkl")

    for path in [doc_ids_path, sentences_path, bm25_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到索引文件：{path}\n"
                "请先运行 build_bm25_index_filtered() 构建索引。\n"
                "示例：from src.retriever import build_bm25_index_filtered; "
                "build_bm25_index_filtered()"
            )

    print("正在加载 BM25 索引到内存（首次加载约需数十秒）...")

    with open(doc_ids_path, 'rb') as f:
        _BM25_DOC_IDS = pickle.load(f)
    print(f"  doc_ids 加载完成：{len(_BM25_DOC_IDS)} 篇文档")

    with open(sentences_path, 'rb') as f:
        _SENTENCES_STORE = pickle.load(f)
    print("  sentences 加载完成")

    with open(bm25_path, 'rb') as f:
        _BM25_INDEX = pickle.load(f)
    print("  BM25 对象加载完成")

    print("索引加载完毕，可以开始检索。")
    return _BM25_INDEX, _BM25_DOC_IDS, _SENTENCES_STORE


# ---------------------------------------------------------------------------
# 两阶段检索主函数
# ---------------------------------------------------------------------------

def retrieve_evidence_from_dump(
    claim:       str,
    bm25_top_n:  int = BM25_TOP_N_DOCS,
    sbert_top_k: int = SBERT_TOP_K_SENTENCES,
    index_dir:   str = INDEX_DIR,
) -> str:
    """
    从本地 FEVER Wikipedia dump 中检索与 claim 相关的证据。

    两阶段流程：
      阶段1 - BM25 文档检索：
        对 claim 分词，用 BM25 从索引中召回 bm25_top_n 篇最相关文档。

      阶段2 - Sentence-BERT 句子筛选：
        从召回文档中取所有句子（每篇最多 MAX_SENTENCES_PER_DOC 句），
        编码成向量后与 claim 向量做余弦相似度排序，
        返回 Top sbert_top_k 个句子作为最终证据。

    Args:
        claim:       待验证的陈述文本（直接传入，不需要预先提取关键词）
        bm25_top_n:  文档召回数，覆盖 config 默认值
        sbert_top_k: 最终句子数，覆盖 config 默认值
        index_dir:   索引目录，覆盖 config 默认值

    Returns:
        带来源标注的证据字符串，格式：
          [1] (Page Title, 句#0) 句子文本...
          [2] (Another Page, 句#3) 另一句文本...
        或以 "RETRIEVAL_ERROR:" 开头的错误信息。
    """
    try:
        bm25, doc_ids, sentences_store = load_bm25_index(index_dir)

        # ------------------------------------------------------------------
        # 阶段1：BM25 文档检索
        # 分词方式与建索引时完全一致（_tokenize），保证词表对齐
        # ------------------------------------------------------------------
        query_tokens = _tokenize(claim)
        if not query_tokens:
            return "RETRIEVAL_ERROR: claim 分词后为空，无法检索。"

        top_doc_ids = bm25.get_top_n(query_tokens, doc_ids, n=bm25_top_n)
        if not top_doc_ids:
            return "RETRIEVAL_ERROR: BM25 未能检索到任何文档。"

        logger.debug(f"BM25 召回文档（Top-{bm25_top_n}）：{top_doc_ids}")

        # ------------------------------------------------------------------
        # 阶段2（前半）：收集候选句子
        # 每篇文档最多取 MAX_SENTENCES_PER_DOC 句，防止长文章独占候选池
        # ------------------------------------------------------------------
        candidate_sentences = []  # [(doc_id, sent_id, sent_text), ...]

        for doc_id in top_doc_ids:
            sent_dict = sentences_store.get(doc_id, {})
            if not sent_dict:
                continue

            doc_count = 0
            for sent_id in sorted(sent_dict.keys()):
                sent_text = sent_dict[sent_id]
                if len(sent_text) < 15:  # 过滤噪声短句
                    continue
                candidate_sentences.append((doc_id, sent_id, sent_text))
                doc_count += 1
                if doc_count >= MAX_SENTENCES_PER_DOC:
                    break

        if not candidate_sentences:
            return "RETRIEVAL_ERROR: 召回文档中未找到有效句子。"

        logger.debug(f"候选句子总数：{len(candidate_sentences)}")

        # ------------------------------------------------------------------
        # 阶段2（后半）：Sentence-BERT 语义排序
        # ------------------------------------------------------------------
        sent_texts = [s[2] for s in candidate_sentences]
        model      = get_st_model()

        claim_embedding  = model.encode(claim,      convert_to_tensor=True)
        corpus_embedding = model.encode(sent_texts, convert_to_tensor=True)

        # cos_sim 返回 (1, N) 矩阵，取第0行得到每句对 claim 的相似度分数
        cos_scores = util.cos_sim(claim_embedding, corpus_embedding)[0]

        actual_k    = min(sbert_top_k, len(candidate_sentences))
        top_results = torch.topk(cos_scores, k=actual_k)
        top_indices = top_results.indices.tolist()

        # ------------------------------------------------------------------
        # 格式化证据输出，带来源标注
        # ------------------------------------------------------------------
        evidence_lines = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_id, sent_id, sent_text = candidate_sentences[idx]
            readable_title = doc_id.replace("_", " ")
            evidence_lines.append(
                f"[{rank}] ({readable_title}, 句#{sent_id}) {sent_text}"
            )

        evidence_text = "\n".join(evidence_lines)
        logger.debug(f"最终证据：\n{evidence_text}")
        return evidence_text

    except FileNotFoundError as e:
        return f"RETRIEVAL_ERROR: 索引未找到。{str(e)}"
    except Exception as e:
        logger.exception("retrieve_evidence_from_dump 发生异常")
        return f"RETRIEVAL_ERROR: {str(e)}"


# ---------------------------------------------------------------------------
# 以下为原有函数，保留不动
# ---------------------------------------------------------------------------

def retrieve_evidence_pipeline(claim, entity_query, top_k_sentences=3):
    """
    原有方案：实时调用 Wikipedia API + Sentence-BERT。
    用于 EXTENDED_PIPELINE 模式，不依赖本地 dump。
    """
    try:
        search_results = wikipedia.search(entity_query, results=3)
        if not search_results:
            return f"ERROR: 找不到关键词 '{entity_query}' 相关的页面"

        all_sentences = []
        for title in search_results:
            try:
                content = wikipedia.page(title, auto_suggest=False).content
                sentences = nltk.sent_tokenize(content)
                clean_sentences = [s.strip() for s in sentences if len(s) > 20][:30]
                all_sentences.extend(clean_sentences)
            except Exception:
                continue

        if not all_sentences:
            return "ERROR: 无法从页面提取有效内容。"

        model             = get_st_model()
        claim_embedding   = model.encode(claim,         convert_to_tensor=True)
        corpus_embeddings = model.encode(all_sentences, convert_to_tensor=True)

        cos_scores  = util.cos_sim(claim_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k_sentences, len(all_sentences)))

        evidence_chain = [all_sentences[idx] for idx in top_results[1]]
        return "\n".join([f"• {s}" for s in evidence_chain])

    except Exception as e:
        return f"RETRIEVAL_ERROR: {str(e)}"


def retrieve_evidence(entity_query, top_k=1, num_sentences=3):
    """
    原有方案：实时调用 Wikipedia API，取页面摘要。
    用于 RAG / RAG_COT 模式，不依赖本地 dump。
    """
    try:
        if re.search(r'[\u4e00-\u9fa5]', entity_query):
            wikipedia.set_lang("zh")
        else:
            wikipedia.set_lang("en")

        search_results = wikipedia.search(entity_query, results=top_k + 2)
        if not search_results:
            return "未找到相关证据。"

        all_evidence = []
        count = 0
        for title in search_results:
            if count >= top_k:
                break
            if any(x in title.lower() for x in ["list of", "列表", "album"]):
                continue
            try:
                summary = wikipedia.summary(title, sentences=num_sentences, auto_suggest=False)
                all_evidence.append(f"[来源 {count+1}: {title}]\n{summary}")
                count += 1
            except Exception:
                continue

        return "\n\n".join(all_evidence) if all_evidence else "未找到有效证据。"

    except Exception as e:
        return f"检索错误: {str(e)}"