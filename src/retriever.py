# ====== src/retriever.py ======
"""
FEVER事实验证系统 - 证据检索模块

包含两套检索方案：
  1. 原有方案（保留不动）：实时调用 Wikipedia API
     - retrieve_evidence()           用于 RAG / RAG_COT 模式
     - retrieve_evidence_pipeline()  用于 EXTENDED_PIPELINE 模式

  2. 新方案（本文件新增）：从本地 FEVER Wikipedia dump 检索
     - build_bm25_index()            首次运行，构建并持久化 BM25 索引
     - load_bm25_index()             加载已有索引（进程内单例）
     - retrieve_evidence_from_dump() 标准三阶段检索
         阶段1：BM25 文档检索（从500万篇中召回 Top-N 篇文档）
         阶段2：Sentence-BERT 句子筛选（从召回文档中选 Top-K 句）
         阶段3：返回证据文本供 LLM 判断

dump 文件约定：
  - 存放路径：wiki-pages/wiki-*.jsonl（109个文件）
  - 每行格式：{"id": "Page_Title", "text": "...", "lines": "0\t句子0\n1\t句子1\n..."}
  - id 字段与 FEVER evidence 里的 wiki_url 完全对应
  - lines 字段的句子编号与 FEVER evidence 里的 sentence_id 对应
  - -LRB- / -RRB- 是 dump 对括号的转义，读取时统一还原

索引持久化路径：data/bm25_index/
  - doc_ids.pkl       所有文档 id 列表（顺序与 BM25 内部索引对齐）
  - tokenized.pkl     分词后的语料（list of list of str）
  - bm25.pkl          rank_bm25 的 BM25Okapi 对象
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

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------

# ⚠️ 本地代理：部署到海外服务器时注释掉这两行
os.environ['http_proxy'] = 'http://127.0.0.1:17890'
os.environ['https_proxy'] = 'http://127.0.0.1:17890'

wikipedia.set_lang("en")
wikipedia.set_user_agent("FeverFactChecker/1.0 (Student_Project)")

# dump 和索引的存放路径（相对于项目根目录）
DUMP_DIR = "wiki-pages"
INDEX_DIR = "data/bm25_index"

# BM25 文档检索：每个 claim 召回的候选文档数
BM25_TOP_N_DOCS = 5

# Sentence-BERT 句子筛选：从候选文档中最终选出的句子数，可以试试3到5
SBERT_TOP_K_SENTENCES = 5

# 每篇文档最多读取的句子数（控制内存，超长文章截断）
MAX_SENTENCES_PER_DOC = 50

logger = logging.getLogger("FEVER_Retriever")

# ---------------------------------------------------------------------------
# NLTK 资源（分词用）
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
    """懒加载 Sentence-BERT 模型，全进程只加载一次"""
    global _ST_MODEL
    if _ST_MODEL is None:
        logger.info("正在加载 Sentence-BERT 模型 (all-MiniLM-L6-v2)...")
        _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence-BERT 模型加载完成")
    return _ST_MODEL

# ---------------------------------------------------------------------------
# BM25 索引单例
# ---------------------------------------------------------------------------
_BM25_INDEX: Optional[BM25Okapi] = None
_BM25_DOC_IDS: Optional[list] = None  # 与 BM25 内部顺序对齐的文档 id 列表

def _clean_text(text: str) -> str:
    """
    还原 FEVER dump 对特殊字符的转义：
      -LRB-  ->  (
      -RRB-  ->  )
      -LSB-  ->  [
      -RSB-  ->  ]
      -LCB-  ->  {
      -RCB-  ->  }
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
      - 空句子（\t 后面没有内容）直接跳过
      - 句子编号从 0 开始，与 FEVER evidence 的 sentence_id 对应
    """
    result = {}
    for line in lines_str.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)  # 最多分割一次，防止句子本身含 \t
        if len(parts) < 2:
            continue
        sent_id_str, sent_text = parts
        sent_text = _clean_text(sent_text)
        if not sent_text:  # 跳过空句子
            continue
        try:
            result[int(sent_id_str)] = sent_text
        except ValueError:
            continue  # 编号不是数字则跳过（不应出现，保险起见）
    return result

def _tokenize(text: str) -> list:
    """
    BM25 分词：转小写 + 按空白切分。
    FEVER dump 已经是预处理过的纯文本，不需要复杂的分词器。
    """
    return text.lower().split()

# ---------------------------------------------------------------------------
# 索引构建（首次运行，耗时较长）
# ---------------------------------------------------------------------------

def build_bm25_index(dump_dir: str = DUMP_DIR, index_dir: str = INDEX_DIR) -> None:
    """
    扫描全部 wiki-*.jsonl 文件，构建 BM25 索引并持久化到磁盘。

    索引内容：
      - doc_ids.pkl       文档 id 列表（顺序与 BM25 对齐）
      - sentences.pkl     每篇文档的句子字典 {doc_id: {sent_id: text}}
      - bm25.pkl          BM25Okapi 对象（用于文档级检索）

    BM25 的"文档"定义：每篇 Wikipedia 文章的全文（text 字段），
    用于文档级召回。句子级筛选在第二阶段由 Sentence-BERT 完成。

    参数：
      dump_dir:   wiki-*.jsonl 文件所在目录
      index_dir:  索引保存目录
    """
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    doc_ids_path    = os.path.join(index_dir, "doc_ids.pkl")
    sentences_path  = os.path.join(index_dir, "sentences.pkl")
    bm25_path       = os.path.join(index_dir, "bm25.pkl")

    # 如果三个文件都已存在，直接跳过（不重复构建）
    if all(os.path.exists(p) for p in [doc_ids_path, sentences_path, bm25_path]):
        print(f"BM25 索引已存在于 {index_dir}，跳过构建。")
        print("如需重建，请手动删除该目录再运行。")
        return

    jsonl_files = sorted(glob.glob(os.path.join(dump_dir, "wiki-*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(
            f"在 {dump_dir} 下未找到 wiki-*.jsonl 文件。\n"
            "请先将解压后的 wiki-pages/ 目录放到 data/ 下。"
        )

    print(f"找到 {len(jsonl_files)} 个 jsonl 文件，开始构建 BM25 索引...")
    print("预计耗时 5-15 分钟（取决于 CPU 性能），请耐心等待。")

    doc_ids = []           # 文档 id 列表，与 tokenized_corpus 下标对齐
    tokenized_corpus = []  # BM25 语料：每篇文档的 token 列表
    sentences_store = {}   # 句子存储：{doc_id: {sent_id: text}}

    total_docs = 0
    skipped = 0

    for file_idx, jsonl_path in enumerate(jsonl_files):
        file_name = os.path.basename(jsonl_path)
        # 每处理10个文件打印一次进度
        if (file_idx + 1) % 10 == 0 or file_idx == 0:
            print(f"  处理中：{file_name} ({file_idx + 1}/{len(jsonl_files)})，"
                  f"已加载 {total_docs} 篇文档...")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                doc_id = record.get("id", "").strip()
                text   = record.get("text", "").strip()
                lines  = record.get("lines", "").strip()

                # 跳过 id 或 text 为空的记录（dump 首行通常是空记录）
                if not doc_id or not text:
                    skipped += 1
                    continue

                # 解析 lines 字段，存储句子供第二阶段使用
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
                total_docs += 1

    print(f"文档加载完成：{total_docs} 篇有效文档，{skipped} 条记录跳过。")
    print("正在构建 BM25 索引（rank_bm25）...")

    # 构建 BM25 索引（BM25Okapi 是最常用的 BM25 变体，参数 k1=1.5, b=0.75）
    bm25 = BM25Okapi(tokenized_corpus)

    print("BM25 索引构建完成，正在持久化到磁盘...")

    with open(doc_ids_path, 'wb') as f:
        pickle.dump(doc_ids, f)
    print(f"  已保存 doc_ids.pkl（{len(doc_ids)} 个 id）")

    with open(sentences_path, 'wb') as f:
        pickle.dump(sentences_store, f)
    print(f"  已保存 sentences.pkl（{len(sentences_store)} 篇文档的句子）")

    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"  已保存 bm25.pkl")

    print(f"\nBM25 索引构建完成！文件存放于：{index_dir}")


def load_bm25_index(index_dir: str = INDEX_DIR):
    """
    加载持久化的 BM25 索引到内存（全进程单例，只加载一次）。

    返回：
      (bm25, doc_ids, sentences_store)
        bm25:            BM25Okapi 对象
        doc_ids:         文档 id 列表
        sentences_store: {doc_id: {sent_id: text}}

    如果索引文件不存在，抛出 FileNotFoundError 并给出提示。
    """
    global _BM25_INDEX, _BM25_DOC_IDS
    # 注意：sentences_store 也需要单例，单独存一个模块级变量
    global _SENTENCES_STORE

    # 如果已经加载过，直接返回缓存
    if _BM25_INDEX is not None:
        return _BM25_INDEX, _BM25_DOC_IDS, _SENTENCES_STORE

    doc_ids_path   = os.path.join(index_dir, "doc_ids.pkl")
    sentences_path = os.path.join(index_dir, "sentences.pkl")
    bm25_path      = os.path.join(index_dir, "bm25.pkl")

    for path in [doc_ids_path, sentences_path, bm25_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到索引文件：{path}\n"
                "请先运行 build_bm25_index() 构建索引。\n"
                "示例：from src.retriever import build_bm25_index; build_bm25_index()"
            )

    print("正在加载 BM25 索引到内存（首次加载约需 1-3 分钟）...")

    with open(doc_ids_path, 'rb') as f:
        _BM25_DOC_IDS = pickle.load(f)
    print(f"  doc_ids 加载完成：{len(_BM25_DOC_IDS)} 篇文档")

    with open(sentences_path, 'rb') as f:
        _SENTENCES_STORE = pickle.load(f)
    print(f"  sentences 加载完成")

    with open(bm25_path, 'rb') as f:
        _BM25_INDEX = pickle.load(f)
    print("  BM25 对象加载完成")

    print("索引加载完毕，可以开始检索。")
    return _BM25_INDEX, _BM25_DOC_IDS, _SENTENCES_STORE

# 模块级 sentences_store 单例（由 load_bm25_index 写入）
_SENTENCES_STORE: Optional[dict] = None

# ---------------------------------------------------------------------------
# 新版三阶段检索主函数
# ---------------------------------------------------------------------------

def retrieve_evidence_from_dump(
    claim: str,
    bm25_top_n: int = BM25_TOP_N_DOCS,
    sbert_top_k: int = SBERT_TOP_K_SENTENCES,
    index_dir: str = INDEX_DIR,
) -> str:
    """
    从本地 FEVER Wikipedia dump 中检索与 claim 相关的证据。

    实现标准三阶段流程：
      阶段1 - BM25 文档检索：
        对 claim 分词，用 BM25 从全量索引中召回 bm25_top_n 篇最相关文档。
        BM25Okapi 的打分公式考虑词频、逆文档频率和文档长度归一化。

      阶段2 - Sentence-BERT 句子筛选：
        把召回文档的所有句子编码成向量，计算与 claim 向量的余弦相似度，
        取 Top sbert_top_k 个句子作为最终证据。

      阶段3 - 格式化输出：
        返回带来源标注的证据字符串，供 prompt_builder 拼入 LLM prompt。

    参数：
      claim:       待验证的陈述文本
      bm25_top_n:  BM25 召回的候选文档数（默认 5）
      sbert_top_k: 最终保留的证据句子数（默认 5）
      index_dir:   索引目录路径

    返回：
      证据字符串（多句用换行分隔），或以 "RETRIEVAL_ERROR:" 开头的错误信息
    """
    try:
        # ----------------------------------------------------------------
        # 加载索引（单例，第一次调用时加载，后续直接复用）
        # ----------------------------------------------------------------
        bm25, doc_ids, sentences_store = load_bm25_index(index_dir)

        # ----------------------------------------------------------------
        # 阶段1：BM25 文档检索
        # ----------------------------------------------------------------
        # 对 claim 做和建索引时完全相同的分词处理，保证一致性
        query_tokens = _tokenize(claim)

        if not query_tokens:
            return "RETRIEVAL_ERROR: claim 分词后为空，无法检索。"

        # get_top_n 返回得分最高的 bm25_top_n 篇文档的 id
        # 内部调用 BM25Okapi.get_scores() 对全量文档打分，再取 top-n
        top_doc_ids = bm25.get_top_n(query_tokens, doc_ids, n=bm25_top_n)

        if not top_doc_ids:
            return "RETRIEVAL_ERROR: BM25 未能检索到任何文档。"

        logger.debug(f"BM25 召回文档：{top_doc_ids}")

        # ----------------------------------------------------------------
        # 阶段2：从召回文档中收集候选句子
        # ----------------------------------------------------------------
        candidate_sentences = []  # [(doc_id, sent_id, sent_text), ...]

        for doc_id in top_doc_ids:
            sent_dict = sentences_store.get(doc_id, {})
            if not sent_dict:
                continue  # 该文档没有句子数据，跳过

            # 按句子编号排序，保证顺序稳定
            for sent_id in sorted(sent_dict.keys()):
                sent_text = sent_dict[sent_id]
                # 过滤过短的句子（通常是噪声或空行残留）
                if len(sent_text) < 15:
                    continue
                candidate_sentences.append((doc_id, sent_id, sent_text))
                # 每篇文档最多取 MAX_SENTENCES_PER_DOC 句，防止长文章占主导
                doc_sent_count = sum(1 for d, _, _ in candidate_sentences if d == doc_id)
                if doc_sent_count >= MAX_SENTENCES_PER_DOC:
                    break

        if not candidate_sentences:
            return "RETRIEVAL_ERROR: 召回文档中未找到有效句子。"

        logger.debug(f"候选句子数：{len(candidate_sentences)}")

        # ----------------------------------------------------------------
        # 阶段2：Sentence-BERT 句子筛选
        # ----------------------------------------------------------------
        # 提取纯文本列表供编码
        sent_texts = [s[2] for s in candidate_sentences]

        model = get_st_model()

        # encode 返回 tensor，convert_to_tensor=True 使后续 cos_sim 在 GPU/CPU 上高效运行
        claim_embedding  = model.encode(claim,      convert_to_tensor=True)
        corpus_embedding = model.encode(sent_texts, convert_to_tensor=True)

        # cos_sim 返回 (1, N) 的相似度矩阵，取第0行得到每句的得分
        cos_scores = util.cos_sim(claim_embedding, corpus_embedding)[0]

        # topk：k 不超过候选句子总数
        actual_k = min(sbert_top_k, len(candidate_sentences))
        top_results = torch.topk(cos_scores, k=actual_k)

        # top_results.indices 是得分最高的句子在 candidate_sentences 中的下标
        top_indices = top_results.indices.tolist()

        # ----------------------------------------------------------------
        # 阶段3：格式化证据输出
        # ----------------------------------------------------------------
        evidence_lines = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_id, sent_id, sent_text = candidate_sentences[idx]
            # 带来源标注：方便实验报告分析检索结果，也帮助 LLM 定位信息来源
            # doc_id 中的下划线还原为空格，更易读
            readable_title = doc_id.replace("_", " ")
            evidence_lines.append(
                f"[{rank}] ({readable_title}, 句#{sent_id}) {sent_text}"
            )

        evidence_text = "\n".join(evidence_lines)
        logger.debug(f"最终证据：\n{evidence_text}")
        return evidence_text

    except FileNotFoundError as e:
        # 索引未构建时给出明确提示
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

        model = get_st_model()
        claim_embedding  = model.encode(claim,         convert_to_tensor=True)
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