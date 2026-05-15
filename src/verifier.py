"""
FEVER事实验证系统 - 验证核心逻辑模块

支持的 EXPERIMENT_MODE：
  BASELINE        - 直接用 LLM 内置知识判断，无检索
  COT             - 思维链推理，无检索
  RAG             - 实时 Wikipedia API 检索 + 简单判断
  RAG_COT         - 实时 Wikipedia API 检索 + 思维链
  RAG_BM25        - 本地 dump BM25+SBERT 两阶段检索 + 简单判断（新增）
  EXTENDED_PIPELINE - 实时 API 多跳检索 + LLM-as-Judge
"""
import time
import json
from tqdm import tqdm
from src.api_client import create_client
from src.prompt_builder import (
    build_verification_prompt, build_cot_prompt, build_rag_prompt,
    build_rag_cot_prompt, build_llm_judge_prompt, parse_model_response
)
from src import config
from src.retriever import (
    retrieve_evidence,
    retrieve_evidence_pipeline,
    retrieve_evidence_from_dump,  # 新增：本地 dump 两阶段检索
)


class FactVerifier:
    def __init__(self, logger):
        self.logger = logger
        self.client = create_client(logger)  # 传入 logger 避免 print 破坏进度条

    def verify_claims(self, data_list):
        self.logger.info(f"开始执行验证流水线 (Mode: {config.EXPERIMENT_MODE})")
        y_true, y_pred, detailed_results = [], [], []

        # tqdm 进度条：颜色和单位保持原样
        pbar = tqdm(
            data_list,
            total=len(data_list),
            desc="🔍 正在验证",
            unit="item",
            colour="green",
        )

        for item in pbar:
            claim_id  = item['id']
            claim     = item['claim']
            true_label = item['label']

            # 进度条后缀显示当前 claim id，在同一行变动不换行
            pbar.set_description(f"🔍 正在校验: {str(claim_id)[:15]}")
            pbar.set_postfix({"ID": claim_id})

            prediction, raw_response, evidence = self._verify_single_claim(
                claim,
                claim_id=claim_id,
                mode=config.EXPERIMENT_MODE,
                top_k=config.RETRIEVER_TOP_K,
            )

            detailed_results.append({
                'id':              claim_id,
                'claim':           claim,
                'true_label':      true_label,
                'predicted_label': prediction,
                'evidence':        evidence,
                'correct':         prediction == true_label,
            })

            # 只有成功解析的预测才纳入评估
            if prediction and "ERROR" not in prediction:
                y_true.append(true_label)
                y_pred.append(prediction)

            time.sleep(0.1)  # 避免 API 限速

        return {
            'y_true':           y_true,
            'y_pred':           y_pred,
            'detailed_results': detailed_results,
        }

    def _verify_single_claim(self, claim, claim_id=None, num_sentences=3, mode=None, top_k=1):
        """
        对单条 claim 执行验证，返回 (prediction, raw_response, evidence)。

        参数：
          claim:        待验证的陈述文本
          claim_id:     用于错误日志标记
          num_sentences: 实时 API 检索时取的摘要句数（仅 RAG/RAG_COT 有效）
          mode:         实验模式，优先使用传入值，否则读 config.EXPERIMENT_MODE
          top_k:        实时 API 检索时的文档数（仅 RAG/RAG_COT 有效）

        各模式的证据来源：
          BASELINE / COT      → evidence = None（不检索）
          RAG / RAG_COT       → 实时 Wikipedia API（需要先用 LLM 提取关键词）
          RAG_BM25            → 本地 dump BM25+SBERT（直接用 claim 检索，无需关键词）
          EXTENDED_PIPELINE   → 实时 API 多跳检索
        """
        mode = mode or config.EXPERIMENT_MODE
        evidence = None

        # ------------------------------------------------------------------
        # 旧方案（RAG / RAG_COT / EXTENDED_PIPELINE）：先用 LLM 提取关键词
        # RAG_BM25 不走这里，BM25 直接用 claim 全文检索，效果更好
        # ------------------------------------------------------------------
        search_query = claim
        if mode in ["RAG", "RAG_COT", "EXTENDED_PIPELINE"]:
            kw_prompt = (
                f"Return 1-2 core keywords from this claim for Wikipedia search: "
                f"'{claim}'. Reply ONLY with the keywords, separated by space."
            )
            extracted = self.client.call_api(kw_prompt)
            # 防止模型返回完整句子而不是关键词（超过6个词视为失败）
            if extracted and len(extracted.split()) <= 6:
                search_query = extracted.strip("'\"., ")

        # ------------------------------------------------------------------
        # 按模式分支构造 prompt
        # ------------------------------------------------------------------

        if mode == "EXTENDED_PIPELINE":
            # 多跳检索 + LLM-as-Judge
            evidence = retrieve_evidence_pipeline(claim, search_query, top_k_sentences=3)
            prompt   = build_llm_judge_prompt(claim, evidence)

        elif mode == "COT":
            # 思维链，无检索
            prompt = build_cot_prompt(claim)

        elif mode == "RAG":
            # 实时 API 检索 + 简单判断
            evidence = retrieve_evidence(search_query, top_k=top_k, num_sentences=num_sentences)
            prompt   = build_rag_prompt(claim, evidence)

        elif mode == "RAG_COT":
            # 实时 API 检索 + 思维链
            evidence = retrieve_evidence(search_query, top_k=top_k, num_sentences=num_sentences)
            prompt   = build_rag_cot_prompt(claim, evidence)

        elif mode == "RAG_BM25":
            # ------------------------------------------------------------------
            # 新模式：本地 dump 两阶段检索
            #
            # 为什么不提取关键词：
            #   BM25 对完整 claim 的检索效果优于 1-2 个关键词。
            #   完整 claim 包含所有实体和上下文词，IDF 加权后噪声词自然权重低，
            #   实体词权重高，不需要预先过滤。
            #   而实时 API 的 wikipedia.search() 是全文搜索引擎，关键词越短越好，
            #   两者使用场景不同。
            #
            # retrieve_evidence_from_dump 返回格式示例：
            #   [1] (Oliver Reed, 句#0) Oliver Reed was an English actor...
            #   [2] (Gladiator 2000 film, 句#3) Directed by Ridley Scott...
            #
            # 这个字符串直接作为 build_rag_prompt 的 evidence 参数，
            # 被 f-string 插入 prompt 的 Evidence: 区块，格式完全兼容。
            # ------------------------------------------------------------------
            self.logger.debug(f"[RAG_BM25] 开始检索 claim_id={claim_id}")
            evidence = retrieve_evidence_from_dump(claim)

            # 检索失败时记录日志，但仍继续（LLM 会看到错误信息，通常输出 NEI）
            if evidence.startswith("RETRIEVAL_ERROR"):
                self.logger.warning(f"[RAG_BM25] 检索失败 claim_id={claim_id}: {evidence}")

            prompt = build_rag_prompt(claim, evidence)

        else:
            # BASELINE：直接用 LLM 内置知识，无检索
            prompt = build_verification_prompt(claim)

        # ------------------------------------------------------------------
        # 调用 LLM API 并解析响应
        # ------------------------------------------------------------------
        response = self.client.call_api(prompt)
        if not response:
            return "ERROR", None, evidence

        prediction = parse_model_response(response, claim_id=claim_id, claim=claim)
        return prediction, response, evidence

    def save_results(self, results, output_path):
        """将验证结果保存为 JSON 文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)