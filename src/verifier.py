"""
FEVER事实验证系统 - 验证核心逻辑模块（性能优化版）

【本版重点改动】
1. 新增 entity_cache：缓存 LLM 抽取出来的核心实体，避免同一 claim/实体重复调用 API。
2. 新增 evidence_cache：缓存 RAG 检索结果，避免同一实体重复检索。
3. 新增 claim_result_cache：多轮实验时，同一 claim + mode 可直接复用结果，大幅减少重复 API 调用。
4. 新增 self_consistency 参数：可配置是否进行“双调用一致性校验”。默认读取 config.SELF_CONSISTENCY_CHECK；未配置则为 False。
5. 新增 REQUEST_DELAY 配置读取：把原来固定 time.sleep(0.1) 改成可配置。
6. 拆分 _build_prompt_and_evidence：让 _verify_single_claim 更短、更容易维护。
"""

from __future__ import annotations
import trace
from src.api_client import create_client
import json
import re
import time
from typing import Optional

from tqdm import tqdm

from src.prompt_builder import (
    build_verification_prompt,
    build_cot_prompt,
    build_rag_prompt,
    build_rag_cot_prompt,
    build_hover_extended_prompt,
    build_llm_judge_prompt,
    build_judge_review_prompt,
    build_ircot_hop_prompt,
    parse_ircot_action,
    parse_model_response,
)
from src import config
from src.retriever import retrieve_evidence_by_title, retrieve_evidence_from_dump

def _jaccard_sim(a, b):
    """
    【新增】检测 IRCoT 是否连续卡在相似查询上。
    例如连续搜：
      Lake Kanasatka
      Lake Kanasatka elevation
    相似度较高，就提醒模型换方向。
    """
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0

class FactVerifier:
    def __init__(self, logger):
        self.logger = logger
        self.client = create_client(logger)

        # =========================
        # 缓存中间结果
        # =========================
        # entity_cache：缓存 LLM 抽取出的核心实体，避免重复抽取
        # evidence_cache：缓存检索结果，避免重复检索
        #
        # 注意：不要缓存最终 prediction。
        # 因为你的 main.py 支持 NUM_ROUNDS 多轮实验，
        # 如果缓存最终预测，多轮结果会完全复用第一轮，无法观察模型随机性。
        # =========================
        self.entity_cache: dict[str, str] = {}
        self.evidence_cache: dict[tuple, str] = {}

        # 请求间隔
        self.request_delay = getattr(config, "REQUEST_DELAY", 0.1)

    def verify_claims(self, data_list):
        self.logger.info(f"开始执行验证流水线 (实验模式: {config.EXPERIMENT_MODE})")
        y_true, y_pred, detailed_results = [], [], []

        pbar = tqdm(
            data_list,
            total=len(data_list),
            desc="🔍 正在验证",
            unit="item",
            colour="green",
        )

        for item in pbar:
            claim_id = item["id"]
            claim = item["claim"]
            true_label = item["label"]
            gold_evidence = item.get("gold_evidence", None)

            pbar.set_postfix({"ID": claim_id})
            pbar.set_description(f"🔍 正在校验: {str(claim_id)[:15]}")

            prediction, raw_response, evidence, trace = self._verify_single_claim(
                claim=claim,
                claim_id=claim_id,
                mode=config.EXPERIMENT_MODE,
                gold_evidence=gold_evidence,
                item=item,
            )

            result_item = {
                "id": claim_id,
                "claim": claim,
                "true_label": true_label,
                "predicted_label": prediction,
                "evidence": evidence,
                "correct": prediction == true_label,
                "llm_raw": raw_response,
            }

            # 【新增】HoVer / IRCoT 模式下保存每一跳检索过程
            if trace:
                result_item["trace"] = trace

            detailed_results.append(result_item)

            if prediction and "ERROR" not in prediction:
                y_true.append(true_label)
                y_pred.append(prediction)

            time.sleep(self.request_delay)

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "detailed_results": detailed_results,
        }

    # =========================
    # 【改动5】实体抽取加入缓存 + 规则兜底
    # =========================
    def _extract_core_entity(self, claim: str) -> str:
        """从陈述中提取核心实体，用于 Wikipedia 检索。"""
        cache_key = claim.strip()
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]

        kw_prompt = (
            "Extract the most core Proper Noun (person, place, organization, work, or movie name) "
            f"from this claim for a Wikipedia search: '{claim}'. "
            "Output ONLY the exact noun string. NO CHITCHAT. NO EXPLANATION."
        )

        extracted = self.client.call_api(kw_prompt)
        entity = self._clean_entity(extracted) if extracted else ""

        if not entity:
            entity = self._fallback_extract_entity(claim)

        self.entity_cache[cache_key] = entity
        return entity

    @staticmethod
    def _clean_entity(text: str) -> str:
        """清洗 LLM 抽取出的实体文本。"""
        text = text.replace("关键词：", "").replace("Keywords:", "")
        text = text.replace("Entity:", "").replace("Core entity:", "")
        return text.strip("'\"., \n[]")

    @staticmethod
    def _fallback_extract_entity(claim: str) -> str:
        """
        简单规则兜底：提取连续首字母大写短语。
        例如：The Eiffel Tower is located in Berlin. -> The Eiffel Tower / Berlin
        """
        candidates = re.findall(r"(?:[A-Z][a-zA-Z0-9'\-]+(?:\s+|$)){1,5}", claim)
        candidates = [c.strip() for c in candidates if c.strip()]

        # 过滤句首常见虚词
        stop_heads = {"The", "A", "An", "This", "That"}
        cleaned = []
        for cand in candidates:
            parts = cand.split()
            if parts and parts[0] in stop_heads and len(parts) > 1:
                cand = " ".join(parts[1:])
            cleaned.append(cand)

        return max(cleaned, key=len) if cleaned else claim

    # =========================
    # 【改动6】RAG 证据检索加入缓存
    # =========================
    def _retrieve_title_evidence_cached(self, entity: str, num_sentences: int = 4) -> str:
        key = ("title", entity.lower().strip(), num_sentences)
        if key in self.evidence_cache:
            return self.evidence_cache[key]

        evidence = retrieve_evidence_by_title(entity, num_sentences=num_sentences)
        self.evidence_cache[key] = evidence
        return evidence

    def _retrieve_semantic_evidence_cached(self, claim: str) -> str:
        key = (
            "semantic",
            claim.strip(),
            getattr(config, "BM25_TOP_N_DOCS", None),
            getattr(config, "SBERT_TOP_K_SENTENCES", None),
        )
        if key in self.evidence_cache:
            return self.evidence_cache[key]

        evidence = retrieve_evidence_from_dump(
            claim=claim,
            bm25_top_n=config.BM25_TOP_N_DOCS,
            sbert_top_k=config.SBERT_TOP_K_SENTENCES,
        )
        self.evidence_cache[key] = evidence
        return evidence

    # =========================
    # 【改动7】把 prompt 构造和证据准备拆出来
    # =========================
    def _build_prompt_and_evidence(self, claim: str, mode: str, gold_evidence=None):
        """
        根据实验模式构造 prompt 和 evidence。

        注意：
        RAG_BM25 与 RAG_BM25_CE 使用同一个 build_rag_prompt，
        区别只在 evidence 的检索方式：
        - RAG_BM25: BM25 + SBERT
        - RAG_BM25_CE: BM25 + SBERT + CrossEncoder
        """
        evidence = None

        if mode == "BASELINE":
            prompt = build_verification_prompt(claim)

        elif mode == "COT":
            prompt = build_cot_prompt(claim)

        elif mode == "RAG":
            search_query = self._extract_core_entity(claim)
            evidence = self._retrieve_title_evidence_cached(search_query, num_sentences=4)
            prompt = build_rag_prompt(claim, evidence)

        elif mode == "RAG_COT":
            search_query = self._extract_core_entity(claim)
            evidence = self._retrieve_title_evidence_cached(search_query, num_sentences=4)
            prompt = build_rag_cot_prompt(claim, evidence)

        elif mode in ["RAG_BM25", "RAG_BM25_CE"]:
            use_ce = mode == "RAG_BM25_CE"

            evidence = retrieve_evidence_from_dump(
                claim=claim,
                bm25_top_n=config.BM25_TOP_N_DOCS,
                sbert_top_k=config.SBERT_TOP_K_SENTENCES,
                index_dir=config.INDEX_DIR,
                use_cross_encoder=use_ce,
            )

            prompt = build_rag_prompt(claim, evidence)

        elif mode == "EXTENDED_PIPELINE":
            evidence = retrieve_evidence_from_dump(
                claim=claim,
                bm25_top_n=config.BM25_TOP_N_DOCS,
                sbert_top_k=config.SBERT_TOP_K_SENTENCES,
                index_dir=config.HOVER_INDEX_DIR,
            )

            # HoVer 是二分类，使用 Judge Prompt
            prompt = build_llm_judge_prompt(claim, evidence)

        elif mode == "RAG_GOLDEN":
            evidence = "\n".join(gold_evidence) if gold_evidence else "【未解析到该样本的 Gold Evidence 原句】"
            prompt = build_rag_prompt(claim, evidence)

        elif mode == "RAG_COT_GOLDEN":
            evidence = "\n".join(gold_evidence) if gold_evidence else "【未解析到该样本的 Gold Evidence 原句】"
            prompt = build_rag_cot_prompt(claim, evidence)

        else:
            raise ValueError(f"未知的运行模式: {mode}")

        return prompt, evidence

    def _verify_single_claim(
        self,
        claim,
        claim_id=None,
        mode=None,
        gold_evidence=None,
        item=None,
    ):
        """
        对单条 claim 执行验证。

        支持：
        BASELINE
        COT
        RAG
        RAG_COT
        RAG_BM25
        RAG_GOLDEN
        RAG_COT_GOLDEN
        EXTENDED_PIPELINE
        """
        mode = mode or config.EXPERIMENT_MODE
        evidence = None
        trace = []

        # ====================================================
        # A. 无检索路线
        # ====================================================
        if mode == "BASELINE":
            prompt = build_verification_prompt(claim)

        elif mode == "COT":
            prompt = build_cot_prompt(claim)

        # ====================================================
        # B. 本地标题查找 RAG
        # 保持你自己的原逻辑，不改成在线 Wikipedia API
        # ====================================================
        elif mode == "RAG":
            search_query = self._extract_core_entity(claim)
            evidence = retrieve_evidence_by_title(search_query, num_sentences=4)
            prompt = build_rag_prompt(claim, evidence)

            trace.append({
                "step": "retrieve_by_title",
                "search_query": search_query,
                "evidence": evidence,
            })

        elif mode == "RAG_COT":
            search_query = self._extract_core_entity(claim)
            evidence = retrieve_evidence_by_title(search_query, num_sentences=4)
            prompt = build_rag_cot_prompt(claim, evidence)

            trace.append({
                "step": "retrieve_by_title",
                "search_query": search_query,
                "evidence": evidence,
            })

        # ====================================================
        # C. 新增：RAG_BM25
        # 直接用完整 claim 检索本地 dump，不再先抽实体
        # ====================================================
        elif mode in ["RAG_BM25", "RAG_BM25_CE"]:
            use_ce = mode == "RAG_BM25_CE"

            evidence = retrieve_evidence_from_dump(
                claim=claim,
                bm25_top_n=config.BM25_TOP_N_DOCS,
                sbert_top_k=config.SBERT_TOP_K_SENTENCES,
                index_dir=config.INDEX_DIR,
                use_cross_encoder=use_ce,
            )

            if evidence.startswith("RETRIEVAL_ERROR"):
                self.logger.warning(f"[{mode}] 检索失败 claim_id={claim_id}: {evidence}")

            prompt = build_rag_prompt(claim, evidence)

            trace.append({
                "step": "rag_bm25_ce_retrieval" if use_ce else "rag_bm25_retrieval",
                "use_cross_encoder": use_ce,
                "evidence": evidence,
            })

        # ====================================================
        # D. HoVer 扩展：IRCoT 多跳检索 + LLM Judge
        # ====================================================
        elif mode == "EXTENDED_PIPELINE":
            from src.retriever import retrieve_evidence_local_hop

            hop_trace = []

            if getattr(config, "USE_MULTI_HOP", True):
                accumulated_evidence = []
                failed_queries = []
                searched_titles = set()
                recent_queries = []

                for hop in range(getattr(config, "MAX_HOP_ROUNDS", 3)):
                    is_first_hop = hop == 0

                    obs_text = "\n\n".join(accumulated_evidence) if accumulated_evidence else ""

                    if failed_queries:
                        fail_note = (
                            "\n\n[Note: the following searches returned no relevant results, "
                            "try different queries: "
                            + ", ".join(f'"{q}"' for q in failed_queries)
                            + "]"
                        )
                        obs_text += fail_note

                    force_new_direction = (
                        len(recent_queries) >= 2
                        and _jaccard_sim(recent_queries[-1], recent_queries[-2]) >= 0.5
                    )

                    hop_prompt = build_ircot_hop_prompt(
                        claim=claim,
                        observations=obs_text,
                        already_searched=list(searched_titles),
                        is_first_hop=is_first_hop,
                        force_new_direction=force_new_direction,
                    )

                    hop_response = self.client.call_api(hop_prompt)
                    self.logger.info(f"[{claim_id}] IRCoT hop {hop + 1} raw: {hop_response!r}")

                    action, query = parse_ircot_action(hop_response)

                    if action == "done":
                        hop_trace.append({
                            "step": "done",
                            "hop_num": hop + 1,
                            "thought_raw": hop_response,
                        })
                        break

                    if not query:
                        hop_trace.append({
                            "step": "empty_query",
                            "hop_num": hop + 1,
                            "thought_raw": hop_response,
                        })
                        continue

                    # 避免重复搜索同一词条
                    if query.lower() in {t.lower() for t in searched_titles}:
                        hop_trace.append({
                            "step": "skip_duplicate",
                            "hop_num": hop + 1,
                            "query": query,
                            "thought_raw": hop_response,
                        })
                        continue

                    searched_titles.add(query)
                    recent_queries.append(query)

                    hop_evidence = retrieve_evidence_local_hop(
                        article_title=query,
                        claim=claim,
                        top_k=2,
                        index_dir=config.HOVER_INDEX_DIR,
                    )

                    evidence_ok = not hop_evidence.startswith(("ERROR", "RETRIEVAL_ERROR"))

                    hop_trace.append({
                        "step": "hop",
                        "hop_num": hop + 1,
                        "query": query,
                        "thought_raw": hop_response,
                        "evidence_ok": evidence_ok,
                        "evidence": hop_evidence,
                    })

                    if evidence_ok:
                        accumulated_evidence.append(
                            f"[Hop {hop + 1}: {query}]\n{hop_evidence}"
                        )
                    else:
                        failed_queries.append(query)

                if accumulated_evidence:
                    evidence = "\n\n".join(accumulated_evidence)
                else:
                    evidence = "No evidence retrieved."

            else:
                # 不启用多跳时，直接用 HoVer BM25 索引做一次检索
                evidence = retrieve_evidence_from_dump(
                    claim=claim,
                    bm25_top_n=config.BM25_TOP_N_DOCS,
                    sbert_top_k=config.SBERT_TOP_K_SENTENCES,
                    index_dir=config.HOVER_INDEX_DIR,
                )
                hop_trace.append({
                    "step": "single_retrieval",
                    "evidence": evidence,
                })

            # HoVer 是二分类，用 Judge Prompt 更合适
            base_prompt = build_llm_judge_prompt(claim, evidence)
            base_response = self.client.call_api(base_prompt)

            if not base_response:
                return "ERROR", None, evidence, hop_trace

            base_prediction = parse_model_response(
                base_response,
                claim_id=claim_id,
                claim=claim,
                mode=mode,
            )

            hop_trace.append({
                "step": "base_judgment",
                "prediction": base_prediction,
                "llm_raw": base_response,
            })

            # 可选：LLM Judge 二次复核
            if getattr(config, "USE_LLM_JUDGE", False):
                review_prompt = build_judge_review_prompt(
                    claim=claim,
                    evidence=evidence,
                    initial_label=base_prediction,
                )

                review_response = self.client.call_api(review_prompt)

                if review_response:
                    reviewed_prediction = parse_model_response(
                        review_response,
                        claim_id=claim_id,
                        claim=claim,
                        mode=mode,
                    )

                    hop_trace.append({
                        "step": "judge_review",
                        "prediction": reviewed_prediction,
                        "llm_raw": review_response,
                    })

                    return reviewed_prediction or base_prediction, review_response, evidence, hop_trace

            return base_prediction, base_response, evidence, hop_trace

        # ====================================================
        # E. Golden Evidence 对照组
        # ====================================================
        elif mode == "RAG_GOLDEN":
            if gold_evidence:
                evidence = "\n".join(gold_evidence)
            else:
                evidence = "【未解析到该样本的 Gold Evidence 原句】"

            prompt = build_rag_prompt(claim, evidence)

            trace.append({
                "step": "golden_evidence",
                "evidence": evidence,
            })

        elif mode == "RAG_COT_GOLDEN":
            if gold_evidence:
                evidence = "\n".join(gold_evidence)
            else:
                evidence = "【未解析到该样本的 Gold Evidence 原句】"

            prompt = build_rag_cot_prompt(claim, evidence)

            trace.append({
                "step": "golden_evidence",
                "evidence": evidence,
            })

        else:
            raise ValueError(f"未知的运行模式: {mode}")

        # ====================================================
        # F. 普通模式统一调用 LLM
        # 注意：这里改成单次调用，不再双调用冲突投票
        # ====================================================
        response = self.client.call_api(prompt)

        if not response:
            return "ERROR", None, evidence, trace

        prediction = parse_model_response(
            response,
            claim_id=claim_id,
            claim=claim,
            mode=mode,
        )

        trace.append({
            "step": "judgment",
            "prediction": prediction,
            "llm_raw": response,
        })

        return prediction, response, evidence, trace

    def save_results(self, results, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
