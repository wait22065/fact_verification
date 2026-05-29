"""
FEVER事实验证系统 - 验证核心逻辑模块

支持的 EXPERIMENT_MODE：
  BASELINE        - 直接用 LLM 内置知识判断，无检索
  COT             - 思维链推理，无检索
  RAG             - 实时 Wikipedia API 检索 + 简单判断
  RAG_COT         - 实时 Wikipedia API 检索 + 思维链
  RAG_BM25        - 本地 dump BM25+SBERT 两阶段检索 + 简单判断（新增）
  RAG_GOLDEN      - 直接使用 FEVER 标注 evidence 坐标解析出的原句
  EXTENDED_PIPELINE - 本地 dump IRCoT 多跳检索 + LLM-as-Judge
"""
import time
import json
from tqdm import tqdm
from src.api_client import create_client
from src.prompt_builder import (
    build_verification_prompt, build_cot_prompt, build_rag_prompt,
    build_rag_cot_prompt, build_llm_judge_prompt, build_judge_review_prompt,
    build_ircot_hop_prompt, parse_ircot_action, parse_model_response
)
from src import config


def _jaccard_sim(a, b):
    """计算两个查询词的 Jaccard 词袋相似度，用于检测 IRCoT 是否卡死在同一实体。"""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


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

            prediction, raw_response, evidence, hop_trace = self._verify_single_claim(
                claim,
                claim_id=claim_id,
                item=item,
                mode=config.EXPERIMENT_MODE,
                top_k=config.RETRIEVER_TOP_K,
            )

            result_item = {
                'id':              claim_id,
                'claim':           claim,
                'true_label':      true_label,
                'predicted_label': prediction,
                'evidence':        evidence,
                'correct':         prediction == true_label,
                'llm_raw':         raw_response,
            }
            if hop_trace:
                result_item['hop_trace'] = hop_trace
            detailed_results.append(result_item)

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

    def _verify_single_claim(self, claim, claim_id=None, item=None, num_sentences=3, mode=None, top_k=1):
        """
        对单条 claim 执行验证，返回 (prediction, raw_response, evidence)。

        参数：
          claim:        待验证的陈述文本
          claim_id:     用于错误日志标记
          item:         原始样本；RAG_GOLDEN 模式从中读取 gold_evidence
          num_sentences: 实时 API 检索时取的摘要句数（仅 RAG/RAG_COT 有效）
          mode:         实验模式，优先使用传入值，否则读 config.EXPERIMENT_MODE
          top_k:        实时 API 检索时的文档数（仅 RAG/RAG_COT 有效）

        各模式的证据来源：
          BASELINE / COT      → evidence = None（不检索）
          RAG / RAG_COT       → 实时 Wikipedia API（需要先用 LLM 提取关键词）
          RAG_BM25            → 本地 dump BM25+SBERT（直接用 claim 检索，无需关键词）
          RAG_GOLDEN          → FEVER 标注 page/sentence_id 对应的原句
          EXTENDED_PIPELINE   → 本地 dump IRCoT 多跳检索
        """
        mode = mode or config.EXPERIMENT_MODE
        evidence = None

        # ------------------------------------------------------------------
        # 关键词提取（A+B 方案）：仅 RAG / RAG_COT 使用
        #   A: 要求 LLM 给出单个 Wikipedia 词条标题格式，避免多实体合并或模糊关键词
        #   B: 同时要求一个备选词条，搜索失败时自动回退
        # EXTENDED_PIPELINE / RAG_BM25 不走这里：前者由 IRCoT 动态生成查询词，
        # 后者直接用 claim 全文做 BM25 检索，均不需要关键词提取
        # ------------------------------------------------------------------
        search_query    = claim
        search_fallback = None
        if mode in ["RAG", "RAG_COT"]:
            kw_prompt = (
                f"For the following claim, identify the single Wikipedia article title most "
                f"likely to contain the key fact needed to verify it. "
                f"Then provide one backup article title on the next line.\n"
                f"Format (follow exactly):\nMAIN: [article title]\nBACKUP: [article title]\n\n"
                f"Claim: '{claim}'"
            )
            extracted = self.client.call_api(kw_prompt)
            if extracted:
                for line in extracted.strip().splitlines():
                    line = line.strip()
                    if line.upper().startswith("MAIN:"):
                        val = line[5:].strip().strip("'\"., ")
                        if val:
                            search_query = val
                    elif line.upper().startswith("BACKUP:"):
                        val = line[7:].strip().strip("'\"., ")
                        if val:
                            search_fallback = val
                # LLM 未按格式回复时兜底：取第一行，不超过 8 词
                if search_query == claim:
                    first = extracted.strip().splitlines()[0].strip().strip("'\"., ")
                    if first and len(first.split()) <= 8:
                        search_query = first

        # ------------------------------------------------------------------
        # 按模式分支构造 prompt
        # ------------------------------------------------------------------

        if mode == "EXTENDED_PIPELINE":
            from src.retriever import retrieve_evidence_local_hop, retrieve_evidence_from_dump
            hop_trace = []

            if config.USE_MULTI_HOP:
                accumulated_evidence = []
                failed_queries       = []   # 检索失败的词条，告知 LLM 换方向
                searched_titles      = set()
                recent_queries       = []   # 最近成功发出的查询，用于相似度检测

                for hop in range(config.MAX_HOP_ROUNDS):
                    is_first = (hop == 0)
                    obs_text = "\n\n".join(accumulated_evidence) if accumulated_evidence else None
                    # 把失败的查询词附加到 observations，让 LLM 知道换方向
                    if failed_queries and obs_text:
                        obs_text += "\n\n[Note: the following searches returned no relevant results, try different queries: "
                        obs_text += ", ".join(f'"{q}"' for q in failed_queries) + "]"
                    elif failed_queries:
                        obs_text = "[Note: the following searches returned no relevant results, try different queries: "
                        obs_text += ", ".join(f'"{q}"' for q in failed_queries) + "]"

                    # 检测最近两跳是否卡死在同一实体（Jaccard >= 0.5），若是则触发换方向警告
                    force_new = (
                        len(recent_queries) >= 2 and
                        _jaccard_sim(recent_queries[-1], recent_queries[-2]) >= 0.5
                    )

                    hop_prompt = build_ircot_hop_prompt(
                        claim             = claim,
                        observations      = obs_text,
                        already_searched  = list(searched_titles),
                        is_first_hop      = is_first,
                        force_new_direction = force_new,
                    )
                    raw_resp = self.client.call_api(hop_prompt)
                    self.logger.info(f"[{claim_id}] IRCoT hop {hop+1} raw: {raw_resp!r}")

                    action, query = parse_ircot_action(raw_resp)

                    if action == "done":
                        self.logger.info(f"[{claim_id}] IRCoT hop {hop+1}: Done")
                        hop_trace.append({
                            "step":        "done",
                            "hop_num":     hop + 1,
                            "thought_raw": raw_resp,
                        })
                        break

                    # 去重检查
                    if query.lower() in {t.lower() for t in searched_titles}:
                        self.logger.info(f"[{claim_id}] IRCoT hop {hop+1}: skip duplicate {query!r}")
                        hop_trace.append({
                            "step":        "skip_duplicate",
                            "hop_num":     hop + 1,
                            "thought_raw": raw_resp,
                            "query":       query,
                        })
                        continue
                    searched_titles.add(query)
                    recent_queries.append(query)  # 记录用于下一跳的相似度检测

                    # 本地检索（使用 HoVer 专属索引）
                    hop_evidence = retrieve_evidence_local_hop(
                        query, claim, top_k=2, index_dir=config.HOVER_INDEX_DIR
                    )
                    evidence_ok  = not hop_evidence.startswith(("ERROR", "RETRIEVAL_ERROR"))
                    self.logger.info(
                        f"[{claim_id}] IRCoT hop {hop+1} {'OK' if evidence_ok else 'FAIL'} "
                        f"for {query!r}:\n{hop_evidence[:500]}"
                    )
                    # thought + 检索结果合并为一条记录
                    hop_trace.append({
                        "step":        "hop",
                        "hop_num":     hop + 1,
                        "thought_raw": raw_resp,
                        "query":       query,
                        "evidence_ok": evidence_ok,
                        "evidence":    hop_evidence,
                    })
                    if evidence_ok:
                        accumulated_evidence.append(f"[Hop {hop+1}: {query}]\n{hop_evidence}")
                    else:
                        failed_queries.append(query)

                if not accumulated_evidence:
                    self.logger.warning(f"[{claim_id}] No evidence retrieved across all hops")
                evidence = "\n\n".join(accumulated_evidence) if accumulated_evidence else "No evidence retrieved."

            else:
                # 非多跳：本地检索，直接用 claim 全文做 BM25+SBERT（HoVer 索引）
                evidence = retrieve_evidence_from_dump(claim, index_dir=config.HOVER_INDEX_DIR)

            # Step 1: 基础核验（仅 SUPPORTS/REFUTES，与 HoVer 二分类标签对齐）
            base_prompt   = build_llm_judge_prompt(claim, evidence)
            base_response = self.client.call_api(base_prompt)
            if not base_response:
                return "ERROR", None, evidence, hop_trace
            base_prediction = parse_model_response(base_response, claim_id=claim_id, claim=claim)
            self.logger.info(
                f"[{claim_id}] Base judgment: {base_prediction!r}\n"
                f"LLM raw:\n{base_response}"
            )
            hop_trace.append({
                "step":       "base_judgment",
                "prompt":     base_prompt,
                "prediction": base_prediction,
                "llm_raw":    base_response,
            })

            # Step 2: LLM Judge 二次核验（可选，由 config.USE_LLM_JUDGE 控制）
            if config.USE_LLM_JUDGE and base_prediction and "ERROR" not in str(base_prediction):
                review_prompt   = build_judge_review_prompt(claim, evidence, base_prediction)
                review_response = self.client.call_api(review_prompt)
                if review_response:
                    reviewed = parse_model_response(review_response, claim_id=claim_id, claim=claim)
                    self.logger.info(
                        f"[{claim_id}] Judge review: {reviewed!r}\n"
                        f"LLM raw:\n{review_response}"
                    )
                    hop_trace.append({
                        "step":       "judge_review",
                        "prompt":     review_prompt,
                        "prediction": reviewed,
                        "llm_raw":    review_response,
                    })
                    return reviewed or base_prediction, review_response, evidence, hop_trace

            return base_prediction, base_response, evidence, hop_trace

        elif mode == "COT":
            # 思维链，无检索
            prompt = build_cot_prompt(claim)

        elif mode == "RAG":
            # 实时 API 检索 + 简单判断（B 方案：失败时用备选词条重试）
            from src.retriever import retrieve_evidence

            evidence = retrieve_evidence(search_query, top_k=top_k, num_sentences=num_sentences)
            if search_fallback and ("未找到" in evidence or "检索错误" in evidence):
                evidence = retrieve_evidence(search_fallback, top_k=top_k, num_sentences=num_sentences)
            prompt   = build_rag_prompt(claim, evidence)

        elif mode == "RAG_COT":
            # 实时 API 检索 + 思维链（B 方案：失败时用备选词条重试）
            from src.retriever import retrieve_evidence

            evidence = retrieve_evidence(search_query, top_k=top_k, num_sentences=num_sentences)
            if search_fallback and ("未找到" in evidence or "检索错误" in evidence):
                evidence = retrieve_evidence(search_fallback, top_k=top_k, num_sentences=num_sentences)
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
            from src.retriever import retrieve_evidence_from_dump

            self.logger.debug(f"[RAG_BM25] 开始检索 claim_id={claim_id}")
            evidence = retrieve_evidence_from_dump(claim)

            # 检索失败时记录日志，但仍继续（LLM 会看到错误信息，通常输出 NEI）
            if evidence.startswith("RETRIEVAL_ERROR"):
                self.logger.warning(f"[RAG_BM25] 检索失败 claim_id={claim_id}: {evidence}")

            prompt = build_rag_prompt(claim, evidence)

        elif mode == "RAG_GOLDEN":
            # Golden evidence 模式：不检索、不重排，严格使用 FEVER 标注的原句。
            gold_evidence = (item or {}).get("gold_evidence", [])
            if isinstance(gold_evidence, list):
                evidence = "\n".join(gold_evidence) if gold_evidence else "未解析到标注证据原句。"
            else:
                evidence = str(gold_evidence) if gold_evidence else "未解析到标注证据原句。"
            prompt = build_rag_prompt(claim, evidence)

        else:
            # BASELINE：直接用 LLM 内置知识，无检索
            prompt = build_verification_prompt(claim)

        # ------------------------------------------------------------------
        # 调用 LLM API 并解析响应
        # ------------------------------------------------------------------
        response = self.client.call_api(prompt)
        if not response:
            return "ERROR", None, evidence, None

        prediction = parse_model_response(response, claim_id=claim_id, claim=claim)

        trace = [{"step": "judgment", "prompt": prompt, "llm_raw": response}]
        if search_query != claim:
            trace[0]["search_query"] = search_query  # RAG 类模式记录实际使用的检索词

        return prediction, response, evidence, trace

    def save_results(self, results, output_path):
        """将验证结果保存为 JSON 文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
