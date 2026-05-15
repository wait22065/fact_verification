"""
FEVER事实验证系统 - 验证核心逻辑模块
"""
import time
from tqdm import tqdm
from src.api_client import create_client
from src.prompt_builder import (
    build_verification_prompt, build_cot_prompt, build_rag_prompt,
    build_rag_cot_prompt, build_llm_judge_prompt, parse_model_response
)
from src import config
from src.retriever import retrieve_evidence, retrieve_evidence_pipeline

class FactVerifier:
    def __init__(self, logger):
        self.logger = logger
        self.client = create_client(logger) # 传入 logger 避免 print 破坏进度条

    def verify_claims(self, data_list):
        self.logger.info(f"开始执行验证流水线 (Mode: {config.EXPERIMENT_MODE})")
        y_true, y_pred, detailed_results = [], [], []

        # Tqdm 进度条配置
        pbar = tqdm(data_list, total=len(data_list), desc="🔍 正在验证", unit="item", colour="green")

        for item in pbar:
            claim_id, claim, true_label = item['id'], item['claim'], item['label']
            pbar.set_postfix({"ID": claim_id})
            # 把信息写在进度条的后缀里，这样它就在同一行变动，不会换行
            pbar.set_description(f"🔍 正在校验: {claim_id[:15]}") 

            # 使用全局 mode 运行批量测试
            prediction, raw_response, evidence = self._verify_single_claim(
                claim, claim_id=claim_id, mode=config.EXPERIMENT_MODE, top_k=config.RETRIEVER_TOP_K
            )

            detailed_results.append({
                'id': claim_id, 'claim': claim, 'true_label': true_label,
                'predicted_label': prediction, 'evidence': evidence,
                'correct': prediction == true_label
            })

            if prediction and "ERROR" not in prediction:
                y_true.append(true_label)
                y_pred.append(prediction)
            
            time.sleep(0.1)

        return {
            'y_true': y_true, 'y_pred': y_pred, 'detailed_results': detailed_results
        }

    def _verify_single_claim(self, claim, claim_id=None, num_sentences=3, mode=None, top_k=1):
        """
        接收外部传入的 mode 和 top_k，而不是依赖 config (解决Web端并发冲突)
        """
        mode = mode or config.EXPERIMENT_MODE
        evidence = None
        search_query = claim

        # 提取实体
        if mode in ["RAG", "RAG_COT", "EXTENDED_PIPELINE"]:
            kw_prompt = f"Return 1-2 core keywords from this claim for Wikipedia search: '{claim}'. Reply ONLY with the keywords, separated by space."
            extracted = self.client.call_api(kw_prompt)
            if extracted and len(extracted.split()) <= 6:
                search_query = extracted.strip("'\"., ")

        # 扩展流水线 (多跳检索 + 裁判模型)
        if mode == "EXTENDED_PIPELINE":
            # self.logger.info(f"多跳检索中: {search_query}")
            evidence = retrieve_evidence_pipeline(claim, search_query, top_k_sentences=3)
            prompt = build_llm_judge_prompt(claim, evidence)
        
        # 基础任务
        elif mode == "COT":
            prompt = build_cot_prompt(claim)
        elif mode == "RAG":
            evidence = retrieve_evidence(search_query, top_k=top_k, num_sentences=num_sentences)
            prompt = build_rag_prompt(claim, evidence)
        elif mode == "RAG_COT":
            evidence = retrieve_evidence(search_query, top_k=top_k, num_sentences=num_sentences)
            prompt = build_rag_cot_prompt(claim, evidence)
        else: # BASELINE
            prompt = build_verification_prompt(claim)

        response = self.client.call_api(prompt)
        if not response: return "ERROR", None, evidence
        
        prediction = parse_model_response(response, claim_id=claim_id, claim=claim)
        return prediction, response, evidence

    def save_results(self, results, output_path):
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)