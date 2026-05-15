"""
FEVER事实验证系统 - Prompt构造模块 (深度对齐版)
"""
import re
import json
import os
from datetime import datetime
from src.config import VALID_LABELS

PARSE_ERROR_LOG = "data/results/parse_errors.json"
_parse_errors = []

def is_chinese(claim):
    """判断陈述是否包含中文"""
    return bool(re.search(r'[\u4e00-\u9fa5]', claim))

# ==========================================
# 1. BASELINE: 直接判断 (无多步推理)
# ==========================================
def build_verification_prompt(claim):
    if is_chinese(claim):
        return f"""你是一个权威的事实验证系统。请直接根据你的内置知识判断以下声明的真伪。
不需要长篇大论的推理，请用一句话简述核心事实，然后直接给出结论。

声明: {claim}

输出格式：
一句话事实：<你的直接判断依据>
Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""
    else:
        return f"""You are an authoritative fact-checking system. Verify the claim directly based on your internal knowledge.
Do not use step-by-step reasoning. Provide a one-sentence factual statement, then the verdict.

Claim: {claim}

Output Format:
Fact: <Your direct factual basis>
Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""


# ==========================================
# 2. RAG: 检索增强 (允许基础常识，无多步推理)
# ==========================================
def build_rag_prompt(claim, evidence):
    if is_chinese(claim):
        return f"""请严格根据以下提供的维基百科证据，判断声明的真伪。

声明: {claim}
证据:
{evidence}

判定规则：
1. 你被允许使用基础常识将证据与声明相连接（例如：若证据说某人出生在美国加州，根据常识可判定其为美国人；若证据说在巴黎，可知在法国）。
2. 不需要长篇推理，用一两句话总结证据与声明的对比结果即可。

输出格式：
证据对比：<一两句话简述>
Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""
    else:
        return f"""Verify the claim based ONLY on the provided Wikipedia evidence.

Claim: {claim}
Evidence:
{evidence}

Rules:
1. You are allowed to use basic common sense to bridge evidence and the claim (e.g., if born in California, they are American).
2. Do not use step-by-step reasoning. Provide a brief 1-2 sentence comparison.

Output Format:
Comparison: <Brief comparison>
Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""


# ==========================================
# 3. COT: 思维链 (仅依靠内置知识，强制多步推理)
# ==========================================
def build_cot_prompt(claim):
    if is_chinese(claim):
        return f"""你是一个逻辑严密的专家。请务必使用中文，一步一步地推演以下声明的真伪。

声明: {claim}

推演指令：
1. 识别声明中的核心实体和主张。
2. 检索你的内部知识库，提取与该实体相关的客观事实。
3. 对比主张与事实，展示清晰的逻辑链条（（1）...（2）...（3）...）。
4. 严禁使用 Markdown 加粗符号（不要使用 **）。
5. 推演结束后，换行输出结论。

分析：
[在此处写出你的多步推理过程]

Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""
    else:
        return f"""You are a logical expert. Think step-by-step to verify the following claim.

Claim: {claim}

Instructions:
1. Identify core entities and claims.
2. Retrieve objective facts from your internal knowledge.
3. Show a clear, numbered logical chain (1... 2... 3...).
4. Do NOT use markdown bold formatting (no **).
5. Conclude on a new line.

Analysis:
[Write your step-by-step reasoning here]

Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""


# ==========================================
# 4. RAG_COT: 检索 + 思维链 (最强模式)
# ==========================================
def build_rag_cot_prompt(claim, evidence):
    if is_chinese(claim):
        return f"""你是一个事实验证专家。请务必使用中文，根据提供的证据，一步一步进行逻辑推理。

声明: {claim}

维基百科证据:
{evidence}

推演指令：
1. 允许使用基础常识（如地理归属、出生地国籍等）来解读证据。
2. 明确列出你的逻辑推演步骤（1... 2... 3...），解释证据是如何支持或反驳声明的。如果证据互不相关，说明为何信息不足。
3. 严禁使用 Markdown 加粗符号（不要使用 **）。
4. 推演结束后，换行输出最终结论。

分析：
[在此处写出你的多步推理过程]

Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""
    else:
        return f"""You are a fact-checking expert. Reason step-by-step using the provided evidence.

Claim: {claim}

Wikipedia Evidence:
{evidence}

Instructions:
1. Use basic common sense (e.g., geography, birthright citizenship) to interpret the evidence.
2. List your logical steps (1... 2... 3...), explaining how the evidence supports or refutes the claim.
3. Do NOT use markdown bold formatting (no **).
4. Conclude on a new line.

Analysis:
[Write your step-by-step reasoning here]

Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""


def build_llm_judge_prompt(claim, evidence):
    """【步骤3】大模型作为法官，直接判断一致性"""
    prompt = f"""You are an objective LLM-as-a-Judge evaluating a fact-checking task.

Is the claim consistent with the evidence?

Claim: {claim}

Evidence:
{evidence}

Instructions: 
Based ONLY on the evidence above, determine if the claim is supported or refuted.
Please output your final conclusion on a new line exactly as follows:
Final Label: SUPPORTS 或 REFUTES 或 NOT ENOUGH INFO

Analysis:"""
    return prompt


# ==========================================
# 响应解析与其他功能 (保持不变)
# ==========================================
def parse_model_response(response, claim_id=None, claim=None):
    if not response:
        _log_parse_error(response, claim_id, claim, "空响应")
        return None

    match = re.search(r'Final\s*Label:\s*(SUPPORTS|REFUTES|NOT\s*ENOUGH\s*INFO)', response, re.IGNORECASE)
    if match:
        label = match.group(1).upper()
        return "NOT ENOUGH INFO" if "NOT" in label else label
        
    response_upper = response.upper()
    supports_idx = response_upper.rfind("SUPPORTS")
    refutes_idx = response_upper.rfind("REFUTES")
    nei_idx = max(response_upper.rfind("NOT ENOUGH INFO"), response_upper.rfind("INSUFFICIENT"))
    
    max_idx = max(supports_idx, refutes_idx, nei_idx)
    if max_idx != -1:
        if max_idx == supports_idx: return "SUPPORTS"
        if max_idx == refutes_idx: return "REFUTES"
        if max_idx == nei_idx: return "NOT ENOUGH INFO"

    if "SUPPORT" in response_upper and "NOT" not in response_upper: return "SUPPORTS"
    if "REFUTE" in response_upper: return "REFUTES"

    _log_parse_error(response, claim_id, claim, "无法匹配任何标签")
    return None

def _log_parse_error(response, claim_id, claim, reason):
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "claim_id": claim_id,
        "claim": claim,
        "response": response,
        "reason": reason
    }
    _parse_errors.append(error_entry)

def save_parse_errors():
    if not _parse_errors:
        return
    os.makedirs(os.path.dirname(PARSE_ERROR_LOG), exist_ok=True)
    existing_errors = []
    if os.path.exists(PARSE_ERROR_LOG):
        try:
            with open(PARSE_ERROR_LOG, 'r', encoding='utf-8') as f:
                existing_errors = json.load(f)
        except:
            pass
    all_errors = existing_errors + _parse_errors
    with open(PARSE_ERROR_LOG, 'w', encoding='utf-8') as f:
        json.dump(all_errors, f, ensure_ascii=False, indent=2)
    _parse_errors.clear()

def clear_parse_errors():
    _parse_errors.clear()

def validate_prediction(prediction):
    return prediction in VALID_LABELS