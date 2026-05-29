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
    """基础核验 prompt：基于证据判断声明真伪，仅输出 SUPPORTS/REFUTES（与 HoVer 二分类标签对齐）。"""
    return f"""You are evaluating whether a claim is consistent with the provided evidence.

Claim: {claim}

Evidence:
{evidence}

Instructions:
Based ONLY on the evidence above, determine if the claim is supported or refuted.
1. Directional equivalence: "A is X [higher/older/longer] than B" is the same as "B is X [lower/younger/shorter] than A". Reason through such equivalences explicitly before comparing with the claim.
2. Numerical relationships (age differences, year gaps, height differences): compute the result explicitly before comparing with the claim.
3. Output ONLY 'SUPPORTS' or 'REFUTES'. Do NOT output 'NOT ENOUGH INFO'.

Final Label: SUPPORTS or REFUTES

Analysis:"""


def build_ircot_hop_prompt(claim, observations, already_searched, is_first_hop=False, force_new_direction=False):
    """
    IRCoT 单步 prompt：每调用一次，LLM 输出一句推理 + 下一个搜索词（或 Done）。

    输出格式（严格，无 Markdown）：
      Thought: [一句推理]
      Action: Search[Wikipedia article title]
    或（hop1+ 可用）：
      Thought: [一句推理]
      Action: Done
    """
    obs_section = ""
    if observations:
        obs_section = f"\nEvidence collected so far:\n{observations}\n"

    searched_section = ""
    if already_searched:
        titles = ", ".join(f'"{t}"' for t in already_searched)
        searched_section = f"\nAlready searched (do NOT repeat): {titles}\n"

    done_line = (
        "" if is_first_hop
        else "\n  Action: Done   (if evidence is already sufficient to judge the claim)"
    )

    few_shot = """\
The following is a complete 3-step example showing how to handle one claim from start to finish.

Claim: "The largest lake in New Hampshire sits nine vertical feet lower than Lake Kanasatka."

--- Step 1: no evidence yet ---

Thought: The claim compares elevations, so I should first find Lake Kanasatka's elevation.
Action: Search[Lake Kanasatka]

--- Step 2: after searching Lake Kanasatka ---

Evidence collected so far:
[Hop 1: Lake Kanasatka]
• Lake Kanasatka is a 358-acre lake in Carroll County, New Hampshire, located one-half mile north of and nine vertical feet higher than Lake Winnipesaukee.

Already searched: "Lake Kanasatka"

Thought: The evidence directly states Kanasatka is nine vertical feet higher than Winnipesaukee, and Winnipesaukee is the largest lake in New Hampshire — this is enough to judge the claim.
Action: Done

---
"""

    direction_warning = ""
    if force_new_direction:
        direction_warning = (
            "\nMANDATORY DIRECTION CHANGE: Your last two searches share the same core words and "
            "returned the same document. You MUST search for a COMPLETELY DIFFERENT entity — "
            "one that shares NO main words with your previous queries. "
            "Specifically: do NOT append suffixes like 'cast', 'list', 'actors', 'year', or 'film' "
            "to a term you have already searched. Instead, pick a DIFFERENT entity from the claim "
            "(e.g., a person's full name, a character's name, a show or song title not yet searched).\n"
        )

    return (
        f"You are verifying a multi-hop claim step by step using Wikipedia searches.\n"
        f"Study the examples below, then output your next step in the exact same format.\n\n"
        f"{few_shot}"
        f"Now handle this claim:\n"
        f'Claim: "{claim}"\n'
        f"{obs_section}"
        f"{searched_section}"
        f"{direction_warning}\n"
        f"Output exactly two lines (no markdown, no asterisks, no bold):\n"
        f"  Thought: [one sentence about what you need to find next]\n"
        f"  Action: Search[Wikipedia article title, 2-5 words]"
        f"{done_line}\n\n"
        f"Your response:"
    )


def parse_ircot_action(response):
    """
    解析 IRCoT 单步输出，返回 ('search', title) 或 ('done', None)。
    优先匹配 Action: Search[...] 格式，其次匹配 Action: Done。
    兜底：尝试从响应文本中提取搜索词，避免因格式偏差丢失一整跳。
    """
    if not response:
        return ('done', None)

    m_search = re.search(r'Action:\s*Search\[([^\]]+)\]', response, re.IGNORECASE)
    if m_search:
        return ('search', m_search.group(1).strip().strip("'\""))

    if re.search(r'Action:\s*Done', response, re.IGNORECASE):
        return ('done', None)

    # 兜底：寻找 Search: / search for / look up 等自由文本表达
    m_fallback = re.search(
        r'(?:search(?:\s+for)?|look\s+up)[:\s]+["\']?([A-Z][^\n"\']{2,60})["\']?',
        response, re.IGNORECASE
    )
    if m_fallback:
        return ('search', m_fallback.group(1).strip())

    return ('done', None)


def build_judge_review_prompt(claim, evidence, initial_label):
    """LLM Judge 二次核验 prompt：对基础核验的初步判断进行再审，可纠正错误标签。"""
    return f"""You are an impartial LLM Judge reviewing a fact-checking decision.

Claim: {claim}

Evidence:
{evidence}

Initial Verdict: {initial_label}

Strictly review the initial verdict against the evidence above.
- If the evidence contains numerical or directional relationships, verify the reasoning before deciding.
- Correct the verdict only if the evidence clearly contradicts it.
Output ONLY 'SUPPORTS' or 'REFUTES'.

Final Label: SUPPORTS or REFUTES

Review:"""


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