"""
FEVER事实验证系统 - Prompt构造模块 (深度融合优化版)
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

指令：
1. 如果你对该声明涉及的知识不确定，或者它极其冷门，请诚实地输出 NOT ENOUGH INFO，不要瞎猜。
2. 严禁使用 Markdown 加粗符号（不要使用 **）。

输出格式：
一句话事实：<你的直接判断依据>
Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""
    else:
        return f"""You are an authoritative fact-checking system. Verify the claim directly based on your internal knowledge.
Do not use step-by-step reasoning. Provide a one-sentence factual statement, then the verdict.

Claim: {claim}

Instructions:
1. If you are unsure about the facts or the entity is too obscure, honestly output NOT ENOUGH INFO. Do not guess.
2. Do NOT use markdown bold formatting (no **).

Output Format:
Fact: <Your direct factual basis>
Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""


# ==========================================
# 2. RAG: 检索增强 (证据优先，常识连桥，无多步推理)
# ==========================================
def build_rag_prompt(claim, evidence):
    if is_chinese(claim):
        return f"""请结合提供的维基百科证据，判断声明的真伪。

声明: {claim}
证据:
{evidence}

判定规则：
1. 证据优先：首先参考提供的证据是否能直接支撑或反驳声明。
2. 常识连桥：允许你使用基础常识将证据与声明相连接（例如：若证据说某人出生在加州，根据常识可判定其国籍为美国；若在巴黎，则在法国）。
3. 拒答底线：如果证据和你的内置常识都无法明确确定声明的真伪（例如遇到极其冷门的信息），请诚实输出 NOT ENOUGH INFO。
4. 严禁使用 Markdown 加粗符号（不要使用 **）。

输出格式：
证据对比：<用一两句话简述证据与声明的对比结果>
Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""
    else:
        return f"""Verify the claim based on the provided Wikipedia evidence.

Claim: {claim}
Evidence:
{evidence}

Rules:
1. Evidence First: First check if the provided evidence can directly support or refute the claim.
2. Common-sense Bridge: You are allowed to use basic common sense to bridge evidence and the claim (e.g., if born in California, they are American; if in Paris, they are in France).
3. Strict Refusal: If both the evidence and your internal knowledge cannot definitively verify the claim, honestly output NOT ENOUGH INFO. Do not guess.
4. Do NOT use markdown bold formatting (no **).

Output Format:
Comparison: <Brief 1-2 sentence comparison>
Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""


# ==========================================
# 3. COT: 思维链 (仅依靠内置知识，强制多步推理)
# ==========================================
def build_cot_prompt(claim):
    if is_chinese(claim):
        return f"""你是一个逻辑严密的专家。请务必使用中文，一步一步地推演以下声明的真伪。

声明: {claim}

推演指令：
1. 识别核心：提取声明中的核心实体和具体主张。
2. 知识检索：检索你的内部知识库，确认该实体的客观事实。
3. 置信度检查：如果你发现自己对该实体的信息非常模糊，请在分析中说明，并最终输出 NOT ENOUGH INFO。
4. 逻辑对比：如果知识清晰，展示清晰的无加粗数字逻辑链条（1... 2... 3...），对比主张与事实得出结论。
5. 严禁使用 Markdown 加粗符号（不要使用 **）。

分析过程：
1. 核心主张：...
2. 内部事实：...
3. 逻辑比对：...

Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""
    else:
        return f"""You are a logical expert. Think step-by-step to verify the following claim using ONLY your internal knowledge.

Claim: {claim}

Instructions:
1. Identify Core: Extract entities and the specific claim.
2. Knowledge Retrieval: Recall objective facts about the entities from your internal knowledge.
3. Confidence Check: If your internal knowledge about this is vague or missing, state this in your analysis and output NOT ENOUGH INFO.
4. Logical Comparison: Show a clear, numbered logical chain (1... 2... 3...) comparing facts against the claim.
5. Do NOT use markdown bold formatting (no **).

Analysis Process:
1. Core Claim: ...
2. Internal Facts: ...
3. Logical Comparison: ...

Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""


# ==========================================
# 4. RAG_COT: 检索 + 思维链 (推理 + 证据，最强模式)
# ==========================================
def build_rag_cot_prompt(claim, evidence):
    if is_chinese(claim):
        return f"""你是一个严谨的事实验证专家。请务必使用中文，结合提供的维基百科证据和你的内部常识，一步一步对声明进行验证。

声明: {claim}
证据:
{evidence}

[核心准则]：
1. 证据与常识结合：优先评估提供的证据是否有用；允许使用基础常识（如地理归属、出生地国籍等）来解读证据。
2. 拒绝过度脑补：如果提供的证据和你的内部知识中，都没有包含能判定声明真伪的关键细节，必须且只能输出 NOT ENOUGH INFO。
3. 严禁使用 Markdown 加粗符号（不要使用 **）。

分析过程：
1. 提取声明主张：...
2. 证据评估与常识补充：... (说明证据是否有用，并进行常识链接)
3. 综合逻辑推导：... (清晰列出推理步骤：1... 2... 3...)

Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""
    else:
        return f"""You are a rigorous fact-checking expert. Verify the claim step-by-step using both the provided evidence and your internal knowledge.

Claim: {claim}
Evidence:
{evidence}

[Core Principles]:
1. Evidence & Knowledge Fusion: Prioritize the provided evidence. You are allowed to use basic common sense (e.g., geography, birthright citizenship) to interpret and supplement the evidence.
2. No Over-extrapolation: If neither the provided evidence nor your internal knowledge contains the crucial details to verify the claim, you MUST output NOT ENOUGH INFO. Do not make unfounded guesses to force a conclusion.
3. Do NOT use markdown bold formatting (no **).

Analysis Process:
1. Claim Extraction: ...
2. Evidence Evaluation & Knowledge Supplement: ... (State if evidence is useful, and supplement with basic common sense)
3. Comprehensive Deduction: ... (List your logical steps: 1... 2... 3...)

Final Label: <SUPPORTS | REFUTES | NOT ENOUGH INFO>"""


# ==========================================
# 5. HOVER_EXTENDED: 多跳任务 (专门针对 HoVer 双分类优化)
# ==========================================
def build_hover_extended_prompt(claim, evidence):
    if is_chinese(claim):
        return f"""你是一个高级的事实验证侦探。请根据提供的维基百科证据，一步步进行多跳逻辑推理，判断声明的真伪。

声明: {claim}

维基百科证据:
{evidence}

推演指令：
1. 识别关系：识别声明中涉及的多个实体关系。
2. 侦探联想：像侦探一样，将不同证据片段关联起来（如：A是B的首都 $\rightarrow$ B在欧洲 $\rightarrow$ 推出A在欧洲）。
3. 强力裁决：即使证据碎片化，也请尽力拼凑逻辑链。你必须在 SUPPORTS 或 REFUTES 中做出最终裁决，不需要 NOT ENOUGH INFO 标签。
4. 严禁使用 Markdown 加粗符号（不要使用 **）。

分析过程：
1. 证据片段 A 证明了...
2. 证据片段 B 证明了...
3. 结合得出结论...

Final Label: <SUPPORTS | REFUTES>"""
    else:
        return f"""You are an advanced fact-checking detective. Reason step-by-step using the provided evidence to verify the multi-hop claim.

Claim: {claim}

Wikipedia Evidence:
{evidence}

Instructions:
1. Identify Relationships: Extract and connect multiple entity relationships in the claim.
2. Connect Dots: Bridge the dots between different evidence fragments (e.g., A is the capital of B, B is in Europe -> A is in Europe).
3. Hard Decision: Try your best to build a logical chain. You MUST make a definitive choice between SUPPORTS or REFUTES. No NOT ENOUGH INFO label is allowed.
4. Do NOT use markdown bold formatting (no **).

Analysis Process:
1. Evidence A shows...
2. Evidence B shows...
3. Combining them concludes...

Final Label: <SUPPORTS | REFUTES>"""

def build_llm_judge_prompt(claim, evidence):
    """
    【新增】HoVer / EXTENDED_PIPELINE 专用二分类判断 Prompt
    只允许输出 SUPPORTS / REFUTES，不允许 NOT ENOUGH INFO。
    适合 HoVer 这种双分类事实验证任务。
    """
    return f"""You are evaluating whether a claim is consistent with the provided evidence.

Claim: {claim}

Evidence:
{evidence}

Instructions:
Based ONLY on the evidence above, determine if the claim is supported or refuted.
1. Directional equivalence: "A is X [higher/older/longer] than B" is the same as "B is X [lower/younger/shorter] than A". Reason through such equivalences explicitly before comparing with the claim.
2. Numerical relationships, such as age differences, year gaps, and height differences, should be computed explicitly before comparing with the claim.
3. Output ONLY 'SUPPORTS' or 'REFUTES'. Do NOT output 'NOT ENOUGH INFO'.

Final Label: SUPPORTS or REFUTES

Analysis:"""


def build_ircot_hop_prompt(
    claim,
    observations,
    already_searched,
    is_first_hop=False,
    force_new_direction=False
):
    """
    【新增】IRCoT 单步检索 Prompt
    每次让 LLM 输出：
      Thought: ...
      Action: Search[某个维基词条]
    或：
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
            "returned the same document. You MUST search for a COMPLETELY DIFFERENT entity. "
            "Do NOT append suffixes like 'cast', 'list', 'actors', 'year', or 'film' "
            "to a term you have already searched. Instead, pick a DIFFERENT entity from the claim.\n"
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
        f"Output exactly two lines, no markdown, no asterisks, no bold:\n"
        f"  Thought: [one sentence about what you need to find next]\n"
        f"  Action: Search[Wikipedia article title, 2-5 words]"
        f"{done_line}\n\n"
        f"Your response:"
    )

def parse_ircot_action(response):
    """
    【新增】解析 IRCoT 单步输出。
    返回：
      ('search', title)
      或
      ('done', None)
    """
    if not response:
        return ('done', None)

    m_search = re.search(r'Action:\s*Search\[([^\]]+)\]', response, re.IGNORECASE)
    if m_search:
        return ('search', m_search.group(1).strip().strip("'\""))

    if re.search(r'Action:\s*Done', response, re.IGNORECASE):
        return ('done', None)

    # 兜底解析，防止模型没严格按格式输出
    m_fallback = re.search(
        r'(?:search(?:\s+for)?|look\s+up)[:\s]+["\']?([A-Z][^\n"\']{2,60})["\']?',
        response,
        re.IGNORECASE
    )
    if m_fallback:
        return ('search', m_fallback.group(1).strip())

    return ('done', None)
def build_judge_review_prompt(claim, evidence, initial_label):
    """
    【新增】LLM Judge 二次核验 Prompt。
    用于检查初始判断是否明显错误。
    """
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
# 响应解析与其他功能 (您的鲁棒版本，完美兼容各种输出)
# ==========================================
def parse_model_response(response, claim_id=None, claim=None, mode=None, allowed_labels=None):
    """
    【修改】解析模型响应，提取最终标签。
    兼容三分类 FEVER 和二分类 HoVer / EXTENDED_PIPELINE。
    """

    if allowed_labels is None:
        if mode == "EXTENDED_PIPELINE":
            allowed_labels = ["SUPPORTS", "REFUTES"]
        else:
            allowed_labels = VALID_LABELS

    fallback_label = "REFUTES" if mode == "EXTENDED_PIPELINE" else "NOT ENOUGH INFO"

    if not response:
        _log_parse_error(response, claim_id, claim, "空响应")
        return fallback_label

    cleaned_response = response.replace("**", "").replace("*", "")
    response_upper = cleaned_response.upper()

    # 1. 优先匹配 Final Label / Label / 结论 / 预测
    match = re.search(
        r'(?:FINAL\s*LABEL|LABEL|结论|预测).{0,10}?\s*'
        r'(SUPPORTS|REFUTES|NOT\s*ENOUGH\s*INFO|NEI)',
        response_upper
    )

    if match:
        label = match.group(1).strip()
        parsed = "NOT ENOUGH INFO" if ("NOT" in label or "NEI" in label) else label

        if parsed in allowed_labels:
            return parsed

        # HoVer 二分类中不允许 NEI，兜底成 REFUTES
        _log_parse_error(response, claim_id, claim, f"解析到非法标签: {parsed}")
        return fallback_label

    # 2. 从后往前找最后出现的标签，防止分析过程中提前出现 SUPPORTS / REFUTES
    supports_idx = response_upper.rfind("SUPPORTS")
    refutes_idx = response_upper.rfind("REFUTES")
    nei_idx = max(
        response_upper.rfind("NOT ENOUGH INFO"),
        response_upper.rfind("INSUFFICIENT"),
        response_upper.rfind("NEI")
    )

    max_idx = max(supports_idx, refutes_idx, nei_idx)

    if max_idx != -1:
        if max_idx == supports_idx:
            parsed = "SUPPORTS"
        elif max_idx == refutes_idx:
            parsed = "REFUTES"
        else:
            parsed = "NOT ENOUGH INFO"

        if parsed in allowed_labels:
            return parsed

        _log_parse_error(response, claim_id, claim, f"解析到非法标签: {parsed}")
        return fallback_label

    # 3. 最后兜底
    if "SUPPORT" in response_upper and "SUPPORTS" in allowed_labels:
        return "SUPPORTS"

    if "REFUTE" in response_upper and "REFUTES" in allowed_labels:
        return "REFUTES"

    _log_parse_error(response, claim_id, claim, "无法匹配任何标签")
    return fallback_label

def _log_parse_error(response, claim_id, claim, reason):
    """记录解析错误日志"""
    error_entry = { 
        "timestamp": datetime.now().isoformat(),
        "claim_id": claim_id,
        "claim": claim,
        "response": response,
        "reason": reason
    }
    _parse_errors.append(error_entry)


def save_parse_errors():
    """保存解析错误日志"""
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
    """清空解析错误日志"""
    _parse_errors.clear()


def validate_prediction(prediction):
    """验证预测是否有效"""
    return prediction in VALID_LABELS