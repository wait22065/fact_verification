"""
FEVER事实验证系统 - Prompt构造模块
"""
import re
import json
import os
from datetime import datetime
from src.config import VALID_LABELS

# 错误记录文件路径
PARSE_ERROR_LOG = "data/results/parse_errors.json"

# 全局错误记录列表
_parse_errors = []


def build_verification_prompt(claim):
    """
    构造事实验证的Prompt（任务一：不包含evidence）

    Args:
        claim: 待验证的声明

    Returns:
        str: 完整的prompt
    """
    prompt = f"""You are a fact-checking assistant. Determine whether the following claim is true or false based on your knowledge.

Claim: {claim}

Instructions:
- Use your internal knowledge to evaluate the claim
- Respond with EXACTLY one of: SUPPORTS, REFUTES, NOT ENOUGH INFO

Answer:"""

    return prompt


def parse_model_response(response, claim_id=None, claim=None):
    """
    解析模型输出，提取标签

    Args:
        response: 模型的原始响应文本
        claim_id: claim的ID（用于错误记录）
        claim: claim的内容（用于错误记录）

    Returns:
        str: 提取的标签（SUPPORTS/REFUTES/NOT ENOUGH INFO），如果无法解析则返回None
    """
    if not response:
        _log_parse_error(response, claim_id, claim, "空响应")
        return None

    # 转换为大写便于匹配
    response_upper = response.upper()

    # 尝试直接匹配标签
    for label in VALID_LABELS:
        if label in response_upper:
            return label

    # 尝试匹配简化形式
    if "SUPPORT" in response_upper and "NOT" not in response_upper:
        return "SUPPORTS"
    elif "REFUTE" in response_upper:
        return "REFUTES"
    elif "NOT ENOUGH" in response_upper or "INSUFFICIENT" in response_upper:
        return "NOT ENOUGH INFO"

    # 无法解析，记录错误
    _log_parse_error(response, claim_id, claim, "无法匹配任何标签")
    print(f"\n⚠️  无法解析模型响应:")
    print(f"   Claim ID: {claim_id}")
    print(f"   响应内容: {response[:200]}...")  # 只打印前200字符
    print(f"   已记录到: {PARSE_ERROR_LOG}\n")

    return None


def _log_parse_error(response, claim_id, claim, reason):
    """
    记录解析错误

    Args:
        response: 模型响应
        claim_id: claim ID
        claim: claim内容
        reason: 错误原因
    """
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "claim_id": claim_id,
        "claim": claim,
        "response": response,
        "reason": reason
    }
    _parse_errors.append(error_entry)


def save_parse_errors():
    """
    保存所有解析错误到JSON文件
    """
    if not _parse_errors:
        return

    # 确保目录存在
    os.makedirs(os.path.dirname(PARSE_ERROR_LOG), exist_ok=True)

    # 如果文件已存在，加载现有错误
    existing_errors = []
    if os.path.exists(PARSE_ERROR_LOG):
        try:
            with open(PARSE_ERROR_LOG, 'r', encoding='utf-8') as f:
                existing_errors = json.load(f)
        except:
            pass

    # 合并并保存
    all_errors = existing_errors + _parse_errors
    with open(PARSE_ERROR_LOG, 'w', encoding='utf-8') as f:
        json.dump(all_errors, f, ensure_ascii=False, indent=2)

    print(f"\n已保存 {len(_parse_errors)} 条解析错误到: {PARSE_ERROR_LOG}")

    # 清空全局列表
    _parse_errors.clear()


def clear_parse_errors():
    """
    清空全局错误记录列表（用于新的验证任务）
    """
    _parse_errors.clear()


def validate_prediction(prediction):
    """
    验证预测结果是否有效

    Args:
        prediction: 预测的标签

    Returns:
        bool: 是否有效
    """
    return prediction in VALID_LABELS
