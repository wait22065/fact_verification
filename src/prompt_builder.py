"""
FEVER事实验证系统 - Prompt构造模块
"""
import re
from src.config import VALID_LABELS


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


def parse_model_response(response):
    """
    解析模型输出，提取标签

    Args:
        response: 模型的原始响应文本

    Returns:
        str: 提取的标签（SUPPORTS/REFUTES/NOT ENOUGH INFO），如果无法解析则返回None
    """
    if not response:
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

    # 无法解析
    return None


def validate_prediction(prediction):
    """
    验证预测结果是否有效

    Args:
        prediction: 预测的标签

    Returns:
        bool: 是否有效
    """
    return prediction in VALID_LABELS
