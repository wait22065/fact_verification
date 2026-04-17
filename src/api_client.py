"""
FEVER事实验证系统 - API调用模块
"""
import time
from openai import OpenAI
from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    BACKOFF_FACTOR
)


class DeepSeekClient:
    """DeepSeek API客户端"""

    def __init__(self):
        """初始化客户端"""
        if not DEEPSEEK_API_KEY:
            raise ValueError("未设置DEEPSEEK_API_KEY环境变量")

        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=TIMEOUT
        )

    def call_api(self, prompt, max_retries=MAX_RETRIES):
        """
        调用DeepSeek API

        Args:
            prompt: 输入的prompt
            max_retries: 最大重试次数

        Returns:
            str: 模型的响应文本，失败返回None
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )

                # 提取响应文本
                answer = response.choices[0].message.content.strip()
                return answer

            except Exception as e:
                error_msg = str(e)
                print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")

                # 如果还有重试机会，等待后重试
                if attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (BACKOFF_FACTOR ** attempt)
                    print(f"等待 {wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print("达到最大重试次数，放弃")
                    return None

        return None


def create_client():
    """
    创建API客户端

    Returns:
        DeepSeekClient: 客户端实例
    """
    return DeepSeekClient()
