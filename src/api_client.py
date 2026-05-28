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

    def __init__(self, logger=None):
        """初始化客户端"""
        self.logger = logger  # 保存 logger
        if not DEEPSEEK_API_KEY:
            raise ValueError("未设置DEEPSEEK_API_KEY环境变量")

        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=TIMEOUT
        )

    def call_api(self, prompt, max_retries=MAX_RETRIES):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                error_msg = f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
                
                # 如果有 logger 就用 logger 输出，否则用 print
                if self.logger:
                    self.logger.warning(error_msg)
                else:
                    print(error_msg)

                if attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (BACKOFF_FACTOR ** attempt)
                    time.sleep(wait_time)
                else:
                    return None
        return None

def create_client(logger=None):
    """创建API客户端，支持传入logger"""
    return DeepSeekClient(logger)