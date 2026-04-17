"""
FEVER事实验证系统 - 配置管理模块
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# 模型参数
TEMPERATURE = 0.1  # 低温度保证输出稳定性
MAX_TOKENS = 512
TIMEOUT = 30  # API超时时间（秒）

# 重试配置
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒
BACKOFF_FACTOR = 2  # 指数退避因子

# 数据配置
SAMPLE_SIZE = 50  # 每次采样的数据量
RANDOM_SEED = 42  # 保证可复现
FEVER_SPLIT = "validation"  # 使用验证集

# 标签映射
LABEL_MAPPING = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO"
}

# 有效标签列表
VALID_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# 路径配置
DATA_CACHE_DIR = "data/cache"
RESULTS_DIR = "data/results"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "verification.log")
RESULTS_FILE = os.path.join(RESULTS_DIR, "verification_results.json")

# 确保目录存在
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
