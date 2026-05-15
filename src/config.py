
"""
FEVER事实验证系统 - 配置管理模块
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================= 动态绝对路径配置 =================
# 获取当前 config.py 所在目录的上一级目录 (即项目根目录)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
LOG_DIR = os.path.join(BASE_DIR, "logs")

LOG_FILE = os.path.join(LOG_DIR, "verification.log")
RESULTS_FILE = os.path.join(RESULTS_DIR, "verification_results.json")
PARSE_ERROR_LOG = os.path.join(RESULTS_DIR, "parse_errors.json")

# 确保目录存在
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ================= API 配置 =================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

TEMPERATURE = 0.1 # 控制模型输出的随机性，0.1 表示中等随机性
MAX_TOKENS = 1024  # 最大输出 token 数量
TIMEOUT = 60       # 放宽超时时间
MAX_RETRIES = 3    # 最大重试次数
RETRY_DELAY = 2    # 重试延迟时间，单位秒
RETRY_DELAY_MAX = 10  # 最大重试延迟时间，单位秒
BACKOFF_FACTOR = 2  # 重试延迟增加因子

# ================= 实验配置 (仅作为默认值和 main.py 的运行模式) =================
# Web 运行不应修改此项
EXPERIMENT_MODE = "EXTENDED_PIPELINE"  # 实验模式，"EXTENDED_PIPELINE" 或 "BASELINE"
RETRIEVER_TOP_K = 1  # 检索器返回的 top-k 结果数量
SAMPLE_SIZE = 50  # 样本大小，0 表示使用全部数据
RANDOM_SEED = 42  # 随机种子
FEVER_SPLIT = "labelled_dev"  # FEVER 数据集划分，"labelled_dev" 或 "labelled_test"

VALID_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]