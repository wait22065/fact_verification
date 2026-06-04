"""
FEVER事实验证系统 - 配置管理模块 (完善优化版)
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================= 动态绝对路径配置 =================
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

TEMPERATURE = 0.1 
MAX_TOKENS = 1500 
TIMEOUT = 60       
MAX_RETRIES = 3    
RETRY_DELAY = 2    
RETRY_DELAY_MAX = 10  
BACKOFF_FACTOR = 2  

# ================= 实验配置 =================
# 支持的运行模式对照：
# BASELINE | COT | RAG | RAG_COT | RAG_BM25 | RAG_BM25_CE | EXTENDED_PIPELINE | RAG_GOLDEN | RAG_COT_GOLDEN
EXPERIMENT_MODE = "RAG"

RETRIEVER_TOP_K = 2  
SAMPLE_SIZE = 500
RANDOM_SEED = 42  
FEVER_SPLIT = "labelled_dev"  
VALID_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# 验证评估迭代的轮数（1 表示单轮评估，大于 1 可用于捕捉并测试大模型输出的不确定性）
NUM_ROUNDS = 1  

# ================= 离线维基数据及检索配置 =================
WIKI_PAGES_DIR = os.path.join(BASE_DIR, "data", "wiki-pages")
DUMP_DIR = WIKI_PAGES_DIR  
INDEX_DIR = os.path.join(DATA_CACHE_DIR, "bm25_index")

# BM25 与 SBERT 离线检索微调参数
BM25_TOP_N_DOCS = 5          # BM25 第一阶段召回的候选文档数
SBERT_TOP_K_SENTENCES = 4     # 最终提供给大模型的证据句子数
MAX_SENTENCES_PER_DOC = 5    # 每篇文档最多进入候选池的句子数，防止长文刷屏
SBERT_MODEL_NAME = "all-MiniLM-L6-v2" # Sentence-BERT 模型名称

# SQLite 缓存配置（保留，可用作备用方案）
WIKI_DB_PATH = os.path.join(DATA_CACHE_DIR, "wiki_pages.db")

# ================= Golden Evidence 对照组模式配置 =================
# 指向带有真实 evidence 标注的原始 jsonl 文件
GOLDEN_FEVER_FILE = os.path.join(BASE_DIR, "data", "shared_task_dev.jsonl") 
# 提取出真实原句之后的缓存文件路径，避免重复扫描 5GB 的数据包
GOLDEN_EVIDENCE_CACHE = os.path.join(DATA_CACHE_DIR, "golden_evidence_cache.json")
# 是否强制清空缓存重新扫描
REBUILD_GOLDEN_EVIDENCE_CACHE = False


# ================= CrossEncoder 精排配置 =================
# CrossEncoder 用于在 SBERT 粗排后的候选句中进一步精排
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 默认关闭；如果 EXPERIMENT_MODE = "RAG_BM25_CE"，verifier.py 会主动开启
USE_CROSS_ENCODER = False

# 先从 SBERT 粗排中取多少条候选句给 CrossEncoder 精排
CE_RERANK_TOP_N = 20

# ================= HoVer / IRCoT 扩展配置 =================
HOVER_INDEX_DIR = os.path.join(DATA_CACHE_DIR, "hover_bm25_index")

# 是否启用多跳 IRCoT
USE_MULTI_HOP = True

# 多跳最多轮数
MAX_HOP_ROUNDS = 3

# 是否启用 CrossEncoder 精排，初期建议 False
USE_CROSS_ENCODER = False

# 是否启用 LLM Judge 二次复核，初期建议 False
USE_LLM_JUDGE = False

# 检索模式，初期建议只用 BM25
RETRIEVAL_MODE = "BM25"

# 每条请求之间的延迟
REQUEST_DELAY = 0.1