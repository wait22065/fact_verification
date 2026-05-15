"""
FEVER事实验证系统 - 配置管理模块
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================= 动态绝对路径配置 =================
# 获取当前 config.py 所在目录的上一级目录（即项目根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
RESULTS_DIR    = os.path.join(BASE_DIR, "data", "results")
LOG_DIR        = os.path.join(BASE_DIR, "logs")

LOG_FILE       = os.path.join(LOG_DIR,     "verification.log")
RESULTS_FILE   = os.path.join(RESULTS_DIR, "verification_results.json")
PARSE_ERROR_LOG = os.path.join(RESULTS_DIR, "parse_errors.json")

# 确保目录存在
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)

# ================= API 配置 =================
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME        = "deepseek-chat"

TEMPERATURE    = 0.1   # 模型输出随机性，越低越确定
MAX_TOKENS     = 1024  # 最大输出 token 数
TIMEOUT        = 60    # 单次请求超时时间（秒）
MAX_RETRIES    = 3     # API 失败最大重试次数
RETRY_DELAY    = 2     # 初始重试等待时间（秒）
RETRY_DELAY_MAX = 10   # 最大重试等待时间（秒）
BACKOFF_FACTOR = 2     # 指数退避倍率（每次重试等待时间 × 该值）

# ================= 实验配置 =================
# 可选值：BASELINE | COT | RAG | RAG_COT | RAG_BM25 | EXTENDED_PIPELINE
EXPERIMENT_MODE = "RAG_BM25"

RETRIEVER_TOP_K = 1    # 实时 API 检索时返回的文档数（仅 RAG / RAG_COT 有效）
SAMPLE_SIZE     = 50   # 随机采样条数，0 表示使用全部数据
RANDOM_SEED     = 42   # 随机种子，固定后每次采样结果完全一致
FEVER_SPLIT     = "labelled_dev"  # 数据集划分：labelled_dev 或 labelled_test

VALID_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# ================= BM25 本地检索配置（仅 RAG_BM25 模式使用）=================

# Wikipedia dump 解压后的目录，存放 109 个 wiki-*.jsonl 文件
# 约定放在项目根目录下的 data/wiki-pages/
DUMP_DIR = os.path.join(BASE_DIR, "wiki-pages")

# BM25 索引持久化目录，存放 doc_ids.pkl / sentences.pkl / bm25.pkl 三个文件
# 首次运行 build_bm25_index() 时自动创建，后续直接加载
INDEX_DIR = os.path.join(BASE_DIR, "data", "bm25_index")

# BM25 文档检索阶段：每条 claim 从全量索引中召回的候选文档数
# 增大可提升召回率（不遗漏相关文章），但会增加 SBERT 编码量，速度变慢
# 推荐范围：3（快）~ 10（全），默认 5
BM25_TOP_N_DOCS = 5

# Sentence-BERT 句子筛选阶段：从候选文档中最终保留的证据句子数
# 句子越多 → 命中 gold evidence 概率越高，但也引入更多噪声
# 推荐范围：3 ~ 5，默认 5
SBERT_TOP_K_SENTENCES = 5

# 每篇文档最多读入 SBERT 候选池的句子数
# 防止超长文章（如美国历史、战争页面）独占候选句子池
# 超出部分按句子编号顺序截断（文章后半段通常相关性低）
# 推荐范围：30 ~ 100，默认 50
MAX_SENTENCES_PER_DOC = 50

# Sentence-BERT 使用的模型名称（来自 sentence-transformers 库）
# all-MiniLM-L6-v2：轻量（80MB），384 维，速度快，适合 CPU 运行
# all-mpnet-base-v2：较重（420MB），768 维，精度更高，适合 GPU
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"