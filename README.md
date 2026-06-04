# FEVER / HoVer Fact Verification System

本项目是一个基于大语言模型的事实验证系统，主要面向 FEVER 数据集，并扩展支持 HoVer 多跳事实验证任务。系统能够输入一条自然语言声明，结合不同实验模式输出事实验证标签：

* `SUPPORTS`
* `REFUTES`
* `NOT ENOUGH INFO`

项目实现了 Baseline、CoT、RAG、RAG_COT、BM25+SBERT、BM25+SBERT+CrossEncoder、Gold Evidence 对照组，以及 HoVer 多跳扩展流程。

---

## 1. 项目功能

本项目支持以下功能：

1. 基于大模型内置知识的直接事实判断。
2. 基于 Chain-of-Thought 的逐步推理事实判断。
3. 基于 Wikipedia 证据的 RAG 事实验证。
4. 基于 BM25 + Sentence-BERT 的自动证据选择。
5. 基于 CrossEncoder 的证据精排。
6. 基于 Gold Evidence 的上限对照实验。
7. 基于 HoVer 的多跳事实验证扩展。
8. 多轮实验与均值、标准差统计。
9. Accuracy、Precision、Recall、F1、Hallucination Rate 等指标计算。
10. 日志保存、结果保存、解析错误保存。
11. 支持不同实验模式自动加载不同数据集。
12. 支持自动检查和构建 BM25 索引。

---

## 2. 项目结构

```text
fact_verification/
├── main.py
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── api_client.py
│   ├── data_loader.py
│   ├── golden_loader.py
│   ├── prompt_builder.py
│   ├── retriever.py
│   ├── verifier.py
│   ├── evaluator.py
│   └── utils.py
├── data/
│   ├── shared_task_dev.jsonl
│   ├── wiki-pages/
│   ├── cache/
│   └── results/
├── logs/
└── local_models/
```

---

## 3. 环境安装

### 3.1 创建虚拟环境

推荐使用 `uv`：

```bash
uv venv
```

Windows 激活：

```bash
.venv\Scripts\activate
```

Linux / macOS 激活：

```bash
source .venv/bin/activate
```

也可以使用普通 venv：

```bash
python -m venv .venv
```

---

### 3.2 安装依赖

```bash
pip install -r requirements.txt
```

如果使用 Sentence-BERT 和 CrossEncoder，需要安装：

```bash
pip install sentence-transformers
```

如果使用 tqdm、scikit-learn、dotenv 等依赖：

```bash
pip install tqdm scikit-learn python-dotenv
```

---

## 4. API Key 配置

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

系统会在 `src/config.py` 中通过：

```python
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
```

读取 API Key。

---

## 5. 数据准备

### 5.1 FEVER 数据

请将 FEVER 官方 dev 文件放到：

```text
data/shared_task_dev.jsonl
```

系统首次运行时会解析该文件，并缓存为：

```text
data/fever_labelled_dev.json
```

后续运行会优先读取本地缓存。

---

### 5.2 Wikipedia Dump

请将 FEVER Wikipedia dump 解压到：

```text
data/wiki-pages/
```

目录中应包含类似：

```text
wiki-001.jsonl
wiki-002.jsonl
...
```

该 dump 用于：

* BM25 索引构建
* Sentence-BERT 句子筛选
* Gold Evidence 原句反查
* HoVer 检索

---

### 5.3 HoVer 数据

如果需要运行 HoVer 扩展实验，请将 HoVer dev 数据放到：

```text
data/cache/hover_dev.json
```

数据加载模块会读取其中的 `claim`、`label` 和 `supporting_facts` 字段，并将 HoVer 标签映射为系统统一标签：

```text
SUPPORTED -> SUPPORTS
REFUTED   -> REFUTES
```

---

## 6. 配置文件说明

主要配置位于：

```text
src/config.py
```

### 6.1 API 配置

```python
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

TEMPERATURE = 0.1
MAX_TOKENS = 1024
TIMEOUT = 90
MAX_RETRIES = 3
RETRY_DELAY = 5
BACKOFF_FACTOR = 2
REQUEST_DELAY = 1.0
```

建议事实验证任务使用较低温度：

```python
TEMPERATURE = 0.1
```

如果 API 经常连接失败，建议增大：

```python
REQUEST_DELAY = 1.0
TIMEOUT = 90
RETRY_DELAY = 5
```

---

### 6.2 实验模式

核心参数：

```python
EXPERIMENT_MODE = "RAG_GOLDEN"
```

支持以下模式：

| 模式                  | 说明                                     |
| ------------------- | -------------------------------------- |
| `BASELINE`          | 直接使用大模型内置知识判断                          |
| `COT`               | 无证据，使用逐步推理                             |
| `RAG`               | 标题检索 Wikipedia 页面前几句                   |
| `RAG_COT`           | 标题检索 + CoT                             |
| `RAG_BM25`          | BM25 + Sentence-BERT 自动证据选择            |
| `RAG_BM25_CE`       | BM25 + Sentence-BERT + CrossEncoder 精排 |
| `RAG_GOLDEN`        | 使用人工 Gold Evidence                     |
| `RAG_COT_GOLDEN`    | Gold Evidence + CoT                    |
| `EXTENDED_PIPELINE` | HoVer 多跳事实验证扩展                         |

---

### 6.3 样本与轮次

```python
SAMPLE_SIZE = 50
RANDOM_SEED = 42
NUM_ROUNDS = 1
```

调试时建议：

```python
SAMPLE_SIZE = 5
```

正式实验建议：

```python
SAMPLE_SIZE = 50
```

如果 API 稳定，可改为：

```python
SAMPLE_SIZE = 100
```

---

### 6.4 检索参数

```python
BM25_TOP_N_DOCS = 5
SBERT_TOP_K_SENTENCES = 4
MAX_SENTENCES_PER_DOC = 5
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
```

含义：

| 参数                      | 说明                 |
| ----------------------- | ------------------ |
| `BM25_TOP_N_DOCS`       | BM25 召回候选文档数量      |
| `SBERT_TOP_K_SENTENCES` | 最终返回给 LLM 的证据句数量   |
| `MAX_SENTENCES_PER_DOC` | 每篇文档最多取多少句参与排序     |
| `SBERT_MODEL_NAME`      | Sentence-BERT 模型名称 |

---

### 6.5 CrossEncoder 参数

```python
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_CROSS_ENCODER = False
CE_RERANK_TOP_N = 20
```

`RAG_BM25_CE` 模式会启用 CrossEncoder 精排，用于进一步筛选更相关的证据句。

---

### 6.6 Gold Evidence 参数

```python
GOLDEN_FEVER_FILE = os.path.join(BASE_DIR, "data", "shared_task_dev.jsonl")
GOLDEN_EVIDENCE_CACHE = os.path.join(DATA_CACHE_DIR, "golden_evidence_cache.json")
REBUILD_GOLDEN_EVIDENCE_CACHE = False
```

如果修改了 Gold Evidence 解析逻辑，第一次运行建议：

```python
REBUILD_GOLDEN_EVIDENCE_CACHE = True
```

缓存重建完成后改回：

```python
REBUILD_GOLDEN_EVIDENCE_CACHE = False
```

---

### 6.7 HoVer 扩展参数

```python
HOVER_INDEX_DIR = os.path.join(DATA_CACHE_DIR, "hover_bm25_index")

USE_MULTI_HOP = True
MAX_HOP_ROUNDS = 3
USE_LLM_JUDGE = False
RETRIEVAL_MODE = "BM25"
```

含义：

| 参数               | 说明              |
| ---------------- | --------------- |
| `USE_MULTI_HOP`  | 是否启用 IRCoT 多跳检索 |
| `MAX_HOP_ROUNDS` | 最大检索跳数          |
| `USE_LLM_JUDGE`  | 是否启用二次 Judge 复核 |
| `RETRIEVAL_MODE` | 检索模式，默认 BM25    |

---

## 7. 运行方法

在项目根目录运行：

```bash
python main.py
```

程序会自动根据 `EXPERIMENT_MODE` 执行对应实验。

---

## 8. 推荐运行顺序

### 8.1 调试阶段

先用小样本跑通：

```python
SAMPLE_SIZE = 5
NUM_ROUNDS = 1
REQUEST_DELAY = 1.0
```

依次测试：

```python
EXPERIMENT_MODE = "BASELINE"
EXPERIMENT_MODE = "RAG_GOLDEN"
EXPERIMENT_MODE = "RAG_BM25"
EXPERIMENT_MODE = "RAG_BM25_CE"
EXPERIMENT_MODE = "EXTENDED_PIPELINE"
```

---

### 8.2 正式 FEVER 实验

建议依次运行：

```python
EXPERIMENT_MODE = "BASELINE"
EXPERIMENT_MODE = "COT"
EXPERIMENT_MODE = "RAG"
EXPERIMENT_MODE = "RAG_COT"
EXPERIMENT_MODE = "RAG_BM25"
EXPERIMENT_MODE = "RAG_BM25_CE"
EXPERIMENT_MODE = "RAG_GOLDEN"
EXPERIMENT_MODE = "RAG_COT_GOLDEN"
```

推荐参数：

```python
SAMPLE_SIZE = 50
NUM_ROUNDS = 1
RANDOM_SEED = 42
REQUEST_DELAY = 1.0
TEMPERATURE = 0.1
```

---

### 8.3 HoVer 扩展实验

单跳检索：

```python
EXPERIMENT_MODE = "EXTENDED_PIPELINE"
USE_MULTI_HOP = False
USE_LLM_JUDGE = False
```

IRCoT 多跳：

```python
EXPERIMENT_MODE = "EXTENDED_PIPELINE"
USE_MULTI_HOP = True
MAX_HOP_ROUNDS = 3
USE_LLM_JUDGE = False
```

IRCoT + Judge：

```python
EXPERIMENT_MODE = "EXTENDED_PIPELINE"
USE_MULTI_HOP = True
MAX_HOP_ROUNDS = 3
USE_LLM_JUDGE = True
```

---

## 9. 输出文件

实验结果保存在：

```text
data/results/
```

日志保存在：

```text
logs/
```

输出文件名示例：

```text
verification_results_rag_bm25_20260101_120000.json
verification_rag_bm25_20260101_120000.log
```

结果 JSON 主要包含：

```json
{
  "experiment_mode": "RAG_BM25",
  "timestamp": "...",
  "num_rounds": 1,
  "valid_rounds": 1,
  "random_seed": 42,
  "sample_size": 50,
  "aggregated_metrics": {
    "accuracy": {"mean": 0.0, "std": 0.0},
    "macro_f1": {"mean": 0.0, "std": 0.0}
  },
  "rounds": [
    {
      "round": 1,
      "metrics": {},
      "detailed_results": []
    }
  ]
}
```

每条样本的详细结果包括：

* `id`
* `claim`
* `true_label`
* `predicted_label`
* `evidence`
* `correct`
* `llm_raw`
* `trace`

---

## 10. 评估指标

系统计算以下指标：

| 指标                 | 说明     |
| ------------------ | ------ |
| Accuracy           | 整体准确率  |
| Macro Precision    | 宏平均精确率 |
| Macro Recall       | 宏平均召回率 |
| Macro F1           | 宏平均 F1 |
| Weighted Precision | 加权精确率  |
| Weighted Recall    | 加权召回率  |
| Weighted F1        | 加权 F1  |
| Hallucination Rate | 幻觉率    |
| Confusion Matrix   | 混淆矩阵   |

幻觉率定义为：

```text
真实标签为 NOT ENOUGH INFO 的样本中，被模型预测为 SUPPORTS 或 REFUTES 的比例。
```

---

## 11. 绘图建议

实验完成后，可以从结果 JSON 中读取 `aggregated_metrics`，绘制以下图表：

1. FEVER 各模式 Accuracy 对比。
2. FEVER 各模式 Macro F1 对比。
3. FEVER 各模式 Hallucination Rate 对比。
4. HoVer 单跳、多跳、多跳+Judge 消融对比。

推荐 FEVER 图表横轴：

```text
BASELINE
COT
RAG
RAG_COT
RAG_BM25
RAG_BM25_CE
RAG_GOLDEN
RAG_COT_GOLDEN
```

推荐 HoVer 图表横轴：

```text
Single Retrieval
IRCoT
IRCoT + Judge
```

---

## 12. 常见问题

### 12.1 API Connection error

如果出现：

```text
API调用失败: Connection error.
```

建议修改：

```python
REQUEST_DELAY = 1.0
TIMEOUT = 90
RETRY_DELAY = 5
MAX_RETRIES = 3
```

并先使用：

```python
SAMPLE_SIZE = 5
```

确认能跑通。

---

### 12.2 找不到 DeepSeek API Key

检查 `.env` 文件中是否存在：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

---

### 12.3 找不到 Wikipedia dump

确认目录：

```text
data/wiki-pages/
```

下存在：

```text
wiki-*.jsonl
```

---

### 12.4 找不到 HoVer 数据

确认文件存在：

```text
data/cache/hover_dev.json
```

---

### 12.5 CrossEncoder 第一次运行很慢

`RAG_BM25_CE` 第一次运行时可能需要下载或加载 CrossEncoder 模型。建议先用：

```python
SAMPLE_SIZE = 5
```

测试，确认模型加载成功后再正式运行。

---

## 13. 实验注意事项

1. `RAG_BM25` 和 `RAG_BM25_CE` 使用的是 filtered BM25 索引，即候选页面来自标注 evidence pages，报告中需要说明这不是完全开放域检索。
2. `RAG_GOLDEN` 和 `RAG_COT_GOLDEN` 是上限对照组，不应与普通 RAG 混淆。
3. HoVer 是二分类任务，与 FEVER 三分类任务应分开分析。
4. 多轮实验不要缓存最终 prediction，否则无法观察模型输出随机性。
5. 如果网络不稳定，应降低样本量并增大请求间隔。
6. Prompt 解析失败会保存到 `data/results/parse_errors.json`，可用于后续调试。

---

## 14. 项目总结

本项目实现了一个从基础大模型事实判断到检索增强、多步推理、自动证据选择、CrossEncoder 精排、Gold Evidence 上限对照和 HoVer 多跳验证的完整事实验证实验系统。

该系统能够支持课程作业中的核心要求，包括：

* 至少 50 条数据测试
* Baseline 实验
* RAG 检索增强实验
* CoT 推理实验
* Accuracy、Precision、Recall、F1、幻觉率计算
* 不同方法性能对比
* HoVer 多跳扩展
* BM25、Sentence-BERT、CrossEncoder 自动证据选择扩展

项目结构清晰，实验模式可配置，结果可复现，便于继续扩展和撰写实验报告。
