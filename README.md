# FEVER / HoVer 大模型事实验证系统

本项目是一个基于大语言模型的事实验证系统，主要面向 **FEVER** 三分类事实校验任务，并扩展支持 **HoVer** 多跳事实验证任务。

系统输入一条自然语言声明 Claim，输出对应的事实验证标签：

* `SUPPORTS`：声明被证据支持
* `REFUTES`：声明被证据反驳
* `NOT ENOUGH INFO`：证据信息不足，无法判断

本项目实现了从基础大模型判断到检索增强、思维链推理、BM25 + Sentence-BERT 自动证据选择、CrossEncoder 精排、Gold Evidence 上限对照以及 HoVer 多跳验证的完整实验流程。

---

## 目录

* [项目简介](#项目简介)
* [功能特点](#功能特点)
* [项目结构](#项目结构)
* [环境配置](#环境配置)
* [API Key 配置](#api-key-配置)
* [模型下载](#模型下载)
* [数据准备](#数据准备)
* [核心配置说明](#核心配置说明)
* [实验模式说明](#实验模式说明)
* [运行方法](#运行方法)
* [推荐实验顺序](#推荐实验顺序)
* [输出结果](#输出结果)
* [评价指标](#评价指标)
* [结果绘图建议](#结果绘图建议)
* [常见问题](#常见问题)
* [GitHub 提交说明](#github-提交说明)
* [项目总结](#项目总结)

---

## 项目简介

大语言模型虽然具备较强的自然语言理解与生成能力，但在事实性任务中仍可能出现幻觉，即模型生成看似合理但实际错误的内容。因此，事实验证是评估和提升大语言模型可靠性的重要任务。

本项目围绕 FEVER 数据集构建事实验证系统，并设计多个递进式实验模式，用于比较不同方法对事实校验性能的影响。

示例：

```text
Claim: The Eiffel Tower is located in Paris.
Label: SUPPORTS

Claim: The Eiffel Tower is located in Berlin.
Label: REFUTES

Claim: The Eiffel Tower is the tallest structure in Europe.
Label: NOT ENOUGH INFO
```

本项目主要支持以下实验路线：

**FEVER 实验路线：**

```text
Baseline
CoT
RAG
RAG + CoT
BM25 + Sentence-BERT
BM25 + Sentence-BERT + CrossEncoder
Gold Evidence
Gold Evidence + CoT
```

**HoVer 扩展实验路线（EXTENDED_PIPELINE）：**

```text
单跳检索（USE_MULTI_HOP=False）
IRCoT 多跳检索（USE_MULTI_HOP=True）
IRCoT + Cross-Encoder 精排（USE_MULTI_HOP=True, USE_CROSS_ENCODER=True）  ← 当前最佳
IRCoT + Cross-Encoder + LLM Judge 二次核验（USE_LLM_JUDGE=True）
可选检索模式：BM25（推荐）/ Dense / Hybrid（通过 RETRIEVAL_MODE 切换）
```

---

## 功能特点

本项目支持：

* 基于大模型内置知识的直接事实判断
* 基于 Chain-of-Thought 的逐步推理事实判断
* 基于 Wikipedia 证据的 RAG 事实验证
* 基于本地 Wikipedia dump 的 BM25 文档检索
* 基于 Sentence-BERT 的证据句语义排序
* 基于 CrossEncoder 的证据精排
* 基于 FEVER Gold Evidence 的上限对照实验
* 基于 HoVer 的多跳事实验证扩展
* IRCoT 风格的逐跳检索流程
* LLM-as-a-Judge 二次核验
* 多轮实验与均值、标准差统计
* Accuracy、Precision、Recall、F1、Hallucination Rate 指标计算
* 混淆矩阵输出
* 日志保存
* 详细实验结果 JSON 保存
* 模型响应解析错误记录
* 本地模型优先加载支持

---

## 项目结构

```text
fact_verification/
├── main.py                         # 主程序入口
├── app_flash.py                    # 可选 Web 展示入口
├── download_models.py              # 本地模型下载脚本
├── README.md                       # 项目说明文档
├── METRICS.md                      # 指标说明文档
├── requirements.txt                # 项目依赖
├── .env.example                    # API Key 配置模板
├── .gitignore                      # Git 忽略规则
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # 全局配置文件
│   ├── api_client.py               # DeepSeek API 调用模块
│   ├── data_loader.py              # FEVER / HoVer 数据加载模块
│   ├── golden_loader.py            # Gold Evidence 解析模块
│   ├── prompt_builder.py           # Prompt 构造与模型响应解析
│   ├── retriever.py                # BM25 / SBERT / CrossEncoder 检索模块
│   ├── verifier.py                 # 事实验证核心逻辑
│   ├── evaluator.py                # 评估指标计算与报告生成
│   └── utils.py                    # 日志、JSON、时间格式化工具
│
├── templates/
│   └── index.html                  # Web 页面模板
│
├── data/                           # 本地数据目录，不上传 GitHub
│   ├── shared_task_dev.jsonl       # FEVER dev 数据
│   ├── wiki-pages/                 # FEVER Wikipedia dump
│   ├── cache/                      # 缓存文件与索引
│   └── results/                    # 实验结果输出目录
│
├── local_models/                   # 本地模型目录，不上传 GitHub
│   ├── all-MiniLM-L6-v2/
│   └── cross-encoder/
│
└── logs/                           # 日志目录，不上传 GitHub
```

---

## 环境配置

### 1. 创建虚拟环境

推荐使用 Python 3.10 或以上版本。

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\activate
```

如果使用 Conda：

```powershell
conda create -n factcheck python=3.10
conda activate factcheck
```

---

### 2. 安装依赖

在项目根目录运行：

```powershell
pip install -r requirements.txt
```

主要依赖包括：

```text
openai
python-dotenv
scikit-learn
tqdm
nltk
torch
sentence-transformers
rank-bm25
huggingface-hub
```

如果需要运行 BM25、Sentence-BERT 或 CrossEncoder 相关模式，请确保安装：

```powershell
pip install sentence-transformers
```

---

## API Key 配置

本项目使用 DeepSeek API，采用 OpenAI 兼容接口调用。

首先复制环境变量模板：

```powershell
copy .env.example .env
```

然后在 `.env` 文件中填写 API Key：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

`src/config.py` 会自动读取 `.env`：

```python
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
```

注意：

```text
.env 文件包含私密 API Key，不要上传到 GitHub。
```

---

## 模型下载

模型文件体积较大，不上传到 GitHub，需要在本地下载。

运行：

```powershell
python download_models.py
```

下载完成后，目录结构应类似：

```text
local_models/
├── all-MiniLM-L6-v2/
└── cross-encoder/
    └── ms-marco-MiniLM-L-6-v2/
```

其中：

| 模型                       | 作用                         |
| ------------------------ | -------------------------- |
| `all-MiniLM-L6-v2`       | Sentence-BERT 模型，用于证据句语义排序 |
| `ms-marco-MiniLM-L-6-v2` | CrossEncoder 模型，用于证据精排     |

如果你的下载脚本仍然叫 `load.py`，可以改名为：

```powershell
Rename-Item load.py download_models.py
```

然后再提交到 GitHub。

---

## 数据准备

### 1. FEVER 数据

请将 FEVER 官方 dev 数据文件放到：

```text
data/shared_task_dev.jsonl
```

首次运行时，系统会解析该文件，并生成缓存文件：

```text
data/fever_labelled_dev.json
```

后续运行会优先读取缓存，以提升速度。

---

### 2. Wikipedia Dump

请将 FEVER Wikipedia dump 解压到：

```text
data/wiki-pages/
```

目录中应包含：

```text
wiki-001.jsonl
wiki-002.jsonl
wiki-003.jsonl
...
```

该 dump 用于：

* BM25 索引构建
* Sentence-BERT 句子筛选
* Gold Evidence 原句反查
* HoVer 检索

---

### 3. HoVer 数据

如果需要运行 HoVer 扩展实验，请将 HoVer dev 数据放到：

```text
data/cache/hover_dev.json
```

系统会读取其中的：

* `claim`
* `label`
* `supporting_facts`

并将 HoVer 标签统一映射为本项目标签：

```text
SUPPORTED -> SUPPORTS
REFUTED   -> REFUTES
```

---

## 核心配置说明

主要配置文件位于：

```text
src/config.py
```

---

### 1. API 配置

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

参数说明：

| 参数               | 说明               |
| ---------------- | ---------------- |
| `MODEL_NAME`     | 调用的大模型名称         |
| `TEMPERATURE`    | 模型输出随机性，事实验证建议较低 |
| `MAX_TOKENS`     | 单次输出最大 token 数   |
| `TIMEOUT`        | API 超时时间         |
| `MAX_RETRIES`    | API 调用失败最大重试次数   |
| `RETRY_DELAY`    | 初始重试等待时间         |
| `BACKOFF_FACTOR` | 指数退避系数           |
| `REQUEST_DELAY`  | 每条样本之间的请求间隔      |

如果出现 `Connection error`，建议增大：

```python
REQUEST_DELAY = 1.0
TIMEOUT = 90
RETRY_DELAY = 5
```

---

### 2. 实验配置

```python
EXPERIMENT_MODE = "RAG_GOLDEN"

SAMPLE_SIZE = 50
RANDOM_SEED = 42
NUM_ROUNDS = 1

VALID_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
```

参数说明：

| 参数                | 说明             |
| ----------------- | -------------- |
| `EXPERIMENT_MODE` | 当前实验模式         |
| `SAMPLE_SIZE`     | 每次采样数量         |
| `RANDOM_SEED`     | 随机种子，用于保证采样可复现 |
| `NUM_ROUNDS`      | 实验轮数           |
| `VALID_LABELS`    | 合法标签空间         |

---

### 3. 检索参数

```python
BM25_TOP_N_DOCS = 5
SBERT_TOP_K_SENTENCES = 4
MAX_SENTENCES_PER_DOC = 5
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
```

参数说明：

| 参数                      | 说明                 |
| ----------------------- | ------------------ |
| `BM25_TOP_N_DOCS`       | BM25 第一阶段召回的候选文档数  |
| `SBERT_TOP_K_SENTENCES` | 最终提供给大模型的证据句数量     |
| `MAX_SENTENCES_PER_DOC` | 每篇候选文档最多进入候选池的句子数  |
| `SBERT_MODEL_NAME`      | Sentence-BERT 模型名称 |

---

### 4. CrossEncoder 配置

```python
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_CROSS_ENCODER = False
CE_RERANK_TOP_N = 20
```

CrossEncoder 用于在 Sentence-BERT 粗排后进一步精排证据句。

普通 BM25 检索模式：

```python
EXPERIMENT_MODE = "RAG_BM25"
```

CrossEncoder 精排模式：

```python
EXPERIMENT_MODE = "RAG_BM25_CE"
```

---

### 5. Gold Evidence 配置

```python
GOLDEN_FEVER_FILE = os.path.join(BASE_DIR, "data", "shared_task_dev.jsonl")
GOLDEN_EVIDENCE_CACHE = os.path.join(DATA_CACHE_DIR, "golden_evidence_cache.json")
REBUILD_GOLDEN_EVIDENCE_CACHE = False
```

如果修改了 Gold Evidence 解析逻辑，可以临时设置：

```python
REBUILD_GOLDEN_EVIDENCE_CACHE = True
```

缓存重建完成后改回：

```python
REBUILD_GOLDEN_EVIDENCE_CACHE = False
```

---

### 6. HoVer / IRCoT 配置

以下参数仅对 `EXPERIMENT_MODE = "EXTENDED_PIPELINE"` 生效。

```python
HOVER_INDEX_DIR = os.path.join(DATA_CACHE_DIR, "hover_bm25_index")

USE_MULTI_HOP = True       # 是否启用 IRCoT 多跳检索
MAX_HOP_ROUNDS = 3         # 最大跳数上限
USE_CROSS_ENCODER = True   # 是否在多跳检索后启用 Cross-Encoder 精排
USE_LLM_JUDGE = False      # 是否启用 LLM Judge 二次核验（净负效果，建议保持 False）
RETRIEVAL_MODE = "BM25"    # 第一阶段文档检索模式
```

参数说明：

| 参数                | 说明                                              |
| ----------------- | ------------------------------------------------ |
| `HOVER_INDEX_DIR` | HoVer BM25 索引目录                                 |
| `USE_MULTI_HOP`   | 是否启用 IRCoT 多跳检索；False 为单跳模式                   |
| `MAX_HOP_ROUNDS`  | 最大检索跳数，推荐 3；LLM 判断证据充足时可提前退出                |
| `USE_CROSS_ENCODER` | 是否在 SBERT 粗排后启用 Cross-Encoder 精排；HoVer 专用开关，推荐 True |
| `USE_LLM_JUDGE`   | 是否在基础核验后增加一轮 LLM 二次审核；实验显示准确率下降 4%，建议 False |
| `RETRIEVAL_MODE`  | 第一阶段文档检索模式，可选值见下表                             |

**RETRIEVAL_MODE 可选值：**

| 值          | 说明                                                        |
| ---------- | ----------------------------------------------------------- |
| `"BM25"`   | BM25 关键词匹配（推荐）；对命名实体查询精度高，与 IRCoT 搭配效果最佳  |
| `"DENSE"`  | FAISS 稠密向量检索；理论上能处理词汇缺口，但对专有名词区分能力弱于 BM25   |
| `"HYBRID"` | BM25 与 Dense 取并集后重排；实验中效果与 BM25 相当，噪声略多         |

注意：`USE_CROSS_ENCODER` 在此处（HoVer 配置块）的赋值会覆盖上方 CrossEncoder 精排配置中的同名变量，两者共用同一开关。`EXTENDED_PIPELINE` 模式下，建议显式设为 `True` 以开启精排。

---

## 实验模式说明

| 模式                  | 说明                                     |
| ------------------- | -------------------------------------- |
| `BASELINE`          | 无外部证据，直接让大模型判断                         |
| `COT`               | 无外部证据，要求大模型逐步推理                        |
| `RAG`               | 抽取核心实体，检索 Wikipedia 词条前几句作为证据          |
| `RAG_COT`           | 检索证据 + Chain-of-Thought 推理             |
| `RAG_BM25`          | BM25 文档召回 + Sentence-BERT 句子排序         |
| `RAG_BM25_CE`       | BM25 + Sentence-BERT + CrossEncoder 精排（verifier.py 自动开启 CE） |
| `RAG_GOLDEN`        | 使用 FEVER 人工标注 Gold Evidence            |
| `RAG_COT_GOLDEN`    | Gold Evidence + Chain-of-Thought 推理    |
| `EXTENDED_PIPELINE` | HoVer 多跳事实验证扩展，行为由下列开关控制：             |

`EXTENDED_PIPELINE` 子变体（通过 `config.py` 中的开关组合切换）：

| 子变体                          | `USE_MULTI_HOP` | `USE_CROSS_ENCODER` | `USE_LLM_JUDGE` | `RETRIEVAL_MODE` |
| ----------------------------- | --------------- | ------------------- | --------------- | ---------------- |
| 单跳检索                         | `False`         | `False`             | `False`         | `"BM25"`         |
| IRCoT 多跳                      | `True`          | `False`             | `False`         | `"BM25"`         |
| IRCoT + CE 精排（当前最佳）          | `True`          | `True`              | `False`         | `"BM25"`         |
| IRCoT + CE + LLM Judge 二次核验   | `True`          | `True`              | `True`          | `"BM25"`         |
| IRCoT + CE（Dense 检索，供消融对比）   | `True`          | `True`              | `False`         | `"DENSE"`        |
| IRCoT + CE（Hybrid 检索，供消融对比）  | `True`          | `True`              | `False`         | `"HYBRID"`       |

---

## 运行方法

在项目根目录运行：

```powershell
python main.py
```

程序会自动：

1. 根据 `EXPERIMENT_MODE` 加载对应数据
2. 检查所需索引是否存在
3. 必要时构建 BM25 索引
4. 初始化 FactVerifier
5. 调用大模型进行事实验证
6. 解析模型输出标签
7. 计算评估指标
8. 保存结果和日志

---

## 推荐实验顺序

### 1. 小样本调试

建议先使用小样本确认流程是否跑通：

```python
SAMPLE_SIZE = 5
NUM_ROUNDS = 1
REQUEST_DELAY = 1.0
```

推荐依次测试：

```python
EXPERIMENT_MODE = "BASELINE"
EXPERIMENT_MODE = "RAG_GOLDEN"
EXPERIMENT_MODE = "RAG_BM25"
EXPERIMENT_MODE = "RAG_BM25_CE"
```

---

### 2. FEVER 正式对比实验

推荐公共参数：

```python
SAMPLE_SIZE = 50
RANDOM_SEED = 42
NUM_ROUNDS = 1
TEMPERATURE = 0.1
REQUEST_DELAY = 1.0
```

依次运行：

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

---

### 3. HoVer 扩展实验

单跳检索版本：

```python
EXPERIMENT_MODE = "EXTENDED_PIPELINE"
USE_MULTI_HOP = False
USE_CROSS_ENCODER = False
USE_LLM_JUDGE = False
RETRIEVAL_MODE = "BM25"
```

IRCoT 多跳版本：

```python
EXPERIMENT_MODE = "EXTENDED_PIPELINE"
USE_MULTI_HOP = True
MAX_HOP_ROUNDS = 3
USE_CROSS_ENCODER = False
USE_LLM_JUDGE = False
RETRIEVAL_MODE = "BM25"
```

IRCoT + Cross-Encoder 精排版本（当前最佳，Accuracy 66%）：

```python
EXPERIMENT_MODE = "EXTENDED_PIPELINE"
USE_MULTI_HOP = True
MAX_HOP_ROUNDS = 3
USE_CROSS_ENCODER = True   # 在多跳结果上启用 CE 精排
USE_LLM_JUDGE = False
RETRIEVAL_MODE = "BM25"
```

IRCoT + CE + LLM Judge 二次核验版本（实验性，准确率低于不加 Judge，仅供消融）：

```python
EXPERIMENT_MODE = "EXTENDED_PIPELINE"
USE_MULTI_HOP = True
MAX_HOP_ROUNDS = 3
USE_CROSS_ENCODER = True
USE_LLM_JUDGE = True       # 额外增加一轮 LLM 审核；实验结果净负收益
RETRIEVAL_MODE = "BM25"
```

检索模式消融（固定 IRCoT + CE，仅切换 RETRIEVAL_MODE）：

```python
RETRIEVAL_MODE = "BM25"    # 推荐，命名实体查询精度最高
RETRIEVAL_MODE = "DENSE"   # FAISS 稠密检索，专有名词精度弱于 BM25
RETRIEVAL_MODE = "HYBRID"  # 并集后重排，效果与 BM25 相当
```

---

## 输出结果

实验结果保存在：

```text
data/results/
```

日志文件保存在：

```text
logs/
```

结果文件示例：

```text
data/results/verification_results_rag_bm25_20260604_225100.json
logs/verification_rag_bm25_20260604_225100.log
```

结果 JSON 中包含：

```json
{
  "experiment_mode": "RAG_BM25",
  "timestamp": "20260604_225100",
  "num_rounds": 1,
  "valid_rounds": 1,
  "random_seed": 42,
  "sample_size": 50,
  "aggregated_metrics": {},
  "rounds": []
}
```

每条样本详细结果包括：

* `id`
* `claim`
* `true_label`
* `predicted_label`
* `evidence`
* `correct`
* `llm_raw`
* `trace`

其中 `trace` 用于记录检索过程、使用的证据、是否启用 CrossEncoder、HoVer 多跳搜索过程等信息。

---

## 评价指标

系统支持以下评价指标：

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

该指标用于衡量模型在证据不足时是否倾向于强行给出确定判断。

---

## 结果绘图建议

实验完成后，可以读取 `data/results/` 下的 JSON 文件绘图。

推荐绘制：

1. FEVER 各模式 Accuracy 对比图
2. FEVER 各模式 Macro F1 对比图
3. FEVER 各模式 Hallucination Rate 对比图
4. HoVer 单跳、多跳、多跳 + Judge 消融对比图

FEVER 图表横轴建议：

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

HoVer 图表横轴建议：

```text
Single Retrieval
IRCoT
IRCoT + CE
IRCoT + CE + Judge
```

---

## 常见问题

### 1. API Connection error

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

如果仍然不稳定，可以先减小样本量：

```python
SAMPLE_SIZE = 5
```

---

### 2. 找不到 API Key

检查 `.env` 文件是否存在，并确认其中包含：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

---

### 3. 找不到 Wikipedia dump

确认目录：

```text
data/wiki-pages/
```

下存在：

```text
wiki-*.jsonl
```

---

### 4. 找不到 HoVer 数据

确认文件存在：

```text
data/cache/hover_dev.json
```

---

### 5. CrossEncoder 第一次运行很慢

`RAG_BM25_CE` 第一次运行时需要加载或下载 CrossEncoder 模型。

建议先使用：

```python
SAMPLE_SIZE = 5
```

确认模型加载成功后，再扩大样本量。

---

### 6. GitHub 上没有 data 和 local_models

这是正常的。

本项目的 `.gitignore` 会忽略：

```text
data/
local_models/
logs/
.env
.venv/
```

这些内容需要用户在本地自行准备。

---

## GitHub 提交说明

建议上传到 GitHub 的内容：

```text
src/
main.py
app_flash.py
templates/
README.md
METRICS.md
requirements.txt
.env.example
download_models.py
.gitignore
```

不要上传：

```text
.env
.venv/
data/
data.zip
src.zip
local_models/
logs/
*.pkl
*.bin
*.safetensors
*.h5
*.ot
*.onnx
```

原因：

* `.env` 包含 API Key
* `data/` 包含数据集、缓存、索引
* `local_models/` 包含模型权重
* `logs/` 是运行日志
* `.pkl`、`.bin`、`.safetensors` 等文件通常体积较大

---

## 项目总结

本项目实现了一个完整的大模型事实验证实验系统，覆盖了课程作业所需的核心任务：

* Baseline 直接事实判断
* RAG 检索增强事实校验
* CoT 推理事实校验
* Accuracy、Precision、Recall、F1、幻觉率评估
* 不同方法之间的性能对比
* BM25、Sentence-BERT、CrossEncoder 自动证据选择
* Gold Evidence 上限对照
* HoVer 多跳事实验证扩展

该系统采用模块化设计，配置灵活，便于进行消融实验、结果复现和后续扩展。
