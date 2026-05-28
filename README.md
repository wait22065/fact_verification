# FEVER事实验证系统

基于FEVER数据集的事实验证系统，用于评估大模型在事实核查任务上的表现。

## 项目简介

本项目实现了两个任务：

- **任务一（当前）：Baseline** - 不使用证据，只基于claim让模型判断事实正确性
- **任务二（待实现）：RAG增强** - 使用检索增强，提供evidence给模型

## 功能特点

- 使用FEVER公开数据集（labelled_dev验证集，37566条）
- 调用DeepSeek API进行事实验证
- 支持随机采样和可复现实验
- 首次运行自动下载并缓存数据集到本地，后续直接读取
- 完整的评估指标（accuracy、precision、recall、F1、幻觉率）
- 自动记录无法解析的模型响应，方便调整prompt
- 详细的日志记录和结果保存

## 安装

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd fact_verification
```

### 2. 创建虚拟环境（推荐使用uv）

```bash
# 安装uv（如果还没安装）
pip install uv

# 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate
```

### 3. 安装依赖

```bash
# 使用uv安装（速度更快）
uv pip install -r requirements.txt

# 或使用pip
pip install -r requirements.txt
```

依赖包：

- `openai` - DeepSeek API调用
- `python-dotenv` - 环境变量管理
- `scikit-learn` - 评估指标计算
- `tqdm` - 进度条显示
- `datasets` - Hugging Face数据集加载

### 4. 配置API密钥

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的DeepSeek API密钥：

```
DEEPSEEK_API_KEY=your_api_key_here
```

## 使用方法

### 运行验证

```bash
python main.py
```

程序会自动：

1. **首次运行**：从Hugging Face下载FEVER验证集并保存到 `data/fever_labelled_dev.json`
2. **后续运行**：直接从本地JSON文件加载（速度快）
3. 随机采样50条数据（可在`src/config.py`中配置）
4. 调用DeepSeek API进行验证
5. 计算评估指标
6. 保存结果到`data/results/verification_results.json`
7. 如有解析错误，保存到`data/results/parse_errors.json`

### 配置参数

编辑 `src/config.py` 可修改：

```python
SAMPLE_SIZE = 50        # 采样数量
RANDOM_SEED = 42        # 随机种子（固定后每次采样结果相同）
TEMPERATURE = 0.1       # 模型温度
MAX_RETRIES = 3         # API重试次数
```

## 输出结果

### 控制台输出

```
============================================================
FEVER事实验证系统 - 任务一：Baseline（无证据）
============================================================

2026-04-17 15:05:54 - FEVER_Verification - INFO - 步骤1: 加载FEVER数据集

首次运行，从Hugging Face下载FEVER数据集 (labelled_dev)...
数据集总数: 37566条
数据验证完成: 37566条有效数据
保存数据集到本地: data/fever_labelled_dev.json
保存完成！
随机采样: 50条 (随机种子: 42)

标签分布:
  REFUTES: 13条
  SUPPORTS: 23条
  NOT ENOUGH INFO: 14条

2026-04-17 15:05:59 - FEVER_Verification - INFO - 步骤2: 执行事实验证
验证进度: 100%|██████████| 50/50 [00:45<00:00,  1.11it/s]

==================================================
评估结果
==================================================

Accuracy (准确率): 0.8200
Macro Precision (宏平均精确率): 0.7900
Macro Recall (宏平均召回率): 0.7800
Macro F1-Score (宏平均F1): 0.7850
Hallucination Rate (幻觉率): 0.1500

==================================================
各类别详细指标
==================================================
类别                 Precision    Recall       F1-Score     Support
----------------------------------------------------------------------
SUPPORTS             0.8500       0.8800       0.8650       23
REFUTES              0.8000       0.7500       0.7750       13
NOT ENOUGH INFO      0.7300       0.7300       0.7300       14

已保存 2 条解析错误到: data/results/parse_errors.json
结果已保存到: data/results/verification_results.json

============================================================
任务完成！
总耗时: 45.3秒
结果文件: data/results/verification_results.json
============================================================
```

### 结果文件

**`data/results/verification_results.json`** - 完整验证结果：

```json
{
  "metrics": {
    "accuracy": 0.82,
    "macro_precision": 0.79,
    "macro_recall": 0.78,
    "macro_f1": 0.785,
    "hallucination_rate": 0.15,
    "per_class_metrics": {...},
    "confusion_matrix": [...]
  },
  "detailed_results": [
    {
      "id": 123,
      "claim": "...",
      "true_label": "SUPPORTS",
      "predicted_label": "SUPPORTS",
      "correct": true
    },
    ...
  ],
  "summary": {
    "total_samples": 50,
    "successful_predictions": 50,
    "failed_predictions": 0
  }
}
```

**`data/results/parse_errors.json`** - 解析错误记录（如果有）：

```json
[
  {
    "timestamp": "2026-04-17T15:06:23.123456",
    "claim_id": 12345,
    "claim": "某个声明...",
    "response": "模型的原始响应...",
    "reason": "无法匹配任何标签"
  }
]
```

## 项目结构

```
fact_verification/
├── .venv/                      # 虚拟环境（不提交）
├── .git/                       # Git仓库
├── src/                        # 源代码
│   ├── __init__.py
│   ├── config.py               # 配置管理
│   ├── data_loader.py          # 数据加载（支持本地缓存）
│   ├── api_client.py           # API调用（含重试机制）
│   ├── prompt_builder.py       # Prompt构造（含错误记录）
│   ├── verifier.py             # 验证逻辑
│   ├── evaluator.py            # 评估指标计算
│   └── utils.py                # 工具函数
├── data/
│   ├── fever_labelled_dev.json # FEVER数据集本地缓存（不提交）
│   └── results/                # 结果输出（不提交）
│       ├── verification_results.json
│       └── parse_errors.json
├── logs/
│   └── verification.log        # 运行日志
├── main.py                     # 主程序入口
├── requirements.txt            # 依赖包
├── .env                        # 环境变量（不提交）
├── .env.example                # 环境变量模板
├── .gitignore                  # Git忽略规则
├── README.md                   # 项目文档
└── METRICS.md                  # 评估指标详细说明
```

## 评估指标说明

详见 [METRICS.md](METRICS.md)

- **Accuracy（准确率）** - 所有预测中正确的比例
- **Precision（精确率）** - 预测为某类别中真正属于该类别的比例
- **Recall（召回率）** - 真实为某类别中被正确预测的比例
- **F1-Score** - Precision和Recall的调和平均
- **Hallucination Rate（幻觉率）** - 真实为NOT ENOUGH INFO但预测为SUPPORTS/REFUTES的比例

## 注意事项

1. **API密钥安全** - 不要将`.env`文件提交到Git
2. **速率限制** - 程序已添加请求延迟（0.5秒），避免触发API限制
3. **数据缓存** - 首次运行会下载约40MB数据，保存到本地后续直接读取
4. **随机种子** - 固定随机种子后，每次采样结果完全一致，保证可复现
5. **任务说明** - 任务一不使用evidence，测试模型的内在知识
6. **错误记录** - 无法解析的响应会自动记录，方便调整prompt

## 任务对比

| 特性       | 任务一（Baseline） | 任务二（RAG）    |
| ---------- | ------------------ | ---------------- |
| 输入       | 只有claim          | claim + evidence |
| 目的       | 测试模型内在知识   | 测试检索增强效果 |
| 预期准确率 | 较低               | 较高             |
| 幻觉率     | 可能较高           | 应该降低         |

## 常见问题

### Q: 如何修改采样数量？

编辑 `src/config.py`，修改 `SAMPLE_SIZE` 参数。

### Q: 如何使用不同的模型？

编辑 `src/config.py`，修改 `MODEL_NAME` 参数。

### Q: 如何重新下载数据集？

删除 `data/fever_labelled_dev.json` 文件，再次运行程序即可。

### Q: API调用失败怎么办？

检查：

1. API密钥是否正确（查看`.env`文件）
2. 网络连接是否正常
3. 查看 `logs/verification.log` 了解详细错误

### Q: 如何查看无法解析的响应？

查看 `data/results/parse_errors.json` 文件，根据实际响应调整 `src/prompt_builder.py` 中的解析逻辑。

### Q: 如何推送到GitHub？

```bash
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

## 许可证

MIT License

## 致谢

- [FEVER数据集](https://fever.ai/)
- [Hugging Face Datasets](https://huggingface.co/datasets/fever/fever)
- [DeepSeek API](https://www.deepseek.com/)
- [uv - 快速Python包管理器](https://github.com/astral-sh/uv)
