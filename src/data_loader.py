"""
FEVER事实验证系统 - 数据加载模块
"""
import os
import json
import random
from pathlib import Path
from src.config import FEVER_SPLIT, SAMPLE_SIZE, RANDOM_SEED, VALID_LABELS

# 手动下载的原始 FEVER 验证集路径（从 fever.ai 官网下载的 shared_task_dev.jsonl）
RAW_JSONL_FILE = "data/shared_task_dev.jsonl"


def load_hover_data(sample_size=50, seed=42):
    """读取手动下载的 HoVer 官方 Dev 集文件"""

    file_path = "data/cache/hover_dev.json"

    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        print("请确认你已经下载了 Dev set 并重命名放在了 data/cache/ 目录下。")
        return []

    print(f"正在读取手动下载的 HoVer 数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data_list = []
    for item in raw_data:
        original_label = item.get('label', '')
        label = "SUPPORTS" if original_label == "SUPPORTED" else "REFUTES"
        data_list.append({
            'id': f"hover_{item.get('uid', 'unknown')}",
            'claim': item.get('claim', ''),
            'label': label
        })

    random.seed(seed)
    sampled_data = random.sample(data_list, min(sample_size, len(data_list)))
    print(f"成功加载 {len(sampled_data)} 条 HoVer 真实数据！")
    return sampled_data


def load_fever_data(split=FEVER_SPLIT, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    加载FEVER数据集并随机采样。

    优先读取本地缓存文件（data/fever_{split}.json）；
    缓存不存在时，从手动下载的 shared_task_dev.jsonl 解析并生成缓存。

    每条数据包含四个字段：
      - id:             claim 编号
      - claim:          待验证的陈述文本
      - label:          真实标签（SUPPORTS / REFUTES / NOT ENOUGH INFO）
      - evidence_pages: 该 claim 对应的 Wikipedia 页面名列表（去重，下划线格式），
                        与 FEVER wiki dump 的 id 字段直接对应。
                        NOT ENOUGH INFO 的 claim 没有标注证据，列表为空。

    Args:
        split:       数据集分割标识（目前只用 labelled_dev）
        sample_size: 随机采样数量
        seed:        随机种子，固定后每次采样结果完全一致

    Returns:
        list[dict]: 采样后的数据列表
    """
    cache_file = f"data/fever_{split}.json"

    if os.path.exists(cache_file):
        # ------------------------------------------------------------------
        # 读本地缓存，直接加载，跳过解析步骤
        # ------------------------------------------------------------------
        print(f"从本地加载FEVER数据集: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        print(f"数据集总数: {len(data_list)}条")

    else:
        # ------------------------------------------------------------------
        # 缓存不存在，从 shared_task_dev.jsonl 解析
        #
        # 原始格式（每行一个 JSON）：
        #   {
        #     "id": 91198,
        #     "verifiable": "NOT VERIFIABLE",
        #     "label": "NOT ENOUGH INFO",
        #     "claim": "...",
        #     "evidence": [
        #       [                        <- annotation 组（可有多组）
        #         [annotation_id, evidence_id, wikipedia_page, sentence_id],
        #         ...
        #       ],
        #       ...
        #     ]
        #   }
        #
        # wikipedia_page 是下划线格式（如 "Nikolaj_Coster-Waldau"），
        # 与 FEVER wiki dump 的 id 字段完全对应，无需转换。
        # NOT ENOUGH INFO 的 evidence 里 wikipedia_page 为 null，过滤掉即可。
        # ------------------------------------------------------------------
        if not os.path.exists(RAW_JSONL_FILE):
            raise FileNotFoundError(
                f"找不到原始数据文件：{RAW_JSONL_FILE}\n"
                "请从 fever.ai 官网下载 shared_task_dev.jsonl 并放到 data/ 目录下。"
            )

        print(f"从原始文件解析FEVER数据集: {RAW_JSONL_FILE}")
        data_list = []
        skipped = 0

        with open(RAW_JSONL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                # 从 evidence 嵌套列表中提取所有 wikipedia_page，去重
                pages = set()
                for annotation_group in item.get('evidence', []):
                    for evidence_item in annotation_group:
                        # evidence_item 结构：
                        #   [annotation_id, evidence_id, wikipedia_page, sentence_id]
                        # wikipedia_page 为 None 时是 NOT ENOUGH INFO，跳过
                        if len(evidence_item) >= 3 and evidence_item[2] is not None:
                            pages.add(evidence_item[2])

                data_list.append({
                    'id':             item['id'],
                    'claim':          item['claim'],
                    'label':          item['label'],
                    'evidence_pages': sorted(pages),  # 排序保证每次输出一致
                })

        print(f"解析完成: {len(data_list)}条，跳过{skipped}条格式错误行")

        # 验证并过滤非法数据
        data_list = validate_data(data_list)

        # 保存缓存，下次直接读，不用重新解析
        print(f"保存数据集缓存到: {cache_file}")
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print("保存完成！")

    # 随机采样
    random.seed(seed)
    if sample_size < len(data_list):
        sampled_data = random.sample(data_list, sample_size)
        print(f"随机采样: {sample_size}条 (随机种子: {seed})")
    else:
        sampled_data = data_list
        print(f"使用全部数据: {len(sampled_data)}条")

    return sampled_data


def validate_data(data_list):
    """
    验证数据格式，过滤掉字段缺失或标签非法的条目。

    Args:
        data_list: 原始数据列表

    Returns:
        list: 验证后的数据列表
    """
    valid_data = []
    for item in data_list:
        if 'claim' not in item or 'label' not in item:
            continue
        if not item['claim'] or not isinstance(item['claim'], str):
            continue
        if item['label'] not in VALID_LABELS:
            continue
        # evidence_pages 字段若缺失，补一个空列表（兼容旧缓存文件）
        if 'evidence_pages' not in item:
            item['evidence_pages'] = []
        valid_data.append(item)

    print(f"数据验证完成: {len(valid_data)}条有效数据")
    return valid_data


def get_label_distribution(data_list):
    """
    统计标签分布。

    Args:
        data_list: 数据列表

    Returns:
        dict: {label: count}
    """
    distribution = {}
    for item in data_list:
        label = item['label']
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def collect_evidence_pages(data_list):
    """
    从数据列表中收集所有出现过的 Wikipedia 页面名（去重）。

    用于 build_bm25_index_filtered() 确定需要加载哪些文章。
    直接从已加载的数据读取，不需要重新解析原始数据集。

    Args:
        data_list: load_fever_data() 返回的数据列表

    Returns:
        set[str]: 页面名集合（下划线格式，与 dump 的 id 字段直接对应）
    """
    pages = set()
    for item in data_list:
        for page in item.get('evidence_pages', []):
            if page:
                pages.add(page)
    return pages