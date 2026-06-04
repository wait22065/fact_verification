"""
FEVER事实验证系统 - 数据加载模块 (路径鲁棒性优化版)
"""
import os
import json
import random
from pathlib import Path
from src.config import BASE_DIR, FEVER_SPLIT, SAMPLE_SIZE, RANDOM_SEED, VALID_LABELS

# 使用 BASE_DIR 动态组装绝对路径，防止执行环境路径变化引发文件缺失报错
RAW_JSONL_FILE = os.path.join(BASE_DIR, "data", "shared_task_dev.jsonl")


def load_hover_data(sample_size=50, seed=42):
    """
    读取 HoVer 官方 Dev 集文件，并进行二分类均衡采样。

    HoVer 原始标签可能是：
      SUPPORTED / REFUTED
      或 SUPPORTS / REFUTES
      或 NOT_SUPPORTED

    统一映射为：
      SUPPORTS / REFUTES
    """
    file_path = os.path.join(BASE_DIR, "data", "cache", "hover_dev.json")

    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        print("请确认你已经下载 HoVer Dev set，并重命名为 hover_dev.json 放到 data/cache/ 目录下。")
        return []

    print(f"正在读取 HoVer 数据: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data_list = []
    raw_label_counter = {}

    for item in raw_data:
        original_label = str(item.get("label", "")).strip()
        label_upper = original_label.upper()

        raw_label_counter[label_upper] = raw_label_counter.get(label_upper, 0) + 1

        # 统一标签映射
        if label_upper in ["SUPPORTED", "SUPPORTS"]:
            label = "SUPPORTS"
        elif label_upper in ["REFUTED", "REFUTES", "NOT_SUPPORTED", "NOT SUPPORTS", "NOT-SUPPORTED"]:
            label = "REFUTES"
        else:
            # 遇到未知标签，跳过，避免污染评估
            continue

        # 提取 supporting_facts 页面名，空格转下划线，与 wiki dump id 对齐
        seen, pages = set(), []

        for fact in item.get("supporting_facts", []):
            if fact and len(fact) >= 1:
                page = str(fact[0]).replace(" ", "_")
                if page and page not in seen:
                    seen.add(page)
                    pages.append(page)

        data_list.append({
            "id": f"hover_{item.get('uid', item.get('id', 'unknown'))}",
            "claim": item.get("claim", ""),
            "label": label,
            "evidence_pages": pages,
        })

    print("HoVer 原始标签分布：")
    for label, count in raw_label_counter.items():
        print(f"  {label}: {count} 条")

    supports = [item for item in data_list if item["label"] == "SUPPORTS"]
    refutes = [item for item in data_list if item["label"] == "REFUTES"]

    print("HoVer 映射后标签分布：")
    print(f"  SUPPORTS: {len(supports)} 条")
    print(f"  REFUTES: {len(refutes)} 条")

    if not supports or not refutes:
        print("警告：HoVer 数据中某一类为空，请检查 hover_dev.json 是否完整，或 label 字段格式是否不同。")

    rng = random.Random(seed)

    if sample_size < len(data_list):
        # 二分类均衡采样
        half = sample_size // 2

        sampled_supports = rng.sample(
            supports,
            min(half, len(supports))
        )

        sampled_refutes = rng.sample(
            refutes,
            min(sample_size - len(sampled_supports), len(refutes))
        )

        sampled_data = sampled_supports + sampled_refutes

        # 如果某一类不足，用剩余样本补齐
        remaining = sample_size - len(sampled_data)

        if remaining > 0:
            used_ids = {item["id"] for item in sampled_data}
            leftovers = [
                item for item in data_list
                if item["id"] not in used_ids
            ]

            sampled_extra = rng.sample(
                leftovers,
                min(remaining, len(leftovers))
            )

            sampled_data.extend(sampled_extra)

        rng.shuffle(sampled_data)

    else:
        sampled_data = data_list
        rng.shuffle(sampled_data)

    print(f"成功加载 {len(sampled_data)} 条 HoVer 真实数据！")
    print(f"  SUPPORTS: {sum(1 for item in sampled_data if item['label'] == 'SUPPORTS')} 条")
    print(f"  REFUTES: {sum(1 for item in sampled_data if item['label'] == 'REFUTES')} 条")

    return sampled_data


def load_fever_data(split=FEVER_SPLIT, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    加载FEVER数据集并随机采样。

    优先读取本地缓存文件（data/fever_{split}.json）；
    缓存不存在时，从手动下载的 shared_task_dev.jsonl 解析并生成缓存。
    """
    cache_file = os.path.join(BASE_DIR, "data", f"fever_{split}.json")

    if os.path.exists(cache_file):
        print(f"从本地加载FEVER数据集: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        print(f"数据集总数: {len(data_list)}条")

    else:
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
    """验证数据格式"""
    valid_data = []
    for item in data_list:
        if 'claim' not in item or 'label' not in item:
            continue
        if not item['claim'] or not isinstance(item['claim'], str):
            continue
        if item['label'] not in VALID_LABELS:
            continue
        if 'evidence_pages' not in item:
            item['evidence_pages'] = []
        valid_data.append(item)

    print(f"数据验证完成: {len(valid_data)}条有效数据")
    return valid_data


def get_label_distribution(data_list):
    distribution = {}
    for item in data_list:
        label = item['label']
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def collect_evidence_pages(data_list):
    pages = set()
    for item in data_list:
        for page in item.get('evidence_pages', []):
            if page:
                pages.add(page)
    return pages