"""
FEVER事实验证系统 - 数据加载模块
"""
from datasets import load_dataset
import random
from src.config import FEVER_SPLIT, SAMPLE_SIZE, RANDOM_SEED, VALID_LABELS


def load_fever_data(split=FEVER_SPLIT, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    加载FEVER数据集并随机采样

    Args:
        split: 数据集分割（train/validation/test）
        sample_size: 采样数量
        seed: 随机种子

    Returns:
        list: 采样后的数据列表，每个元素包含 {'id', 'claim', 'label'}
    """
    print(f"正在加载FEVER数据集 ({split})...")

    # 从Hugging Face加载FEVER数据集
    dataset = load_dataset("fever", "v1.0", split=split)

    print(f"数据集总数: {len(dataset)}条")

    # 转换为列表格式
    data_list = []
    for item in dataset:
        # 只提取claim和label，不使用evidence（任务一）
        data_list.append({
            'id': item['id'],
            'claim': item['claim'],
            'label': item['label']
        })

    # 验证数据
    data_list = validate_data(data_list)

    # 随机采样
    random.seed(seed)
    if sample_size < len(data_list):
        sampled_data = random.sample(data_list, sample_size)
        print(f"随机采样: {sample_size}条")
    else:
        sampled_data = data_list
        print(f"使用全部数据: {len(sampled_data)}条")

    return sampled_data


def validate_data(data_list):
    """
    验证数据格式

    Args:
        data_list: 数据列表

    Returns:
        list: 验证后的数据列表
    """
    valid_data = []

    for item in data_list:
        # 检查必需字段
        if 'claim' not in item or 'label' not in item:
            continue

        # 检查claim不为空
        if not item['claim'] or not isinstance(item['claim'], str):
            continue

        # 检查label是否有效
        if item['label'] not in VALID_LABELS:
            continue

        valid_data.append(item)

    print(f"数据验证完成: {len(valid_data)}条有效数据")
    return valid_data


def get_label_distribution(data_list):
    """
    获取标签分布统计

    Args:
        data_list: 数据列表

    Returns:
        dict: 标签分布字典
    """
    distribution = {}
    for item in data_list:
        label = item['label']
        distribution[label] = distribution.get(label, 0) + 1

    return distribution
