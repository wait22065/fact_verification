"""
FEVER事实验证系统 - 数据加载模块
"""
from datasets import load_dataset
import random
import os
import json
from pathlib import Path
from src.config import FEVER_SPLIT, SAMPLE_SIZE, RANDOM_SEED, VALID_LABELS
import os
import json
import random
from pathlib import Path
from datasets import load_dataset
# 修改 src/data_loader.py
import os
import json
import random
from pathlib import Path

def load_hover_data(sample_size=50, seed=42):
    """读取手动下载的 HoVer 官方 Dev 集文件"""
    
    # 刚才让你放文件的路径
    file_path = "data/cache/hover_dev.json" 
    
    if not os.path.exists(file_path):
        print(f"🚨 错误：找不到文件 {file_path}")
        print("请确认你已经下载了 Dev set 并重命名放在了 data/cache/ 目录下。")
        return []

    print(f"✅ 正在读取手动下载的 HoVer 数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data_list = []
    for item in raw_data:
        # HoVer 的标签转换：SUPPORTED -> SUPPORTS, 其他 -> REFUTES
        # 注意：HoVer 有时用 NOT_SUPPORTED，我们统一转为 REFUTES 适配你的评估模块
        original_label = item.get('label', '')
        label = "SUPPORTS" if original_label == "SUPPORTED" else "REFUTES"
        
        data_list.append({
            'id': f"hover_{item.get('uid', 'unknown')}",
            'claim': item.get('claim', ''),
            'label': label
        })

    # 随机采样
    random.seed(seed)
    sampled_data = random.sample(data_list, min(sample_size, len(data_list)))
    print(f"成功加载 {len(sampled_data)} 条 HoVer 真实数据！")
    return sampled_data

def load_fever_data(split=FEVER_SPLIT, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    加载FEVER数据集并随机采样

    首次运行时从Hugging Face下载并保存到本地
    后续运行直接从本地JSON文件加载

    Args:
        split: 数据集分割（train/validation/test）
        sample_size: 采样数量
        seed: 随机种子

    Returns:
        list: 采样后的数据列表，每个元素包含 {'id', 'claim', 'label'}
    """
    # 本地缓存文件路径
    cache_file = f"data/fever_{split}.json"

    # 检查本地缓存是否存在
    if os.path.exists(cache_file):
        print(f"从本地加载FEVER数据集: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        print(f"数据集总数: {len(data_list)}条")
    else:
        print(f"首次运行，从Hugging Face下载FEVER数据集 ({split})...")

        # 从Hugging Face加载FEVER数据集
        # 使用新的数据集路径格式
        dataset = load_dataset("fever", "v1.0", split=split, trust_remote_code=True)

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

        # 保存到本地
        print(f"保存数据集到本地: {cache_file}")
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
