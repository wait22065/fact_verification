"""
FEVER事实验证系统 - 工具函数模块
"""
import json
import logging
import time
from pathlib import Path


def setup_logger(log_file, level=logging.INFO):
    """
    配置日志系统

    Args:
        log_file: 日志文件路径
        level: 日志级别

    Returns:
        logger: 配置好的logger对象
    """
    # 创建logger
    logger = logging.getLogger("FEVER_Verification")
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_json(data, file_path):
    """
    保存JSON文件

    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path):
    """
    加载JSON文件

    Args:
        file_path: 文件路径

    Returns:
        加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_time(seconds):
    """
    格式化时间显示

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"
