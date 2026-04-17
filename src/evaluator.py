"""
FEVER事实验证系统 - 评估指标模块
"""
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import numpy as np
from src.config import VALID_LABELS


def calculate_metrics(y_true, y_pred):
    """
    计算所有评估指标

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表

    Returns:
        dict: 包含所有指标的字典
    """
    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)

    # 计算precision, recall, f1（macro和weighted平均）
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # 各类别的详细指标
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=VALID_LABELS, zero_division=0
    )

    # 幻觉率
    hallucination_rate = calculate_hallucination_rate(y_true, y_pred)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)

    metrics = {
        'accuracy': accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_weighted,
        'weighted_recall': recall_weighted,
        'weighted_f1': f1_weighted,
        'hallucination_rate': hallucination_rate,
        'per_class_metrics': {
            VALID_LABELS[i]: {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i],
                'support': int(support_per_class[i])
            }
            for i in range(len(VALID_LABELS))
        },
        'confusion_matrix': conf_matrix.tolist()
    }

    return metrics


def calculate_hallucination_rate(y_true, y_pred):
    """
    计算幻觉率

    幻觉率定义：真实标签为NOT ENOUGH INFO，但模型预测为SUPPORTS或REFUTES的比例

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表

    Returns:
        float: 幻觉率（0-1之间）
    """
    # 找出所有真实标签为NOT ENOUGH INFO的样本
    not_enough_info_indices = [i for i, label in enumerate(y_true) if label == "NOT ENOUGH INFO"]

    if len(not_enough_info_indices) == 0:
        return 0.0

    # 统计这些样本中，预测为SUPPORTS或REFUTES的数量
    hallucination_count = 0
    for idx in not_enough_info_indices:
        if y_pred[idx] in ["SUPPORTS", "REFUTES"]:
            hallucination_count += 1

    hallucination_rate = hallucination_count / len(not_enough_info_indices)
    return hallucination_rate


def generate_report(metrics):
    """
    生成评估报告

    Args:
        metrics: 指标字典

    Returns:
        str: 格式化的报告文本
    """
    report = "\n" + "=" * 50 + "\n"
    report += "评估结果\n"
    report += "=" * 50 + "\n\n"

    # 整体指标
    report += f"Accuracy (准确率): {metrics['accuracy']:.4f}\n"
    report += f"Macro Precision (宏平均精确率): {metrics['macro_precision']:.4f}\n"
    report += f"Macro Recall (宏平均召回率): {metrics['macro_recall']:.4f}\n"
    report += f"Macro F1-Score (宏平均F1): {metrics['macro_f1']:.4f}\n"
    report += f"Hallucination Rate (幻觉率): {metrics['hallucination_rate']:.4f}\n\n"

    # 各类别详细指标
    report += "=" * 50 + "\n"
    report += "各类别详细指标\n"
    report += "=" * 50 + "\n"
    report += f"{'类别':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n"
    report += "-" * 70 + "\n"

    for label in VALID_LABELS:
        metrics_class = metrics['per_class_metrics'][label]
        report += f"{label:<20} {metrics_class['precision']:<12.4f} {metrics_class['recall']:<12.4f} "
        report += f"{metrics_class['f1']:<12.4f} {metrics_class['support']:<10}\n"

    # 混淆矩阵
    report += "\n" + "=" * 50 + "\n"
    report += "混淆矩阵\n"
    report += "=" * 50 + "\n"
    report += print_confusion_matrix(metrics['confusion_matrix'], VALID_LABELS)

    return report


def print_confusion_matrix(conf_matrix, labels):
    """
    格式化打印混淆矩阵

    Args:
        conf_matrix: 混淆矩阵（列表格式）
        labels: 标签列表

    Returns:
        str: 格式化的混淆矩阵文本
    """
    matrix_str = "\n预测 →\n真实 ↓\n\n"

    # 表头
    matrix_str += f"{'':>20}"
    for label in labels:
        matrix_str += f"{label[:15]:>18}"
    matrix_str += "\n"

    # 矩阵内容
    for i, label in enumerate(labels):
        matrix_str += f"{label[:18]:>20}"
        for j in range(len(labels)):
            matrix_str += f"{conf_matrix[i][j]:>18}"
        matrix_str += "\n"

    return matrix_str
