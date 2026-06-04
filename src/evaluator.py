"""
FEVER事实验证系统 - 评估指标模块 
"""
import math

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import numpy as np
from src.config import VALID_LABELS

# 安全数值工具函数
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    """样本标准差；只有 1 轮实验时标准差记为 0。"""
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _format_float(value: float) -> str:
    return f"{value:.4f}"

def calculate_metrics(y_true, y_pred, labels=None):
    """
    计算所有评估指标
    """
    # 动态适配标签空间：如果 y_true 里面不包含某种标签（如 HoVer 任务无 NOT ENOUGH INFO），自动过滤以保持指标整洁
    if labels is None:
        unique_true = set(y_true)
        labels = [lbl for lbl in VALID_LABELS if lbl in unique_true]
        if not labels:
            labels = VALID_LABELS

    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)

    # 计算 precision, recall, f1（macro 和 weighted 平均）
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # 各类别的详细指标
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # 幻觉率
    hallucination_rate = calculate_hallucination_rate(y_true, y_pred)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        'accuracy': accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_weighted,
        'weighted_recall': recall_weighted,
        'weighted_f1': f1_weighted,
        'hallucination_rate': hallucination_rate,
        'active_labels': labels,  # 动态记录本次评估实际使用的标签
        'per_class_metrics': {
            labels[i]: {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i],
                'support': int(support_per_class[i])
            }
            for i in range(len(labels))
        },
        'confusion_matrix': conf_matrix.tolist()
    }

    return metrics


def calculate_hallucination_rate(y_true, y_pred):
    """
    计算幻觉率
    """
    # 找出所有真实标签为NOT ENOUGH INFO的样本
    not_enough_info_indices = [i for i, label in enumerate(y_true) if label == "NOT ENOUGH INFO"]

    if len(not_enough_info_indices) == 0:
        return 0.0

    # 统计这些样本中，预测为 SUPPORTS 或 REFUTES 的数量
    hallucination_count = 0
    for idx in not_enough_info_indices:
        if y_pred[idx] in ["SUPPORTS", "REFUTES"]:
            hallucination_count += 1

    hallucination_rate = hallucination_count / len(not_enough_info_indices)
    return hallucination_rate

# 多轮指标聚合
def aggregate_round_metrics(all_round_metrics: list[dict]) -> dict:
    """计算多轮评估指标的 mean ± std。"""
    scalar_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "hallucination_rate",
    ]

    aggregated = {}
    for key in scalar_keys:
        values = [float(m[key]) for m in all_round_metrics if key in m]
        aggregated[key] = {
            "mean": round(_mean(values), 4),
            "std": round(_std(values), 4),
        }
    return aggregated

# 多轮汇总报告统一生成
def generate_aggregated_report(aggregated: dict, num_rounds: int) -> str:
    """生成多轮实验汇总报告字符串。"""
    label_map = {
        "accuracy": "Accuracy（准确率）",
        "macro_precision": "Macro Precision（宏平均精确率）",
        "macro_recall": "Macro Recall（宏平均召回率）",
        "macro_f1": "Macro F1-Score（宏平均F1）",
        "weighted_precision": "Weighted Precision（加权精确率）",
        "weighted_recall": "Weighted Recall（加权召回率）",
        "weighted_f1": "Weighted F1-Score（加权F1）",
        "hallucination_rate": "Hallucination Rate（幻觉率）",
    }

    lines = []
    lines.append("\n" + "=" * 72)
    lines.append(f"📌 多轮验证汇总（有效轮次：{num_rounds}）")
    lines.append("=" * 72)
    lines.append(f"{'指标':<36}{'均值':>12}{'标准差':>14}")
    lines.append("-" * 72)

    for key, name in label_map.items():
        if key not in aggregated:
            continue
        mean_val = aggregated[key].get("mean", 0.0)
        std_val = aggregated[key].get("std", 0.0)
        lines.append(f"{name:<36}{mean_val:>12.4f}{std_val:>14.4f}")

    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


def generate_final_report(
    experiment_mode: str,
    task_name: str,
    valid_rounds: int,
    num_rounds: int,
    sample_size: int,
    random_seed: int,
    aggregated: dict,
    last_metrics: dict,
    result_file: str,
    log_file: str,
    total_time_text: str,
) -> str:
    """
    生成最终实验报告。
    设计原则：
    1. 单轮实验：只打印详细分类报告，不额外打印 mean/std 汇总，避免重复。
    2. 多轮实验：先打印多轮均值±标准差，再打印最后一轮分类细节。
    3. 最后统一打印文件路径和耗时。
    """
    lines = []

    lines.append("\n" + "=" * 88)
    lines.append("FEVER/HoVer 事实验证系统 - 最终实验报告")
    lines.append("=" * 88)
    lines.append(f"实验模式: {experiment_mode}")
    lines.append(f"任务名称: {task_name}")
    lines.append(f"有效轮次: {valid_rounds} / {num_rounds}")
    lines.append(f"样本数量: {sample_size}")
    lines.append(f"随机种子: {random_seed}")
    lines.append("-" * 88)

    # 多轮时才打印均值±标准差；单轮不打印，避免和详细报告重复
    if valid_rounds > 1:
        lines.append("一、多轮汇总指标（mean ± std）")
        lines.append("-" * 88)

        metric_names = {
            "accuracy": "Accuracy（准确率）",
            "macro_precision": "Macro Precision（宏平均精确率）",
            "macro_recall": "Macro Recall（宏平均召回率）",
            "macro_f1": "Macro F1-Score（宏平均F1）",
            "weighted_f1": "Weighted F1-Score（加权F1）",
            "hallucination_rate": "Hallucination Rate（幻觉率）",
        }

        lines.append(f"{'指标':<36}{'均值':>12}{'标准差':>14}")
        lines.append("-" * 88)

        for key, name in metric_names.items():
            if key not in aggregated:
                continue
            mean_val = aggregated[key].get("mean", 0.0)
            std_val = aggregated[key].get("std", 0.0)

            # HoVer 双分类没有 NEI 时，幻觉率通常没有意义，但保留也不影响
            lines.append(f"{name:<36}{mean_val:>12.4f}{std_val:>14.4f}")

        lines.append("-" * 88)
        lines.append("二、最后一轮详细分类报告")
        lines.append("-" * 88)

    else:
        lines.append("一、单轮详细分类报告")
        lines.append("-" * 88)

    # 详细报告里已经包含整体指标、类别指标、混淆矩阵
    lines.append(generate_report(last_metrics).strip())

    lines.append("\n" + "-" * 88)
    lines.append("二、运行信息" if valid_rounds == 1 else "三、运行信息")
    lines.append("-" * 88)
    lines.append(f"总耗时:   {total_time_text}")
    lines.append(f"结果文件: {result_file}")
    lines.append(f"日志文件: {log_file}")
    lines.append("=" * 88 + "\n")

    return "\n".join(lines)


# 更稳的详细分类报告
def generate_report(metrics):
    """生成详细分类评估报告，自动适配不同标签空间。"""
    labels = metrics.get("active_labels", VALID_LABELS)
    conf_matrix = metrics.get("confusion_matrix", [])

    report = "\n" + "=" * 88 + "\n"
    report += "📊 FEVER/HoVer 事实验证系统 - 详细分类评估报告\n"
    report += "=" * 88 + "\n\n"

    report += "一、整体指标\n"
    report += "-" * 88 + "\n"
    report += f"Accuracy（准确率）:              {_format_float(metrics['accuracy'])}\n"
    report += f"Macro Precision（宏精确率）:     {_format_float(metrics['macro_precision'])}\n"
    report += f"Macro Recall（宏召回率）:        {_format_float(metrics['macro_recall'])}\n"
    report += f"Macro F1-Score（宏平均F1）:      {_format_float(metrics['macro_f1'])}\n"
    report += f"Weighted F1-Score（加权F1）:     {_format_float(metrics.get('weighted_f1', 0.0))}\n"

    if "NOT ENOUGH INFO" in labels:
        report += f"Hallucination Rate（幻觉率）:    {_format_float(metrics['hallucination_rate'])}\n"
    report += "\n"

    report += "二、各类别详细指标\n"
    report += "-" * 88 + "\n"
    report += f"{'Class':<24}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Correct/Support':>20}\n"
    report += "-" * 88 + "\n"

    for i, label in enumerate(labels):
        class_metrics = metrics["per_class_metrics"].get(label, {})
        precision = class_metrics.get("precision", 0.0)
        recall = class_metrics.get("recall", 0.0)
        f1 = class_metrics.get("f1", 0.0)
        support = class_metrics.get("support", 0)
        correct = conf_matrix[i][i] if i < len(conf_matrix) and i < len(conf_matrix[i]) else 0
        correct_support = f"{correct} / {support}"

        report += f"{label:<24}{precision:>12.4f}{recall:>12.4f}{f1:>12.4f}{correct_support:>20}\n"

    report += "-" * 88 + "\n"
    report += f"{'Macro Average':<24}{metrics['macro_precision']:>12.4f}{metrics['macro_recall']:>12.4f}{metrics['macro_f1']:>12.4f}{'-':>20}\n"
    report += "-" * 88 + "\n\n"

    report += "三、混淆矩阵（行=真实标签，列=预测标签）\n"
    report += "-" * 88 + "\n"
    report += print_confusion_matrix(conf_matrix, labels)
    report += "=" * 88 + "\n"

    return report

def print_confusion_matrix(conf_matrix, labels):
    """格式化混淆矩阵。"""
    if not conf_matrix:
        return "暂无混淆矩阵。\n"

    short_labels = [label[:16] for label in labels]
    matrix_header = "True \\ Pred"
    matrix_str = f"{matrix_header:<20}"
    for label in short_labels:
        matrix_str += f"{label:>18}"
    matrix_str += "\n"

    for i, label in enumerate(short_labels):
        matrix_str += f"{label:<20}"
        for j in range(len(labels)):
            value = conf_matrix[i][j] if i < len(conf_matrix) and j < len(conf_matrix[i]) else 0
            matrix_str += f"{value:>18}"
        matrix_str += "\n"

    return matrix_str

