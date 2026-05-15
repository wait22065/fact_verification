"""
FEVER事实验证系统 - 主程序（多轮验证版）

运行逻辑：
  1. 用固定的 RANDOM_SEED 采样一批数据，所有轮次共用同一批数据
  2. 对同一批数据重复执行 NUM_ROUNDS 轮推理，捕捉大模型输出的随机性
  3. 所有轮次跑完后，对各标量指标取均值与标准差，打印汇总报告
  4. 将每轮明细 + 汇总指标一起写入同一个 JSON 文件

输出文件命名（带 mode 和时间戳，不会互相覆盖）：
  results/verification_results_{mode}_{timestamp}.json
  logs/verification_{mode}_{timestamp}.log
"""
import os
import time
import math
from datetime import datetime

from src.config import (
    LOG_DIR, RESULTS_DIR, SAMPLE_SIZE, EXPERIMENT_MODE,
    RANDOM_SEED, NUM_ROUNDS,
)
from src.utils import setup_logger, format_time
from src.data_loader import load_fever_data, load_hover_data, get_label_distribution
from src.verifier import FactVerifier
from src.evaluator import calculate_metrics, generate_report


# ──────────────────────────────────────────────────────────────
# 辅助函数：对多轮指标列表计算均值和标准差
# ──────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    """计算列表均值，列表为空时返回 0.0"""
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    """
    计算列表样本标准差（分母为 n-1）。
    只有 1 个值时标准差无意义，返回 0.0。
    """
    if len(values) < 2:
        return 0.0
    mu  = _mean(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def aggregate_metrics(all_round_metrics: list[dict]) -> dict:
    """
    对多轮 metrics 字典取标量指标的均值和标准差。

    返回格式：
    {
        "accuracy":           {"mean": 0.82, "std": 0.03},
        "macro_precision":    {"mean": ...,  "std": ...},
        "macro_recall":       {"mean": ...,  "std": ...},
        "macro_f1":           {"mean": ...,  "std": ...},
        "hallucination_rate": {"mean": ...,  "std": ...},
    }
    """
    # 只对这几个标量指标做聚合，per_class_metrics / confusion_matrix 不聚合
    scalar_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "hallucination_rate",
    ]
    aggregated = {}
    for key in scalar_keys:
        vals = [m[key] for m in all_round_metrics if key in m]
        aggregated[key] = {
            "mean": round(_mean(vals), 4),
            "std":  round(_std(vals),  4),
        }
    return aggregated


def print_aggregated_report(aggregated: dict, num_rounds: int) -> None:
    """
    将聚合指标以对齐格式打印到控制台。

    示例输出：
    ============================================================
    多轮验证汇总（共 3 轮）
    ============================================================
    指标                   均值        标准差
    ------------------------------------------------------------
    Accuracy              0.8200      ±0.0300
    Macro Precision       0.7900      ±0.0200
    ...
    ============================================================
    """
    label_map = {
        "accuracy":           "Accuracy（准确率）",
        "macro_precision":    "Macro Precision（宏平均精确率）",
        "macro_recall":       "Macro Recall（宏平均召回率）",
        "macro_f1":           "Macro F1-Score（宏平均F1）",
        "hallucination_rate": "Hallucination Rate（幻觉率）",
    }

    print("\n" + "=" * 60)
    print(f"多轮验证汇总（共 {num_rounds} 轮）")
    print("=" * 60)
    print(f"{'指标':<28}{'均值':>10}{'标准差':>12}")
    print("-" * 60)

    for key, label in label_map.items():
        if key in aggregated:
            mean_val = aggregated[key]["mean"]
            std_val  = aggregated[key]["std"]
            print(f"{label:<28}{mean_val:>10.4f}{'±' + f'{std_val:.4f}':>12}")

    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────
# 核心函数：单轮验证
# ──────────────────────────────────────────────────────────────

def run_single_round(
    round_idx:   int,
    logger,
    verifier:    "FactVerifier",
    data_list:   list,
) -> dict | None:
    """
    对已加载的 data_list 完成一轮完整的事实验证。

    参数
    ----
    round_idx : 当前轮次编号（从 1 开始，用于打印）
    logger    : 日志对象（整个 main() 共用，不重复创建）
    verifier  : 已初始化的 FactVerifier 实例（多轮共用，避免重复加载模型）
    data_list : 本次运行的数据（所有轮次传入同一批，观察模型随机性）

    返回
    ----
    包含 'metrics' 和 'detailed_results' 的字典；
    若本轮完全失败（无有效预测），返回 None。
    """
    print(f"\n{'─' * 60}")
    print(f"第 {round_idx} / {NUM_ROUNDS} 轮  |  数据量：{len(data_list)} 条")
    print(f"{'─' * 60}")

    logger.info(f"[轮次 {round_idx}] 开始事实验证，数据量：{len(data_list)} 条")

    # ------------------------------------------------------------------
    # 步骤A：执行验证（调用大模型逐条判断）
    # 每轮传入同一批 data_list，模型因 temperature>0 输出略有差异
    # ------------------------------------------------------------------
    results = verifier.verify_claims(data_list)

    # 若本轮无任何有效预测（全部 API 超时 / 解析失败），跳过该轮
    if len(results['y_true']) == 0:
        logger.warning(f"[轮次 {round_idx}] 无有效预测，跳过本轮")
        print(f"  [警告] 第 {round_idx} 轮无有效预测，已跳过")
        return None

    # ------------------------------------------------------------------
    # 步骤B：计算评估指标
    # ------------------------------------------------------------------
    logger.info(f"[轮次 {round_idx}] 计算评估指标")
    metrics = calculate_metrics(results['y_true'], results['y_pred'])

    # 打印本轮关键指标（不打印完整报告，避免多轮输出过于冗长）
    print(f"  Accuracy: {metrics['accuracy']:.4f}  |  "
          f"Macro F1: {metrics['macro_f1']:.4f}  |  "
          f"幻觉率: {metrics['hallucination_rate']:.4f}")
    logger.info(
        f"[轮次 {round_idx}] 完成 — "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"Macro F1={metrics['macro_f1']:.4f}, "
        f"幻觉率={metrics['hallucination_rate']:.4f}"
    )

    return {
        "round":            round_idx,
        "metrics":          metrics,
        "detailed_results": results["detailed_results"],
        "summary": {
            "total_samples":          len(data_list),
            "successful_predictions": len(results["y_pred"]),
            "failed_predictions":     len(data_list) - len(results["y_pred"]),
        },
    }


# ──────────────────────────────────────────────────────────────
# 主程序入口
# ──────────────────────────────────────────────────────────────

def main():
    """
    多轮验证主流程：
      1. 初始化日志、路径、verifier
      2. RAG_BM25 模式：检查/构建 BM25 索引（仅一次，多轮复用）
      3. 用固定 RANDOM_SEED 加载一次数据，所有轮次共用
      4. 循环执行 NUM_ROUNDS 轮推理，捕捉模型输出随机性
      5. 聚合所有轮次的指标，打印均值 ± 标准差
      6. 将每轮明细 + 汇总指标写入一个 JSON 文件
    """

    # ------------------------------------------------------------------
    # 初始化：时间戳 + 文件路径 + 日志
    # ------------------------------------------------------------------
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_lower = EXPERIMENT_MODE.lower()

    actual_log_file     = os.path.join(LOG_DIR,     f"verification_{mode_lower}_{timestamp}.log")
    actual_results_file = os.path.join(RESULTS_DIR, f"verification_results_{mode_lower}_{timestamp}.json")

    os.makedirs(LOG_DIR,     exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger = setup_logger(actual_log_file)
    logger.info(f"多轮验证启动 — 模式：{EXPERIMENT_MODE}，时间戳：{timestamp}")
    logger.info(f"共 {NUM_ROUNDS} 轮，固定采样 seed：{RANDOM_SEED}，每轮采样：{SAMPLE_SIZE} 条")
    logger.info(f"日志文件：{actual_log_file}")
    logger.info(f"结果文件：{actual_results_file}")

    # ------------------------------------------------------------------
    # 打印任务标题
    # ------------------------------------------------------------------
    mode_titles = {
        "BASELINE":          "任务一：基于大模型的直接事实判断（Baseline）",
        "RAG":               "任务二：基于检索增强的事实校验（RAG，实时 API）",
        "COT":               "任务三：基于推理的事实校验（CoT）",
        "RAG_COT":           "探索任务：检索增强 + 逐步推理（RAG + CoT）",
        "RAG_BM25":          "任务二变体：基于本地 dump 的 BM25+SBERT 检索增强",
        "EXTENDED_PIPELINE": "实验扩展：多跳推理 + 自动选证 + LLM 裁判",
    }
    task_name = mode_titles.get(EXPERIMENT_MODE, f"自定义模式（{EXPERIMENT_MODE}）")

    print("\n" + "=" * 60)
    print(f"FEVER事实验证系统 - {task_name}")
    print(f"模式：{EXPERIMENT_MODE}　|　时间戳：{timestamp}")
    print(f"轮数：{NUM_ROUNDS}　|　采样 seed：{RANDOM_SEED}　|　每轮数据量：{SAMPLE_SIZE} 条")
    print("=" * 60)

    total_start_time = time.time()

    try:
        # ------------------------------------------------------------------
        # RAG_BM25 模式：在所有轮次开始前检查/构建 BM25 索引
        # 只检查一次，后续各轮直接复用已有索引，避免重复构建
        # ------------------------------------------------------------------
        if EXPERIMENT_MODE == "RAG_BM25":
            from src.config import INDEX_DIR, DUMP_DIR
            from src.retriever import build_bm25_index_filtered

            index_files  = ["doc_ids.pkl", "sentences.pkl", "bm25.pkl"]
            index_missing = any(
                not os.path.exists(os.path.join(INDEX_DIR, f))
                for f in index_files
            )

            if index_missing:
                logger.info("BM25 索引不存在，开始自动构建（仅此一次）...")
                print("\n" + "=" * 60)
                print("BM25 索引不存在，开始自动构建（仅需一次，约 5-15 分钟）")
                print(f"  dump 目录：{DUMP_DIR}")
                print(f"  索引目录：{INDEX_DIR}")
                print("=" * 60)
                build_bm25_index_filtered(dump_dir=DUMP_DIR, index_dir=INDEX_DIR)
                logger.info("BM25 索引构建完成，后续轮次直接复用")
            else:
                logger.info(f"BM25 索引已就绪：{INDEX_DIR}，多轮复用")
                print(f"\nBM25 索引已就绪：{INDEX_DIR}（多轮共用，无需重建）")

        # ------------------------------------------------------------------
        # 初始化 FactVerifier（多轮共用同一实例）
        # 避免每轮重复初始化（对 RAG_BM25 尤为重要，SBERT 模型加载较慢）
        # ------------------------------------------------------------------
        logger.info("初始化 FactVerifier（多轮共用）")
        verifier = FactVerifier(logger)

        # ------------------------------------------------------------------
        # 数据加载：用固定 RANDOM_SEED 采样，所有轮次共用同一批数据
        # 目的是控制变量——轮间差异仅来自大模型输出的随机性
        # ------------------------------------------------------------------
        logger.info(f"加载数据（seed={RANDOM_SEED}，采样 {SAMPLE_SIZE} 条）")
        if EXPERIMENT_MODE == "EXTENDED_PIPELINE":
            logger.info("加载 HoVer 多跳数据集")
            data_list = load_hover_data(sample_size=SAMPLE_SIZE)
        else:
            logger.info("加载 FEVER 单跳数据集")
            data_list = load_fever_data(sample_size=SAMPLE_SIZE)

        # 打印标签分布，确认采样是否均匀
        distribution = get_label_distribution(data_list)
        print("\n数据加载完成，标签分布：")
        for label, count in distribution.items():
            print(f"  {label}: {count} 条")
        print(f"（以上数据在 {NUM_ROUNDS} 轮中共用，seed={RANDOM_SEED}）")

        # ------------------------------------------------------------------
        # 多轮循环：对同一批数据重复推理 NUM_ROUNDS 次
        # ------------------------------------------------------------------
        all_round_results: list[dict] = []   # 每轮的完整结果（明细 + 指标）
        all_round_metrics: list[dict] = []   # 仅指标，用于聚合

        for round_idx in range(1, NUM_ROUNDS + 1):
            round_start = time.time()

            round_result = run_single_round(
                round_idx=round_idx,
                logger=logger,
                verifier=verifier,
                data_list=data_list,
            )

            round_elapsed = time.time() - round_start

            if round_result is None:
                # run_single_round 内已打印警告，此处只记录日志
                logger.warning(f"第 {round_idx} 轮跳过（无有效预测）")
                continue

            # 记录本轮耗时并追加到结果列表
            round_result["elapsed_seconds"] = round(round_elapsed, 1)
            all_round_results.append(round_result)
            all_round_metrics.append(round_result["metrics"])

            print(f"  耗时：{format_time(round_elapsed)}")

        # ------------------------------------------------------------------
        # 检查：若所有轮次均失败，直接退出
        # ------------------------------------------------------------------
        if not all_round_metrics:
            print("\n错误：所有轮次均无有效预测，无法生成汇总报告")
            logger.error("所有轮次均无有效预测，程序终止")
            return

        # ------------------------------------------------------------------
        # 聚合指标：均值 + 标准差
        # ------------------------------------------------------------------
        valid_rounds = len(all_round_metrics)
        logger.info(f"开始聚合指标（有效轮次：{valid_rounds} / {NUM_ROUNDS}）")

        aggregated = aggregate_metrics(all_round_metrics)

        # 打印汇总报告（只打印均值和标准差，不再逐轮输出完整报告）
        print_aggregated_report(aggregated, valid_rounds)

        # ------------------------------------------------------------------
        # 保存结果：每轮明细 + 汇总指标 → 单个 JSON
        # ------------------------------------------------------------------
        logger.info(f"保存结果到：{actual_results_file}")

        output_data = {
            # ---- 实验元信息 ----
            "experiment_mode": EXPERIMENT_MODE,
            "timestamp":       timestamp,
            "num_rounds":      NUM_ROUNDS,
            "valid_rounds":    valid_rounds,
            "random_seed":     RANDOM_SEED,   # 固定采样 seed，所有轮次共用
            "sample_size":     SAMPLE_SIZE,

            # ---- 汇总指标（均值 ± 标准差） ----
            # 格式：{"accuracy": {"mean": 0.82, "std": 0.03}, ...}
            "aggregated_metrics": aggregated,

            # ---- 每轮明细 ----
            # 列表长度 == valid_rounds，每项包含 round / metrics / detailed_results / summary
            "rounds": all_round_results,
        }

        # 复用 verifier 的 save_results 方法（已有 JSON 序列化 + 异常处理）
        verifier.save_results(output_data, actual_results_file)
        logger.info(f"结果已保存至：{actual_results_file}")

        # ------------------------------------------------------------------
        # 完成总结
        # ------------------------------------------------------------------
        total_elapsed = time.time() - total_start_time
        print("=" * 60)
        print("全部轮次完成！")
        print(f"有效轮次:   {valid_rounds} / {NUM_ROUNDS}")
        print(f"总耗时:     {format_time(total_elapsed)}")
        print(f"结果文件:   {actual_results_file}")
        print(f"日志文件:   {actual_log_file}")
        print("=" * 60 + "\n")

        logger.info(
            f"多轮验证完成 — 有效轮次 {valid_rounds}/{NUM_ROUNDS}，"
            f"总耗时 {format_time(total_elapsed)}"
        )

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}")


if __name__ == "__main__":
    main()