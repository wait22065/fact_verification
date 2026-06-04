"""
FEVER事实验证系统 - 主程序 (多轮对齐优化版)
"""
import os
import time
from datetime import datetime
from src.prompt_builder import save_parse_errors
from src import config
from src.utils import setup_logger, format_time
from src.data_loader import load_fever_data, load_hover_data, get_label_distribution
from src.golden_loader import load_fever_gold_data  # 对齐到您的 golden_loader
from src.verifier import FactVerifier
from src.evaluator import (
    calculate_metrics,
    aggregate_round_metrics,
    generate_final_report,
)


# ------------------------------------------------------------------
# 配置安全降级读取 (防止 config.py 中未定义 NUM_ROUNDS 导致报错)
# ------------------------------------------------------------------
NUM_ROUNDS = getattr(config, "NUM_ROUNDS", 1)
SAMPLE_SIZE = getattr(config, "SAMPLE_SIZE", 50)
RANDOM_SEED = getattr(config, "RANDOM_SEED", 42)
EXPERIMENT_MODE = getattr(config, "EXPERIMENT_MODE", "RAG")


# ──────────────────────────────────────────────────────────────
# 核心验证单轮循环
# ──────────────────────────────────────────────────────────────
def run_single_round(round_idx: int, logger, verifier: FactVerifier, data_list: list) -> dict | None:
    print(f"\n{'─' * 60}")
    print(f"第 {round_idx} / {getattr(config, 'NUM_ROUNDS', 1)} 轮  |  数据量：{len(data_list)} 条")
    print(f"{'─' * 60}")

    logger.info(f"[轮次 {round_idx}] 开始事实验证，数据量：{len(data_list)} 条")

    results = verifier.verify_claims(data_list)

    if len(results["y_true"]) == 0:
        logger.warning(f"[轮次 {round_idx}] 无有效预测，跳过本轮")
        print(f"  [警告] 第 {round_idx} 轮无有效预测，已跳过")
        return None

    metrics = calculate_metrics(results["y_true"], results["y_pred"])

    print(
        f"  本轮完成：Accuracy={metrics['accuracy']:.4f}，"
        f"Macro F1={metrics['macro_f1']:.4f}，"
        f"有效预测={len(results['y_pred'])}/{len(data_list)}"
    )

    logger.info(
        f"[轮次 {round_idx}] 完成 — "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"Macro F1={metrics['macro_f1']:.4f}, "
        f"幻觉率={metrics['hallucination_rate']:.4f}"
    )

    return {
        "round": round_idx,
        "metrics": metrics,
        "detailed_results": results["detailed_results"],
        "summary": {
            "total_samples": len(data_list),
            "successful_predictions": len(results["y_pred"]),
            "failed_predictions": len(data_list) - len(results["y_pred"]),
        },
    }

# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_lower = EXPERIMENT_MODE.lower()

    # 【新增】EXTENDED_PIPELINE 模式下，把消融配置写进文件名
    if EXPERIMENT_MODE == "EXTENDED_PIPELINE":
        parts = []

        if getattr(config, "USE_MULTI_HOP", False):
            parts.append("multihop")

        if getattr(config, "USE_CROSS_ENCODER", False):
            parts.append("ce")

        if getattr(config, "USE_LLM_JUDGE", False):
            parts.append("judge")

        if parts:
            mode_lower += "_" + "_".join(parts)

    actual_log_file = os.path.join(
        config.LOG_DIR,
        f"verification_{mode_lower}_{timestamp}.log"
    )

    actual_results_file = os.path.join(
        config.RESULTS_DIR,
        f"verification_results_{mode_lower}_{timestamp}.json"
    )

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    logger = setup_logger(actual_log_file)
    logger.info(f"多轮验证启动 — 模式：{EXPERIMENT_MODE}，时间戳：{timestamp}")

    # 打印精确的实验对照标题
    mode_titles = {
        "BASELINE":          "任务一：直接事实判断基线（Baseline）",
        "COT":               "任务三：基于大模型逻辑推理的事实校验（CoT）",
        "RAG":               "任务二：基于本地词条极速查找的离线检索增强（RAG）",
        "RAG_COT":           "核心探索：本地离线检索增强 + 逻辑思维链推理（RAG + CoT）",
        "RAG_BM25":          "任务二增强：本地 BM25 + SBERT 两阶段检索事实校验（RAG_BM25）",
        "RAG_BM25_CE":       "任务二增强：BM25 + SBERT + CrossEncoder 精排事实校验",
        "EXTENDED_PIPELINE": "扩展任务：HoVer 语料多跳推理 + IRCoT + LLM-as-Judge",
        "RAG_GOLDEN":        "对照组一：人工金牌标注事实原句注入检验（RAG_GOLDEN）",
        "RAG_COT_GOLDEN":    "对照组二：人工金牌事实注入 + 逻辑链推理（RAG_COT_GOLDEN）",
    }

    task_name = mode_titles.get(EXPERIMENT_MODE, f"未知运行模式（{EXPERIMENT_MODE}）")

    print("\n" + "=" * 60)
    print(f"FEVER事实验证系统 - {task_name}")
    print(f"模式：{EXPERIMENT_MODE}　|　测试时间戳：{timestamp}")
    print(f"轮数：{NUM_ROUNDS}　|　采样 seed：{RANDOM_SEED}　|　每轮数据量：{SAMPLE_SIZE} 条")
    print("=" * 60)

    total_start_time = time.time()

    try:      
        # 初始化验证器（多轮共用同一实例，SBERT和API客户端只初始化一次，速度飞快）

        # ------------------------------------------------------------------
        # 数据自适应加载逻辑
        # ------------------------------------------------------------------
        logger.info(
            f"加载数据，seed={RANDOM_SEED}，采样 {SAMPLE_SIZE} 条"
        )

        if EXPERIMENT_MODE == "EXTENDED_PIPELINE":
            logger.info("加载 HoVer 多跳数据集")
            data_list = load_hover_data(
                sample_size=SAMPLE_SIZE,
                seed=RANDOM_SEED,
            )

        elif EXPERIMENT_MODE in ["RAG_GOLDEN", "RAG_COT_GOLDEN"]:
            logger.info("加载 FEVER Gold Evidence 数据集")
            data_list = load_fever_gold_data(
                sample_size=SAMPLE_SIZE,
                seed=RANDOM_SEED,
            )

        else:
            logger.info("加载 FEVER 标准数据集")
            data_list = load_fever_data(
                sample_size=SAMPLE_SIZE,
                seed=RANDOM_SEED,
            )

        if EXPERIMENT_MODE in ["RAG_BM25", "RAG_BM25_CE"]:
            from src.retriever import build_bm25_index_filtered

            index_files = ["doc_ids.pkl", "sentences.pkl", "bm25.pkl"]
            index_missing = any(
                not os.path.exists(os.path.join(config.INDEX_DIR, f))
                for f in index_files
            )

            if index_missing:
                logger.info("FEVER BM25 索引不存在，开始自动构建。")

                print("\n" + "=" * 60)
                print("FEVER BM25 索引不存在，开始自动构建。")
                print(f"dump 目录：{config.DUMP_DIR}")
                print(f"索引目录：{config.INDEX_DIR}")
                print("=" * 60)

                build_bm25_index_filtered(
                    dump_dir=config.DUMP_DIR,
                    index_dir=config.INDEX_DIR,
                )

                logger.info("FEVER BM25 索引构建完成。")

            else:
                logger.info(f"FEVER BM25 索引已就绪：{config.INDEX_DIR}")
                print(f"\nFEVER BM25 索引已就绪：{config.INDEX_DIR}")


        if EXPERIMENT_MODE == "EXTENDED_PIPELINE":
            from src.retriever import build_bm25_index_hover

            index_files = ["doc_ids.pkl", "sentences.pkl", "bm25.pkl"]
            index_missing = any(
                not os.path.exists(os.path.join(config.HOVER_INDEX_DIR, f))
                for f in index_files
            )

            if index_missing:
                logger.info("HoVer BM25 索引不存在，开始自动构建。")

                print("\n" + "=" * 60)
                print("HoVer BM25 索引不存在，开始自动构建。")
                print(f"dump 目录：{config.DUMP_DIR}")
                print(f"索引目录：{config.HOVER_INDEX_DIR}")
                print("=" * 60)

                build_bm25_index_hover(
                    dump_dir=config.DUMP_DIR,
                    index_dir=config.HOVER_INDEX_DIR,
                )

                logger.info("HoVer BM25 索引构建完成。")

            else:
                logger.info(f"HoVer BM25 索引已就绪：{config.HOVER_INDEX_DIR}")
                print(f"\nHoVer BM25 索引已就绪：{config.HOVER_INDEX_DIR}")  

        verifier = FactVerifier(logger)
        distribution = get_label_distribution(data_list)
        print("\n数据集载入完成。当前采样下的类别标签分布：")
        for label, count in distribution.items():
            print(f"  {label}: {count} 条")
        print(f"（多轮对比验证中将完全锁死该样本列表。锁定种子：{RANDOM_SEED}）")

        # ------------------------------------------------------------------
        # 多轮对比迭代
        # ------------------------------------------------------------------
        all_round_results = []
        all_round_metrics = []

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
                continue

            round_result["elapsed_seconds"] = round(round_elapsed, 1)
            all_round_results.append(round_result)
            all_round_metrics.append(round_result["metrics"])

            print(f"  单轮耗时：{format_time(round_elapsed)}")

        # 空置防崩溃判定
        if not all_round_metrics:
            print("\n[错误] 本次实验所有轮次运行失败，无法评估指标。")
            return

        # 聚合生成多轮均值和标准差
        valid_rounds = len(all_round_metrics)
        aggregated = aggregate_round_metrics(all_round_metrics)

        # ------------------------------------------------------------------
        # 数据持久化（整合一并导出）
        # ------------------------------------------------------------------
        output_data = {
            "experiment_mode": EXPERIMENT_MODE,
            "timestamp": timestamp,
            "num_rounds": NUM_ROUNDS,
            "valid_rounds": valid_rounds,
            "random_seed": RANDOM_SEED,
            "sample_size": SAMPLE_SIZE,
            "aggregated_metrics": aggregated,
            "rounds": all_round_results,
        }

        # 【新增】只在 EXTENDED_PIPELINE 中保存消融配置
        if EXPERIMENT_MODE == "EXTENDED_PIPELINE":
            output_data["ablation_config"] = {
                "USE_MULTI_HOP": getattr(config, "USE_MULTI_HOP", None),
                "MAX_HOP_ROUNDS": getattr(config, "MAX_HOP_ROUNDS", None),
                "USE_CROSS_ENCODER": getattr(config, "USE_CROSS_ENCODER", None),
                "USE_LLM_JUDGE": getattr(config, "USE_LLM_JUDGE", None),
            }

        verifier.save_results(output_data, actual_results_file)

        total_elapsed = time.time() - total_start_time

        final_report = generate_final_report(
            experiment_mode=EXPERIMENT_MODE,
            task_name=task_name,
            valid_rounds=valid_rounds,
            num_rounds=NUM_ROUNDS,
            sample_size=SAMPLE_SIZE,
            random_seed=RANDOM_SEED,
            aggregated=aggregated,
            last_metrics=all_round_results[-1]["metrics"],
            result_file=actual_results_file,
            log_file=actual_log_file,
            total_time_text=format_time(total_elapsed),
        )

        print(final_report)
        
    except Exception as e:
        # 【新增】即使程序中途异常，也尽量保存已经记录的解析错误
        save_parse_errors()
        logger.error(f"主测试流运行中断: {str(e)}", exc_info=True)
        print(f"\n[错误] 运行异常中断: {str(e)}")


if __name__ == "__main__":
    main()