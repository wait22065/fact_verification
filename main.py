"""
FEVER事实验证系统 - 主程序

输出文件命名规则（均带 mode 和时间戳，不会互相覆盖）：
  结果：results/verification_results_{mode}_{timestamp}.json
  日志：logs/verification_{mode}_{timestamp}.log
"""
import os
import time
from datetime import datetime

from src.config import (
    LOG_DIR, RESULTS_DIR, SAMPLE_SIZE, EXPERIMENT_MODE,
)
from src.utils import setup_logger, format_time
from src.data_loader import load_fever_data, load_hover_data, get_label_distribution
from src.verifier import FactVerifier
from src.evaluator import calculate_metrics, generate_report


def main():
    """主程序入口"""

    # ------------------------------------------------------------------
    # 时间戳：格式 20260417_150523，精确到秒
    # 同一次运行的 results 和 log 共享，方便对照
    # ------------------------------------------------------------------
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_lower = EXPERIMENT_MODE.lower()  # 用于文件名，如 rag_bm25

    # ------------------------------------------------------------------
    # 动态生成本次运行的日志路径和结果路径
    # 不依赖 config 里的固定路径，直接在这里拼好
    # ------------------------------------------------------------------
    actual_log_file     = os.path.join(LOG_DIR,     f"verification_{mode_lower}_{timestamp}.log")
    actual_results_file = os.path.join(RESULTS_DIR, f"verification_results_{mode_lower}_{timestamp}.json")

    os.makedirs(LOG_DIR,     exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger = setup_logger(actual_log_file)
    logger.info(f"本次运行模式：{EXPERIMENT_MODE}，时间戳：{timestamp}")
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
    print(f"模式：{EXPERIMENT_MODE}　时间戳：{timestamp}")
    print("=" * 60 + "\n")

    start_time = time.time()

    try:
        # ------------------------------------------------------------------
        # 步骤1：加载数据
        # ------------------------------------------------------------------
        if EXPERIMENT_MODE == "EXTENDED_PIPELINE":
            logger.info("步骤1: 加载 HoVer 多跳数据集（扩展任务）")
            data_list = load_hover_data(sample_size=SAMPLE_SIZE)
        else:
            logger.info("步骤1: 加载 FEVER 单跳数据集")
            data_list = load_fever_data(sample_size=SAMPLE_SIZE)

        distribution = get_label_distribution(data_list)
        print("标签分布:")
        for label, count in distribution.items():
            print(f"  {label}: {count} 条")
        print()

        # ------------------------------------------------------------------
        # 步骤2：执行验证
        # ------------------------------------------------------------------
        logger.info("步骤2: 执行事实验证")
        verifier = FactVerifier(logger)
        results  = verifier.verify_claims(data_list)

        # ------------------------------------------------------------------
        # 步骤3：计算评估指标
        # ------------------------------------------------------------------
        logger.info("步骤3: 计算评估指标")
        if len(results['y_true']) == 0:
            print("\n错误：没有成功的预测结果，无法计算指标")
            return

        metrics = calculate_metrics(results['y_true'], results['y_pred'])

        # ------------------------------------------------------------------
        # 步骤4：生成报告
        # ------------------------------------------------------------------
        logger.info("步骤4: 生成评估报告")
        report = generate_report(metrics)
        print(report)

        # ------------------------------------------------------------------
        # 步骤5：保存结果
        # ------------------------------------------------------------------
        logger.info("步骤5: 保存结果")
        output_data = {
            'experiment_mode': EXPERIMENT_MODE,
            'timestamp':       timestamp,
            'metrics':         metrics,
            'detailed_results': results['detailed_results'],
            'summary': {
                'total_samples':          len(data_list),
                'successful_predictions': len(results['y_pred']),
                'failed_predictions':     len(data_list) - len(results['y_pred']),
            },
        }
        verifier.save_results(output_data, actual_results_file)
        logger.info(f"结果已保存至：{actual_results_file}")

        # ------------------------------------------------------------------
        # 完成总结
        # ------------------------------------------------------------------
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("任务完成！")
        print(f"总耗时:   {format_time(elapsed_time)}")
        print(f"结果文件: {actual_results_file}")
        print(f"日志文件: {actual_log_file}")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}")


if __name__ == "__main__":
    main()