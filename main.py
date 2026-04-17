"""
FEVER事实验证系统 - 主程序
"""
import time
from src.config import LOG_FILE, SAMPLE_SIZE, RESULTS_FILE
from src.utils import setup_logger, format_time
from src.data_loader import load_fever_data, get_label_distribution
from src.verifier import FactVerifier
from src.evaluator import calculate_metrics, generate_report


def main():
    """主程序入口"""
    # 初始化日志
    logger = setup_logger(LOG_FILE)

    print("\n" + "=" * 60)
    print("FEVER事实验证系统 - 任务一：Baseline（无证据）")
    print("=" * 60 + "\n")

    start_time = time.time()

    try:
        # 1. 加载数据
        logger.info("步骤1: 加载FEVER数据集")
        data_list = load_fever_data(sample_size=SAMPLE_SIZE)

        # 显示标签分布
        distribution = get_label_distribution(data_list)
        print("\n标签分布:")
        for label, count in distribution.items():
            print(f"  {label}: {count}条")
        print()

        # 2. 执行验证
        logger.info("步骤2: 执行事实验证")
        verifier = FactVerifier(logger)
        results = verifier.verify_claims(data_list)

        # 3. 计算评估指标
        logger.info("步骤3: 计算评估指标")
        if len(results['y_true']) == 0:
            print("\n错误: 没有成功的预测结果，无法计算指标")
            return

        metrics = calculate_metrics(results['y_true'], results['y_pred'])

        # 4. 生成报告
        logger.info("步骤4: 生成评估报告")
        report = generate_report(metrics)
        print(report)

        # 5. 保存结果
        logger.info("步骤5: 保存结果")
        output_data = {
            'metrics': metrics,
            'detailed_results': results['detailed_results'],
            'summary': {
                'total_samples': len(data_list),
                'successful_predictions': len(results['y_pred']),
                'failed_predictions': len(data_list) - len(results['y_pred'])
            }
        }
        verifier.save_results(output_data, RESULTS_FILE)

        # 显示总结
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("任务完成！")
        print(f"总耗时: {format_time(elapsed_time)}")
        print(f"结果文件: {RESULTS_FILE}")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}")
        return


if __name__ == "__main__":
    main()
