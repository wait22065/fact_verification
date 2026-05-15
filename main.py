"""
FEVER事实验证系统 - 主程序
"""
import os
import time
from src.config import LOG_DIR, LOG_FILE, SAMPLE_SIZE, RESULTS_FILE, EXPERIMENT_MODE  # 引入 EXPERIMENT_MODE
from src.utils import setup_logger, format_time
from src.data_loader import load_fever_data, get_label_distribution
from src.verifier import FactVerifier
from src.evaluator import calculate_metrics, generate_report
from src.data_loader import load_fever_data, load_hover_data, get_label_distribution # 导入 hover 加载器



def main():
    """主程序入口"""
    # 初始化日志
    logger = setup_logger(LOG_FILE)

    # 1. 在 main() 函数开头，根据模式打印标题：
    mode_titles = {
        "BASELINE": "任务一：基于大模型的直接事实判断（Baseline）",
        "RAG": "任务二：基于检索增强的事实校验（RAG）",
        "COT": "任务三：基于推理的事实校验（CoT）",
        "RAG_COT": "探索任务：检索增强 + 逐步推理 (RAG + CoT)" ,
        "EXTENDED_PIPELINE": "实验扩展：多跳推理 + 自动选证 + LLM裁判" # 🌟 新增这一行
    }
    task_name = mode_titles.get(EXPERIMENT_MODE, "未知任务")
    
    print("\n" + "=" * 60)
    print(f"FEVER事实验证系统 - {task_name}")
    print("=" * 60 + "\n")

    start_time = time.time()

    try:
        # 1. 加载数据
        if EXPERIMENT_MODE == "EXTENDED_PIPELINE":
            logger.info("步骤1: 加载 HoVer 多跳数据集 (扩展任务)")
            data_list = load_hover_data(sample_size=SAMPLE_SIZE)
        else:
            logger.info("步骤1: 加载 FEVER 单跳数据集")
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


        # 🟢 5. 动态生成保存路径，防止不同实验结果互相覆盖
        logger.info("步骤5: 保存结果")
        # 🟢 路径配置（修改为绝对路径，防止终端执行目录不同导致文件乱跑）
        # 获取当前 config.py 所在目录 (src) 的上一级目录 (即 fact_verification)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        DATA_CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
        RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
        log_dir = os.path.join(BASE_DIR, "logs")

        log_f = os.path.join(log_dir, "verification.log")
        results_file = os.path.join(RESULTS_DIR, "verification_results.json")

        # 确保目录存在
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        logger.info("步骤5: 保存结果")
        base_dir, file_name = os.path.split(RESULTS_FILE)
        name, ext = os.path.splitext(file_name)
        mode_suffix = EXPERIMENT_MODE.lower() # baseline, rag, 或 cot
        actual_results_file = os.path.join(base_dir, f"{name}_{mode_suffix}{ext}")

        output_data = {
            'experiment_mode': EXPERIMENT_MODE, # 记录实验模式
            'metrics': metrics,
            'detailed_results': results['detailed_results'],
            'summary': {
                'total_samples': len(data_list),
                'successful_predictions': len(results['y_pred']),
                'failed_predictions': len(data_list) - len(results['y_pred'])
            }
        }
        verifier.save_results(output_data, actual_results_file)

        # 显示总结
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("任务完成！")
        print(f"总耗时: {format_time(elapsed_time)}")
        print(f"结果文件已保存至: {actual_results_file}") # 显示实际保存的路径
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}")
        return
        
 



if __name__ == "__main__":
    main()
