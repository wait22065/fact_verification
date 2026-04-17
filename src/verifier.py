"""
FEVER事实验证系统 - 验证核心逻辑模块
"""
import time
from tqdm import tqdm
from src.api_client import create_client
from src.prompt_builder import (
    build_verification_prompt,
    parse_model_response,
    save_parse_errors,
    clear_parse_errors
)
from src.utils import save_json
from src.config import RESULTS_FILE


class FactVerifier:
    """事实验证器"""

    def __init__(self, logger):
        """
        初始化验证器

        Args:
            logger: 日志对象
        """
        self.logger = logger
        self.client = create_client()
        self.results = []

    def verify_claims(self, data_list):
        """
        批量验证claims

        Args:
            data_list: 数据列表，每个元素包含 {'id', 'claim', 'label'}

        Returns:
            dict: 包含预测结果和真实标签的字典
        """
        self.logger.info(f"开始验证 {len(data_list)} 条数据...")

        # 清空之前的解析错误记录
        clear_parse_errors()

        y_true = []  # 真实标签
        y_pred = []  # 预测标签
        detailed_results = []  # 详细结果

        # 使用tqdm显示进度
        for item in tqdm(data_list, desc="验证进度"):
            claim_id = item['id']
            claim = item['claim']
            true_label = item['label']

            # 验证单条claim
            prediction = self._verify_single_claim(claim, claim_id)

            # 记录结果
            result = {
                'id': claim_id,
                'claim': claim,
                'true_label': true_label,
                'predicted_label': prediction,
                'correct': prediction == true_label if prediction else False
            }

            detailed_results.append(result)

            # 只有成功预测的才加入评估
            if prediction:
                y_true.append(true_label)
                y_pred.append(prediction)
            else:
                self.logger.warning(f"ID {claim_id} 预测失败，跳过")

            # 添加小延迟避免API速率限制
            time.sleep(0.5)

        self.logger.info(f"验证完成！成功: {len(y_pred)}/{len(data_list)}")

        # 保存解析错误日志
        save_parse_errors()

        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'detailed_results': detailed_results
        }

    def _verify_single_claim(self, claim, claim_id=None):
        """
        验证单条claim

        Args:
            claim: 待验证的声明
            claim_id: claim的ID（用于错误记录）

        Returns:
            str: 预测的标签，失败返回None
        """
        # 构造prompt（任务一：只包含claim）
        prompt = build_verification_prompt(claim)

        # 调用API
        response = self.client.call_api(prompt)

        if not response:
            return None

        # 解析响应（传入claim_id和claim用于错误记录）
        prediction = parse_model_response(response, claim_id=claim_id, claim=claim)

        if not prediction:
            self.logger.warning(f"无法解析模型响应 (ID: {claim_id})")

        return prediction

    def save_results(self, results, output_path=RESULTS_FILE):
        """
        保存验证结果

        Args:
            results: 结果字典
            output_path: 输出文件路径
        """
        save_json(results, output_path)
        self.logger.info(f"结果已保存到: {output_path}")
