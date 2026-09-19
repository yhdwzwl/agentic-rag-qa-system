# 导入 BM25 算法
from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np
# 导入文本预处理
from mysql_qa.utils.preprocess import preprocess_text
# 导入日志
from base import logger


class BM25Search:
    def __init__(self, mysql_client):
        # 初始化日志
        self.logger = logger
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化问题列表
        self.questions = None
        # 初始化原始问题
        self.original_questions = None
        # 加载数据
        self._load_data()

    def _load_data(self):
        # 从 MySQL 获取所有标准问答对
        qa_pairs = self.mysql_client.get_all_qa_pairs()
        if not qa_pairs:
            self.logger.warning("未加载到标准问答对")
            return
        self.original_questions = [pair['question'] for pair in qa_pairs]
        # 分词
        self.questions = [preprocess_text(q) for q in self.original_questions]
        # 初始化 BM25 模型
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info(f"BM25 模型初始化完成, 共 {len(self.questions)} 个标准问题")

    def search(self, query: str, threshold: float = 0.85) -> tuple[str | None, str]:
        """
        BM25 检索入口
        Returns:
            tuple: (答案内容, 状态码)
            状态码说明:
                - "cache_hit": Redis缓存命中，无需RAG
                - "bm25_hit": MongoDB检索且分数达标，无需RAG
                - "fallback": 未找到可靠答案，需要触发RAG
                - "invalid": 无效查询，需要触发RAG或拒绝回答
                - "error": 检索异常，需要触发RAG兜底
        """
        # 1. 参数校验
        if not query or not isinstance(query, str):
            self.logger.error(f"无效查询: {query}")
            return None, "invalid"

        try:
            # 2. BM25 检索与打分
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            softmax_scores = self._softmax(scores)
            best_idx = softmax_scores.argmax()
            best_score = softmax_scores[best_idx]

            self.logger.debug(f"BM25最高归一化分数: {best_score:.3f} | 候选: {self.original_questions[best_idx]}")

            # 3. 阈值判断
            if best_score >= threshold:
                original_question = self.original_questions[best_idx]
                answer = self.mysql_client.get_answer_by_question(original_question)

                if answer:
                    self.logger.info(f"BM25检索成功 | Score: {best_score:.3f} | Query: '{query}'")
                    return answer, "bm25_hit"

            # 4. 未达标，返回 Fallback 信号
            self.logger.info(f"BM25未达阈值 | Score: {best_score:.3f} < {threshold} | 需触发RAG")
            return None, "fallback"
        except Exception as e:
            self.logger.error(f"BM25检索异常: {e}", exc_info=True)
            return None, "error"

    def _softmax(self, scores):
        # 计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))  # 保证数值稳定性：减去每行的最大值
        # 返回归一化分数, BM25分数 映射到0-1
        return exp_scores / exp_scores.sum()