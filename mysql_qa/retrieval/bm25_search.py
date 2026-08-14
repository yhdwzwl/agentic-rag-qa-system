# 导入 BM25 算法
from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np
# 导入文本预处理
from mysql_qa.utils.preprocess import preprocess_text
# 导入日志
from base import logger


class BM25Search:
    def __init__(self, redis_client, mongo_client):
        # 初始化日志
        self.logger = logger
        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 Mongo 客户端
        self.mongo_client = mongo_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化问题列表
        self.questions = None
        # 初始化原始问题
        self.original_questions = None
        # 加载数据
        self._load_data()

    def _load_data(self):
        # 加载数据 redis key
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        # 从 Redis 获取原始问题
        self.original_questions = self.redis_client.get_data(original_key)
        # 从 Redis 获取分词问题
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        # 如果 Redis 中没有数据，从 Mongo 加载
        if not self.original_questions or not tokenized_questions:
            # 从 Mongo 获取问题
            self.original_questions = self.mongo_client.fetch_questions()
            if not self.original_questions:
                # 记录无问题警告
                self.logger.warning("未加载到问题")
                return
            # 分词问题
            for q in self.original_questions:
                preprocess_text(q)

            tokenized_questions = [preprocess_text(q) for q in self.original_questions]
            # 存储原始问题到 Redis
            self.redis_client.set_data(original_key, [(q) for q in self.original_questions])
            # 存储分词问题到 Redis
            self.redis_client.set_data(tokenized_key, tokenized_questions)
        # 设置问题列表
        self.questions = tokenized_questions
        # 初始化 BM25 模型
        self.bm25 = BM25Okapi(self.questions)
        # 记录 BM25 初始化成功
        self.logger.info("BM25 模型初始化完成")

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

        # 2. L1: Redis 缓存拦截
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            self.logger.info(f"Redis缓存命中 | Query: '{query}'")
            return cached_answer, "cache_hit"

        try:
            # 3. L2: BM25 检索与打分
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            softmax_scores = self._softmax(scores)
            # 【优化】使用 Min-Max 归一化替代 Softmax，避免分数过度平滑
            # min_score, max_score = scores.min(), scores.max()
            # if max_score == min_score:
            #     normalized_scores = np.zeros_like(scores)
            # else:
            #     normalized_scores = (scores - min_score) / (max_score - min_score)
            best_idx = softmax_scores.argmax()
            # 获取最高分
            best_score = softmax_scores[best_idx]
            # best_idx = normalized_scores.argmax()
            # best_score = normalized_scores[best_idx]

            self.logger.debug(f"BM25最高归一化分数: {best_score:.3f} | 候选: {self.original_questions[best_idx]}")

            # 4. 阈值判断与持久层查询
            if best_score >= threshold:
                original_question = self.original_questions[best_idx]
                # 【修正】统一使用 mongo_client
                answer = self.mongo_client.fetch_answer(original_question)

                if answer:
                    # 异步/后台缓存，不阻塞当前响应（可选优化）
                    self.redis_client.set_data(f"answer:{query}", answer)
                    self.logger.info(f"BM25检索成功 | Score: {best_score:.3f} | Query: '{query}'")
                    return answer, "bm25_hit"

            # 5. 未达标，返回 Fallback 信号
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