# 导入 BM25 算法
from rank_bm25 import BM25Okapi
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

    def search(self, query: str, threshold: float = 12.0) -> tuple[str | None, str]:
        """
        BM25 检索入口
        Returns:
            tuple: (答案内容, 状态码)
            状态码说明:
                - "bm25_hit": 命中库内标准问答对，无需RAG
                - "fallback": 未找到可靠答案，需要触发RAG
                - "invalid": 无效查询，需要触发RAG或拒绝回答
                - "error": 检索异常，需要触发RAG兜底

        阈值口径（重要）：threshold 为 BM25 原始分，不是 softmax 归一化分。
        按当前库规模（~100 条 standard_qa）实测标定：
            完全相同问题 raw≈19-35；相似问法/无关 query raw≤9.5，阈值 12 有安全余量。
        ⚠️ QA 库扩容（数千条）后 raw 分布会因候选竞争而变化，需重新标定阈值。
        """
        # 1. 参数校验
        if not query or not isinstance(query, str):
            self.logger.error(f"无效查询: {query}")
            return None, "invalid"

        # 2. BM25 模型未初始化（库里无标准问答对）
        if self.bm25 is None or not self.original_questions:
            self.logger.warning("BM25 模型未初始化（standard_qa 无 active 数据），直接放行 RAG")
            return None, "fallback"

        try:
            # 3. BM25 检索与打分（raw 原始分，直接与阈值比较）
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)  # 兼容 list / numpy array
            if len(scores) == 0:
                return None, "fallback"
            best_idx, best_score = max(enumerate(scores), key=lambda x: x[1])

            self.logger.debug(f"BM25最高原始分数: {best_score:.3f} | 候选: {self.original_questions[best_idx]}")

            # 4. 阈值判断
            if best_score >= threshold:
                original_question = self.original_questions[best_idx]
                answer = self.mysql_client.get_answer_by_question(original_question)

                if answer:
                    self.logger.info(f"BM25检索成功 | Score: {best_score:.3f} | Query: '{query}'")
                    return answer, "bm25_hit"

            # 5. 未达标，返回 Fallback 信号
            self.logger.info(f"BM25未达阈值 | Score: {best_score:.3f} < {threshold} | 需触发RAG")
            return None, "fallback"
        except Exception as e:
            self.logger.error(f"BM25检索异常: {e}", exc_info=True)
            return None, "error"