# -*- coding: utf-8 -*-
"""
[会话体系剥离] SessionQASystem：会话编排单一权威（2026-10-03 从 new_main.py 剥离）

职责：系统初始化（MySQL/Redis/BM25/VectorStore/RAGSystem/MemoryManager 装配）、
会话方法（init_conversation_table / _fetch_recent_history / get_session_history /
update_session_history / clear_session_history）、完整编排（query）、LLM 入口（call_dashscope）。

设计约束（防循环导入 / 前端无关）：
- 从 rag_qa.core.* 直接模块级 import，不走 rag_qa/__init__.py 的聚合导出
- 禁止 import streamlit / fastapi / uvicorn / new_main，可被任意前端复用
- 逻辑与原 new_main.IntegratedQASystem 逐行一致（搬移，非重写）
"""
# 导入MySQL和Redis客户端，管理数据库和缓存
from mysql_qa import MySQLClient, RedisClient, BM25Search
# 导入RAG系统组件（直接模块级导入，不走 rag_qa/__init__.py 聚合导出）
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.new_rag_system import RAGSystem
# [统一模型调用] 主 LLM 轮换调用器：从 config.ini [llm] table_llm_models 读取模型列表，
# 额度耗尽（403/402/429/insufficient_quota 等）自动切换下一个模型（2026-10-03 新增）
from rag_qa.core import llm_router
# [12] 三层记忆模块（user_id 写死 10 个 uuid，见 memory_manager.DUMMY_USER_IDS）
from rag_qa.core.memory_manager import MemoryManager
# 导入配置和日志
from base import logger, Config
# 导入OpenAI客户端，用于DashScope API
from openai import OpenAI
# 导入时间库，记录处理时间
import time
# 导入UUID生成唯一会话ID（new_main.main() 使用；本模块保留同名导入，与剥离源一致）
import uuid
# 导入pymysql错误处理
import pymysql


class SessionQASystem:
    """会话编排层：初始化 + 会话方法 + 完整编排 + LLM 入口，不依赖任何前端框架。

    query() 的 session_id 为 None 时保持降级行为（无会话则不读写记忆、不写历史），
    与 app.py 传入空 session 的行为一致。
    """

    def __init__(self):
        # 初始化日志
        self.logger = logger
        # 初始化配置
        self.config = Config()
        # 初始化MySQL客户端
        self.mysql_client = MySQLClient()
        # 初始化Redis客户端
        self.redis_client = RedisClient()
        # 初始化BM25搜索（BM25Search 只依赖 mysql_client；redis_client 为后续缓存预留）
        self.bm25_search = BM25Search(self.mysql_client)
        try:
            # 初始化OpenAI客户端，连接DashScope API
            self.client = OpenAI(api_key=self.config.DASHSCOPE_API_KEY,
                                 base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            self.logger.error(f"OpenAI客户端初始化失败: {e}")
            raise
        # 初始化向量存储
        self.vector_store = VectorStore()
        # 初始化RAG系统
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)
        # 初始化对话历史表
        self.init_conversation_table()
        # [12] 三层记忆模块初始化（失败降级 None，记忆功能不启用，不影响主流程）
        self.memory_manager = None
        try:
            self.memory_manager = MemoryManager(self.redis_client, self.mysql_client)
            logger.info("三层记忆模块初始化成功")
        except Exception as e:
            logger.warning(f"三层记忆模块初始化失败，记忆功能降级（不影响主流程）: {e}")

    def init_conversation_table(self):
        """初始化MySQL中的conversations表，用于存储对话历史"""
        try:
            self.mysql_client.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(64) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                )
            """)
            self.mysql_client.connection.commit()
            self.logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"初始化对话历史表失败: {e}")
            raise

    # 调用模型（[统一模型调用] 2026-10-03：从 table_llm_models 轮换，额度耗尽自动切下一个）
    def call_dashscope(self, prompt):
        """调用DashScope API生成答案（流式输出；模型列表来自 config.ini [llm] table_llm_models，
        额度耗尽/调用失败自动切换下一个模型；全部失败由调用方 fail-open 降级）"""
        return llm_router.stream_generate(self.config, prompt)

    def _fetch_recent_history(self, session_id: str) -> list:
        """获取最近 N 轮对话历史（[12] 窗口记忆，轮数=conf.WINDOW_ROUNDS，默认3）"""
        try:
            limit = self.config.WINDOW_ROUNDS
            self.mysql_client.cursor.execute("""
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC, id DESC
                LIMIT %s
            """, (session_id, limit))
            history = [{"question": row["question"], "answer": row["answer"]}
                       for row in self.mysql_client.cursor.fetchall()]
            return history[::-1]
        except pymysql.MySQLError as e:
            self.logger.error(f"获取对话历史失败: {e}")
            return []

    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        """更新会话历史到MySQL，保留最近 N 轮对话（[12] 窗口记忆，轮数=conf.WINDOW_ROUNDS）"""
        try:
            self.mysql_client.cursor.execute("""
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """, (session_id, question, answer))
            history = self._fetch_recent_history(session_id)
            limit = self.config.WINDOW_ROUNDS
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id
                        FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC, id DESC
                        LIMIT %s
                    ) AS sub
                )
            """, (session_id, session_id, limit))
            self.mysql_client.connection.commit()
            self.logger.info(f"会话 {session_id} 历史更新成功")
            return history
        except pymysql.MySQLError as e:
            self.logger.error(f"更新会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            raise
        except Exception as e:
            self.logger.error(f"更新会话历史意外错误: {e}")
            self.mysql_client.connection.rollback()
            raise

    def get_session_history(self, session_id: str) -> list:
        """从MySQL获取会话历史"""
        return self._fetch_recent_history(session_id)

    def clear_session_history(self, session_id: str) -> bool:
        """清除指定会话历史"""
        try:
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s
            """, (session_id,))
            self.mysql_client.connection.commit()
            self.logger.info(f"会话 {session_id} 历史已清除")
            return True
        except pymysql.MySQLError as e:
            self.logger.error(f"清除会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            return False

    def query(self, query, source_filter=None, session_id=None, user_id=None):
        """查询集成系统，支持对话历史和流式输出

        [12] 三层记忆：
        - user_id：长期记忆的用户维度（app 可传；None 时按 session_id 稳定映射到写死列表）
        - memory_context：工作记忆+长期记忆注入文本（MemoryManager 不可用时为空串，行为与旧版一致）
        """
        start_time = time.time()
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")
        history = self.get_session_history(session_id) if session_id else []

        # [12] 读取工作记忆+长期记忆 → 注入文本（任何异常降级为空串，不阻断主流程）
        memory_context = ""
        if self.memory_manager is not None and session_id:
            try:
                memory_context = self.memory_manager.build_memory_context(session_id, user_id)
            except Exception as e:
                self.logger.warning(f"读取记忆上下文失败（降级为空串）: {e}")
                memory_context = ""

        # 首先尝试MySQL精确查询（threshold=12：BM25 原始分，按库规模标定）
        answer, need_rag = self.bm25_search.search(query, threshold=12)

        if answer:
            # MySQL找到精确答案，直接返回
            self.logger.info(f"MySQL答案: {answer}")
            if session_id:
                # 历史写入失败只记日志，不阻断回答（Bug2 修复：session_id 列宽/DB 异常降级）
                try:
                    self.update_session_history(session_id, query, answer)  # 把BM25的QA更新到会话历史
                except Exception as e:
                    self.logger.warning(f"会话历史更新失败（不影响回答）: {e}")
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 对于MySQL答案，一次性返回
            yield answer, True  # 第二个参数表示这是完整答案
        elif need_rag:
            # 需要使用RAG系统，支持流式输出
            self.logger.info("无可靠MySQL答案，回退到RAG")

            # 收集完整答案以便存储
            collected_answer = ""

            # 从RAG系统获取流式输出
            for token in self.rag_system.generate_answer(query, source_filter=source_filter,
                                                         history=history, memory_context=memory_context):
                collected_answer += token
                yield token, False  # 第二个参数表示这是部分答案

            # 完整答案收集完毕，更新会话历史
            if session_id:
                # 历史写入失败只记日志，不阻断回答与后续记忆回写（Bug2 修复：session_id 列宽/DB 异常降级）
                try:
                    self.update_session_history(session_id, query, collected_answer)  # 把RAG的QA更新到会话历史
                except Exception as e:
                    self.logger.warning(f"会话历史更新失败（不影响回答与记忆）: {e}")

            # [12] 三层记忆：流式结束后回写（matched_companies 来自分类器现成实体识别；
            # 记忆更新在流式之后，不增加首 token 延迟；任何异常只记日志不影响主流程）
            if self.memory_manager is not None and session_id:
                try:
                    matched = self.rag_system.last_query_result.get("matched_companies", []) or []
                    self.memory_manager.update_memory(
                        session_key=session_id,
                        user_id=user_id,
                        question=query,
                        answer=collected_answer,
                        matched_companies=matched,
                        source_filter=source_filter,
                    )
                except Exception as e:
                    self.logger.warning(f"记忆更新失败（不影响主流程）: {e}")

            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 标记结束
            yield "", True  # 空字符串表示流结束，True表示完成
        else:
            # 未找到答案
            self.logger.info("未找到答案")
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield "未找到答案", True  # 一次性返回
