# db/mysql_client.py
# RAG 系统 MySQL 数据层
import pymysql
import pandas as pd
from typing import List, Dict, Optional
from base import Config, logger


class MySQLClient:
    """RAG 系统 MySQL 数据层，管理文档元数据、对话历史、检索日志"""

    def __init__(self):
        self.logger = logger
        conf = Config()
        try:
            self.connection = pymysql.connect(
                host=conf.MYSQL_HOST,
                port=conf.MYSQL_PORT,
                user=conf.MYSQL_USER,
                password=conf.MYSQL_PASSWORD,
                database=conf.MYSQL_DATABASE,
                charset='utf8mb4'
            )
            self.cursor = self.connection.cursor(pymysql.cursors.DictCursor)
            self.logger.info(f"MySQL 连接成功: {conf.MYSQL_HOST}:{conf.MYSQL_PORT}/{conf.MYSQL_DATABASE}")
            # 自动建表
            self._create_tables_if_not_exists()
        except pymysql.MySQLError as e:
            self.logger.error(f"MySQL 连接失败: {e}")
            raise

    def _create_tables_if_not_exists(self):
        """自动创建所有需要的表（如果不存在）"""
        # standard_qa 标准问答对表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS standard_qa (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键',
                question TEXT NOT NULL COMMENT '标准问题',
                answer TEXT NOT NULL COMMENT '标准答案',
                category VARCHAR(100) COMMENT '分类：重大事项/风险提示/融资公告/持股变动/信息变更/通用',
                status VARCHAR(20) DEFAULT 'active' COMMENT '状态：active/inactive',
                hit_count INT DEFAULT 0 COMMENT '命中次数',
                created_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                updated_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                INDEX idx_category (category),
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='标准问答对表（人工校对）'
        """)
        self.connection.commit()
        self.logger.info("MySQL 表结构检查完成")

    # ==================== parent_docs（父文档表） ====================
    def insert_parent_doc(self, file_id: int = None, file_name: str = None,
                          file_path: str = None,
                          pdf_type: str = None, code: str = None,
                          name: str = None, pdf_time: str = None,
                          doc_type: str = None, parse_method: str = None,
                          complexity_score: int = None,
                          parent_content: str = None,
                          heading: str = None, heading_level: int = None,
                          heading_path: str = None) -> int:
        """插入父文档，返回 doc_id"""
        sql = """INSERT INTO parent_docs
                 (file_id, file_name, file_path, pdf_type, code, name, pdf_time, doc_type,
                  parse_method, complexity_score, parent_content, heading, heading_level, heading_path)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        self.cursor.execute(sql, (file_id, file_name, file_path, pdf_type, code,
                                  name, pdf_time, doc_type, parse_method,
                                  complexity_score, parent_content,
                                  heading, heading_level, heading_path))
        self.connection.commit()
        return self.cursor.lastrowid

    def get_parent_doc(self, doc_id: int) -> Dict:
        self.cursor.execute("SELECT * FROM parent_docs WHERE id = %s", (doc_id,))
        return self.cursor.fetchone()

    def get_parent_docs_by_status(self, status: str) -> List[Dict]:
        """按处理状态查询文档"""
        self.cursor.execute("""
            SELECT p.*, s.status, s.error_msg
            FROM parent_docs p
            LEFT JOIN doc_status s ON p.id = s.parent_id
            WHERE s.status = %s
        """, (status,))
        return self.cursor.fetchall()

    # ==================== child_chunks（子块表） ====================
    def insert_child_chunk(self, parent_id: int, chunk_index: int,
                           content: str, char_count: int = None,
                           chunk_type: str = 'text',
                           is_table: int = 0,
                           has_number: int = 0,
                           heading: str = None, heading_level: int = None,
                           heading_path: str = None, page_no: int = None) -> int:
        """插入子块，返回 chunk_id"""
        sql = """INSERT INTO child_chunks
                 (parent_id, chunk_index, content, char_count, chunk_type, is_table, has_number,
                  heading, heading_level, heading_path, page_no)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        self.cursor.execute(sql, (parent_id, chunk_index, content,
                                  char_count or len(content),
                                  chunk_type, is_table, has_number,
                                  heading, heading_level, heading_path, page_no))
        self.connection.commit()
        return self.cursor.lastrowid

    def insert_image(self, file_id: int, parent_id: int, chunk_id: int,
                     chunk_index: int, image_data: str, image_type: str,
                     image_caption: str = '', image_size_kb: float = 0) -> int:
        """插入图片记录，返回 image_id"""
        sql = """INSERT INTO doc_images
                 (file_id, parent_id, chunk_id, chunk_index, image_data, image_type, image_caption, image_size_kb)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
        self.cursor.execute(sql, (file_id, parent_id, chunk_id, chunk_index,
                                  image_data, image_type, image_caption, image_size_kb))
        self.connection.commit()

        return self.cursor.lastrowid

    def update_chunk_image_ids(self, chunk_id: int, image_ids: list):
        """更新子块的 image_ids 字段"""
        import json
        sql = "UPDATE child_chunks SET image_ids = %s WHERE id = %s"
        self.cursor.execute(sql, (json.dumps(image_ids), chunk_id))
        self.connection.commit()


    def get_child_chunks_by_parent(self, parent_id: int) -> List[Dict]:
        self.cursor.execute(
            "SELECT * FROM child_chunks WHERE parent_id = %s ORDER BY chunk_index",
            (parent_id,))
        return self.cursor.fetchall()

    # ==================== layout_docs（版面分析文档表） ====================
    def insert_layout_doc(self, parent_id: int, markdown_text: str,
                          heading_hierarchy: str = None,
                          table_count: int = 0, image_count: int = 0) -> int:
        """插入版面分析结果"""
        sql = """INSERT INTO layout_docs
                 (parent_id, markdown_text, heading_hierarchy, table_count, image_count)
                 VALUES (%s, %s, %s, %s, %s)"""
        self.cursor.execute(sql, (parent_id, markdown_text,
                                  heading_hierarchy, table_count, image_count))
        self.connection.commit()
        return self.cursor.lastrowid

    def get_layout_doc(self, parent_id: int) -> Dict:
        self.cursor.execute(
            "SELECT * FROM layout_docs WHERE parent_id = %s", (parent_id,))
        return self.cursor.fetchone()

    # ==================== chat_history（对话历史表） ====================
    def insert_chat(self, session_id: str, user_query: str,
                    bot_answer: str, retrieved_chunks: str = None,
                    latency_ms: int = None) -> int:
        """插入对话记录"""
        sql = """INSERT INTO chat_history
                 (session_id, user_query, bot_answer, retrieved_chunks, latency_ms)
                 VALUES (%s, %s, %s, %s, %s)"""
        self.cursor.execute(sql, (session_id, user_query, bot_answer,
                                  retrieved_chunks, latency_ms))
        self.connection.commit()
        return self.cursor.lastrowid

    def get_chat_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        self.cursor.execute("""
            SELECT * FROM chat_history
            WHERE session_id = %s
            ORDER BY created_time DESC
            LIMIT %s
        """, (session_id, limit))
        return list(reversed(self.cursor.fetchall()))

    # ==================== retrieval_logs（检索日志表） ====================
    def insert_retrieval_log(self, query: str, chunk_ids: str,
                             scores: str = None, rerank_scores: str = None,
                             latency_ms: int = None) -> int:
        """插入检索日志"""
        sql = """INSERT INTO retrieval_logs
                 (query, chunk_ids, scores, rerank_scores, latency_ms)
                 VALUES (%s, %s, %s, %s, %s)"""
        self.cursor.execute(sql, (query, chunk_ids, scores, rerank_scores, latency_ms))
        self.connection.commit()
        return self.cursor.lastrowid

    # ==================== doc_status（文档状态表） ====================
    def update_doc_status(self, parent_id: int, status: str,
                          error_msg: str = None):
        """更新文档处理状态: pending / processing / indexed / failed"""
        sql = """INSERT INTO doc_status (parent_id, file_path, status, error_msg)
                 VALUES (%s, %s, %s, %s)
                 ON DUPLICATE KEY UPDATE status = %s, error_msg = %s,
                                         updated_time = CURRENT_TIMESTAMP"""
        # 先查 file_path
        self.cursor.execute("SELECT file_path FROM parent_docs WHERE id = %s", (parent_id,))
        row = self.cursor.fetchone()
        file_path = row['file_path'] if row else ''

        self.cursor.execute(sql, (parent_id, file_path, status, error_msg, status, error_msg))
        self.connection.commit()

    def get_pending_docs(self) -> List[Dict]:
        """获取待处理的文档"""
        self.cursor.execute("""
            SELECT p.*, s.status
            FROM parent_docs p
            LEFT JOIN doc_status s ON p.id = s.parent_id
            WHERE s.status IS NULL OR s.status = 'pending'
        """)
        return self.cursor.fetchall()

    # ==================== pdf_files（PDF文件信息表） ====================
    def get_pdf_by_md5(self, file_md5: str) -> Optional[Dict]:
        """按MD5查PDF文件是否已处理过"""
        self.cursor.execute("SELECT * FROM pdf_files WHERE file_md5 = %s", (file_md5,))
        return self.cursor.fetchone()

    def insert_pdf_file(self, file_name: str, file_path: str, file_md5: str,
                        file_size: int = None, code: str = None,
                        name: str = None, pdf_type: str = None,
                        pdf_time: str = None) -> int:
        """插入PDF文件记录，返回 file_id"""
        sql = """INSERT INTO pdf_files
                 (file_name, file_path, file_md5, file_size, code, name, pdf_type, pdf_time)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                 ON DUPLICATE KEY UPDATE updated_time = CURRENT_TIMESTAMP"""
        self.cursor.execute(sql, (file_name, file_path, file_md5, file_size,
                                  code, name, pdf_type, pdf_time))
        self.connection.commit()
        self.cursor.execute("SELECT id FROM pdf_files WHERE file_md5 = %s", (file_md5,))
        return self.cursor.fetchone()['id']

    def update_pdf_status(self, file_id: int, status: str,
                          parse_method: str = None, complexity_score: int = None,
                          parent_count: int = None, child_count: int = None,
                          error_msg: str = None):
        """更新PDF处理状态"""
        fields = ["status = %s"]
        params = [status]
        if parse_method:
            fields.append("parse_method = %s")
            params.append(parse_method)
        if complexity_score is not None:
            fields.append("complexity_score = %s")
            params.append(complexity_score)
        if parent_count is not None:
            fields.append("parent_count = %s")
            params.append(parent_count)
        if child_count is not None:
            fields.append("child_count = %s")
            params.append(child_count)
        if error_msg is not None:
            fields.append("error_msg = %s")
            params.append(error_msg)
        params.append(file_id)
        sql = f"UPDATE pdf_files SET {', '.join(fields)} WHERE id = %s"
        self.cursor.execute(sql, params)
        self.connection.commit()

    # ==================== standard_qa（标准问答对表） ====================
    def get_all_qa_pairs(self) -> List[Dict]:
        """获取所有标准问答对，用于BM25初始化"""
        sql = "SELECT id, question, answer, category FROM standard_qa WHERE status = 'active'"
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def insert_qa_pair(self, question: str, answer: str,
                       category: str = None, status: str = 'active') -> int:
        """插入标准问答对"""
        sql = """INSERT INTO standard_qa
                 (question, answer, category, status, created_time, updated_time)
                 VALUES (%s, %s, %s, %s, NOW(), NOW())"""
        self.cursor.execute(sql, (question, answer, category, status))
        self.connection.commit()
        return self.cursor.lastrowid

    def get_answer_by_question(self, question: str) -> Optional[str]:
        """根据问题获取答案"""
        sql = "SELECT answer FROM standard_qa WHERE question = %s AND status = 'active'"
        self.cursor.execute(sql, (question,))
        row = self.cursor.fetchone()
        return row['answer'] if row else None

    # ==================== 工具方法 ====================
    def close(self):
        try:
            self.connection.close()
            self.logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            self.logger.error(f"关闭连接失败: {e}")


if __name__ == '__main__':
    # 测试连接
    client = MySQLClient()
    print("MySQL 连接测试成功!")

    # 测试插入
    doc_id = client.insert_parent_doc(
        file_name="test.pdf",
        file_path="/data/test.pdf",
        pdf_type="重大事项",
        code="600000",
        name="测试股票",
        parse_method="markitdown",
        complexity_score=2
    )
    print(f"插入父文档成功, id={doc_id}")

    chunk_id = client.insert_child_chunk(
        parent_id=doc_id,
        chunk_index=0,
        content="这是测试子块内容"
    )
    print(f"插入子块成功, id={chunk_id}")

    client.close()
