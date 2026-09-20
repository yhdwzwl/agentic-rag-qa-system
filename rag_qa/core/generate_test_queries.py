# -*- coding: utf-8 -*-
"""
RAG测试Query生成脚本

生成4种类型的query：
1. simple（简单问题）：单文档单事实
2. direct（直接问题）：单文档直接找答案
3. multi_hop（多跳问题）：多文档关联推理
4. hypothetical（假设问题）：需要推理/假设

用法：
  1. 填好下方的 API_URL 和 API_KEY
  2. 运行：python generate_test_queries.py
"""
import os
import sys
import time
import random
from datetime import datetime

# ========== 配置区（用户填充） ==========
API_URL = "https://llm-166gm3uet4pq0skg.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"  # 大模型API地址
API_KEY = "sk-ws-H.PIIPRLH.u6xG.MEUCIQDIpcUvxopQujujOc0ftk6KOOcxQHH-saXyq1NoPcNXHgIgK1_ldr1TNH7cZCVFUqaO9yxdOonKf3ZrnmjTiinTcf8"  # 大
MODEL_NAME = "qwen3.8-max-0902"  # 模型名称

# 生成配置
DOC_COUNT = 1000  # 从已入库的文档里抽多少篇来生成（跑全量就写大点，自动跳过已用过的）
QUERY_PER_DOC = 4  # 每篇文档生成多少条query

# MongoDB配置
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "rag_doc_data"
# ======================================

from mongoengine import (
    connect, Document, StringField, ListField,
    BooleanField, DateTimeField, IntField
)


# ==========================================
# MongoDB模型定义（mongoengine）
# ==========================================
class QueryModel(Document):
    """RAG测试Query模型"""
    meta = {
        'collection': 'rag_queries',
        'indexes': [
            'query_type',
            'source',
            'is_used',
        ]
    }

    query_text = StringField(required=True)  # 用户问题原文
    query_type = StringField(required=True)  # simple/direct/multi_hop/hypothetical
    difficulty = StringField(default="medium")  # easy/medium/hard
    expected_answer = StringField(default="")  # 标准答案
    expected_doc_ids = ListField(StringField())  # 应该检索到的文档ID列表
    source = StringField(default="llm_generated")  # 数据来源
    category = StringField(default="")  # 业务分类（融资公告/持股变动等）
    created_time = DateTimeField(default=datetime.now)
    updated_time = DateTimeField(default=datetime.now)
    is_used = BooleanField(default=False)  # 是否已经用于测试

    def save(self, *args, **kwargs):
        self.updated_time = datetime.now()
        return super(QueryModel, self).save(*args, **kwargs)

    def __str__(self):
        return f"[{self.query_type}] {self.query_text[:30]}..."


# 连接MongoDB
connect(MONGO_DB, host=MONGO_URI)


# ==========================================
# 从MySQL读已入库的文档（抽样）
# ==========================================
def sample_parent_docs(count: int = 100):
    """从MySQL抽已入库的父块文档（跳过已经生成过的）"""
    import pymysql
    from pymongo import MongoClient

    # 1. 先查MongoDB里已经用过的doc_id
    mongo_client = MongoClient(MONGO_URI)
    mongo_col = mongo_client[MONGO_DB]['rag_queries']
    used_doc_ids = set()
    for doc in mongo_col.find({}, {"expected_doc_ids": 1}):
        for did in doc.get("expected_doc_ids", []):
            used_doc_ids.add(int(did))
    mongo_client.close()

    print(f"已生成过query的文档: {len(used_doc_ids)} 个")

    # 2. 从MySQL抽没生成过的
    conn = pymysql.connect(
        host='localhost', port=3306,
        user='root', password='123456',
        database='df_rag', charset='utf8mb4'
    )
    cur = conn.cursor(pymysql.cursors.DictCursor)

    # 构造排除条件
    if used_doc_ids:
        placeholders = ','.join(['%s'] * len(used_doc_ids))
        query = f"""
            SELECT id, file_id, parent_content, code, name, pdf_type
            FROM parent_docs
            WHERE id NOT IN ({placeholders})
            ORDER BY RAND()
            LIMIT %s
        """
        params = list(used_doc_ids) + [count]
    else:
        query = """
            SELECT id, file_id, parent_content, code, name, pdf_type
            FROM parent_docs
            ORDER BY RAND()
            LIMIT %s
        """
        params = [count]

    cur.execute(query, params)
    docs = cur.fetchall()
    conn.close()

    print(f"本次新抽了 {len(docs)} 个父块文档")
    return docs


# ==========================================
# 生成Query的Prompt
# ==========================================
QUERY_TYPES = [
    ("simple", "简单问题", "单文档、单个事实，答案在文档里直接就能找到"),
    ("direct", "直接问题", "单文档、直接问某个方面的内容，需要总结"),
    ("multi_hop", "多跳问题", "需要结合多个文档/多个信息点推理才能回答"),
    ("hypothetical", "假设问题", "需要基于文档内容做假设、推理、预测，比如'如果...会怎么样'"),
]


def generate_prompt(doc_content: str, doc_name: str, doc_type: str) -> str:
    """生成prompt"""
    return f"""你是一个RAG测试专家。请基于以下股票公告文档，生成4种不同类型的用户测试query。

【文档信息】
- 公司名称：{doc_name}
- 公告类型：{doc_type}
- 文档内容：
{doc_content[:2000]}

【要求】
生成4条query，每种类型1条：
1. simple（简单问题）：单文档、单个事实，答案直接在文档里
2. direct（直接问题）：单文档、需要总结某个方面的内容
3. multi_hop（多跳问题）：需要结合文档里多个信息点推理才能回答
4. hypothetical（假设问题）：基于文档内容做假设推理，比如"如果...会怎么样"

每条query要求：
- 口语化，像真实用户会问的问题
- 长度10-50字
- 同时给出expected_answer（标准答案，尽量简短）

请严格按以下JSON格式返回，不要有多余解释：
{{
  "queries": [
    {{
      "query_type": "simple",
      "query_text": "xxx",
      "expected_answer": "xxx",
      "difficulty": "easy"
    }},
    {{
      "query_type": "direct",
      "query_text": "xxx",
      "expected_answer": "xxx",
      "difficulty": "medium"
    }},
    {{
      "query_type": "multi_hop",
      "query_text": "xxx",
      "expected_answer": "xxx",
      "difficulty": "hard"
    }},
    {{
      "query_type": "hypothetical",
      "query_text": "xxx",
      "expected_answer": "xxx",
      "difficulty": "hard"
    }}
  ]
}}

现在生成："""


# ==========================================
# 调用大模型
# ==========================================
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL,
)


def call_llm(prompt: str) -> list:
    """调用大模型API"""
    import json

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    content = resp.choices[0].message.content

    # 解析JSON
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]
        content = content.strip()
    if content.startswith("json"):
        content = content[4:].strip()

    result = json.loads(content)
    return result["queries"]


# ==========================================
# 主流程
# ==========================================
def main():
    print("=== RAG测试Query生成 ===")
    print(f"  抽样文档数: {DOC_COUNT} 篇")
    print(f"  每篇生成query: {QUERY_PER_DOC} 条")
    print(f"  预计总query数: {DOC_COUNT * QUERY_PER_DOC} 条")
    print()

    # 1. 从MySQL抽样文档
    docs = sample_parent_docs(DOC_COUNT)

    # 2. 逐条生成
    success = 0
    fail = 0

    for i, doc in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] 处理: {doc['name'][:15]} ({doc['pdf_type']})")

        try:
            prompt = generate_prompt(
                doc_content=doc['parent_content'],
                doc_name=doc['name'],
                doc_type=doc['pdf_type'],
            )
            queries = call_llm(prompt)

            # 存入MongoDB
            for q in queries:
                query_model = QueryModel(
                    query_text=q["query_text"],
                    query_type=q["query_type"],
                    difficulty=q.get("difficulty", "medium"),
                    expected_answer=q.get("expected_answer", ""),
                    expected_doc_ids=[str(doc['id'])],
                    category=doc['pdf_type'],
                    source="llm_generated",
                )
                query_model.save()

            success += len(queries)
            print(f"  ✓ 生成 {len(queries)} 条，累计 {success} 条")

        except Exception as e:
            fail += 1
            print(f"  ✗ 失败: {e}")

        time.sleep(0.5)

    print(f"\n===== 完成！ =====")
    print(f"  成功生成: {success} 条")
    print(f"  失败文档: {fail} 个")
    print(f"  存储位置: {MONGO_DB}.rag_queries")


if __name__ == "__main__":
    main()
