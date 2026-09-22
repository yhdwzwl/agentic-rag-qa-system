# -*- coding: utf-8 -*-
"""预建 Milvus df_finance_rag_v4 数据库（集合由 VectorStore 启动自动建）"""
from pymilvus import MilvusClient

temp = MilvusClient(uri="http://127.0.0.1:19530")
dbs = temp.list_databases()
print("当前 Milvus 数据库:", dbs)
if "df_finance_rag_v4" not in dbs:
    temp.create_database("df_finance_rag_v4")
    print("已创建 df_finance_rag_v4")
else:
    print("df_finance_rag_v4 已存在")
print("创建后数据库列表:", temp.list_databases())
temp.close()
