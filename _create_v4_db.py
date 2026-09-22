# -*- coding: utf-8 -*-
"""创建 df_rag_v4：复制 v3 全部表结构；company_info/standard_qa 复制数据（非跑批产物），其余表留空待重新入库"""
import pymysql

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', charset='utf8mb4')
cur = conn.cursor()

# 1. 建库（若不存在）
cur.execute("CREATE DATABASE IF NOT EXISTS df_rag_v4 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci")
print("df_rag_v4 已创建/存在")

# 2. 复制全部表结构
TABLES = ['child_chunks', 'company_info', 'doc_images', 'doc_status', 'parent_docs', 'pdf_files', 'standard_qa']
for t in TABLES:
    cur.execute(f"CREATE TABLE IF NOT EXISTS df_rag_v4.{t} LIKE df_rag_v4.{t}")
    print(f"  表结构复制: {t}")

# 3. 复制非跑批产物数据（company_info 公司字典 / standard_qa 标准问答）
for t in ['company_info', 'standard_qa']:
    cur.execute(f"SELECT COUNT(*) FROM df_rag_v4.{t}")
    n = cur.fetchone()[0]
    cur.execute(f"INSERT IGNORE INTO df_rag_v4.{t} SELECT * FROM df_rag_v4.{t}")
    print(f"  数据复制: {t} ({n} 行)")

conn.commit()

# 4. 验证
cur.execute("SHOW TABLES FROM df_rag_v4")
print("v4 表:", [r[0] for r in cur.fetchall()])
for t in TABLES:
    cur.execute(f"SELECT COUNT(*) FROM df_rag_v4.{t}")
    print(f"  v4.{t} 行数: {cur.fetchone()[0]}")
conn.close()
