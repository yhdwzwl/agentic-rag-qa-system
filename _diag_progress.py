# -*- coding: utf-8 -*-
"""查询 v3 库全量入库进度（临时脚本）"""
import pymysql

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456',
                       database='df_rag_v4', charset='utf8mb4')
cur = conn.cursor()

cur.execute("SELECT status, COUNT(*) FROM pdf_files GROUP BY status")
print("pdf_files status:", cur.fetchall())

cur.execute("SELECT COUNT(*) FROM parent_docs")
print("parent_docs:", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM child_chunks")
print("child_chunks:", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM doc_images")
print("doc_images:", cur.fetchone()[0])

cur.execute("SELECT pdf_type, COUNT(*) FROM parent_docs GROUP BY pdf_type")
print("parent pdf_type:", cur.fetchall())

cur.execute("SELECT id, file_name, error_msg FROM pdf_files WHERE status='failed' LIMIT 10")
print("failed 文件:", cur.fetchall())

# 最新处理的几个 done 文件
cur.execute("SELECT id, file_name, parse_method, parent_count, child_count FROM pdf_files "
            "WHERE status='done' ORDER BY id DESC LIMIT 8")
print("最近 done 的文件:")
for r in cur.fetchall():
    print("  ", r)
conn.close()
