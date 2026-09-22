# -*- coding: utf-8 -*-
"""统计 parse_method 分布估算剩余时间（临时脚本）"""
import pymysql

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456',
                       database='df_rag_v4', charset='utf8mb4')
cur = conn.cursor()

cur.execute("SELECT parse_method, COUNT(*) FROM pdf_files WHERE status='done' GROUP BY parse_method")
print("done 文件 parse_method:", cur.fetchall())

cur.execute("SELECT parse_method, COUNT(*) FROM pdf_files WHERE status='pending' GROUP BY parse_method")
print("pending 文件 parse_method:", cur.fetchall())

# done 总数与 pending 总数
cur.execute("SELECT COUNT(*) FROM pdf_files WHERE status='done'")
print("done:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM pdf_files WHERE status='pending'")
print("pending:", cur.fetchone()[0])

conn.close()
