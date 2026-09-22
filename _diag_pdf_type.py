# -*- coding: utf-8 -*-
"""验证 unknown 根因：pdf_files.pdf_type 来源 & parent_docs 分布（临时脚本）"""
import pymysql

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456',
                       database='df_rag_v4', charset='utf8mb4')
cur = conn.cursor()

# 1. pdf_files 里 pdf_type 为 NULL/空 的有几个？都是谁？
cur.execute("SELECT COUNT(*) FROM pdf_files WHERE pdf_type IS NULL OR pdf_type = ''")
print("pdf_files pdf_type 为空:", cur.fetchone()[0])
cur.execute("SELECT id, file_name FROM pdf_files WHERE pdf_type IS NULL OR pdf_type = '' LIMIT 10")
print("  文件:", cur.fetchall())

# 2. pdf_files pdf_type 分布（确认回填效果）
cur.execute("SELECT pdf_type, COUNT(*) FROM pdf_files GROUP BY pdf_type ORDER BY COUNT(*) DESC")
print("pdf_files pdf_type 分布:", cur.fetchall())

# 3. parent_docs 分类行来源：非 unknown 的父块，其文件状态分布
cur.execute("""
    SELECT f.status, COUNT(*) FROM parent_docs p
    JOIN pdf_files f ON p.file_id = f.id
    WHERE p.pdf_type IS NOT NULL AND p.pdf_type != 'unknown'
    GROUP BY f.status
""")
print("parent_docs 有分类行 → 文件状态:", cur.fetchall())

# 4. parent_docs unknown 行 → 文件状态分布
cur.execute("""
    SELECT f.status, COUNT(*) FROM parent_docs p
    JOIN pdf_files f ON p.file_id = f.id
    WHERE p.pdf_type = 'unknown' OR p.pdf_type IS NULL
    GROUP BY f.status
""")
print("parent_docs unknown 行 → 文件状态:", cur.fetchall())

# 5. 交叉验证：有多少 done 文件的父块全部是 unknown（即回填后任务新写入的）
cur.execute("""
    SELECT COUNT(DISTINCT p.file_id) FROM parent_docs p
    WHERE p.pdf_type = 'unknown' OR p.pdf_type IS NULL
""")
print("unknown 父块涉及的文件数:", cur.fetchone()[0])

conn.close()
