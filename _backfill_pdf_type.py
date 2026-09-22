# -*- coding: utf-8 -*-
"""
回填 pdf_type/name/pdf_time/doc_type（临时脚本，跑完删除）：
1. 从 MongoDB rag_doc_data.doc_url 按文件名 code_art_code 匹配，回填 pdf_files 元数据
2. 同步 UPDATE parent_docs.pdf_type（JOIN pdf_files）
说明：_run_full_ingest.py 入库时 c.metadata["source"]=pf.get("pdf_type") or "unknown"，
回填后新入库父块自动带真实分类；此处只修正存量。
"""
import re
import pymysql
from pymongo import MongoClient

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456',
                       database='df_rag_v4', charset='utf8mb4', autocommit=True)
cur = conn.cursor()

mc = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
col = mc['rag_doc_data']['doc_url']

# 全量拉取元数据映射 (code, art_code) -> doc
meta_map = {}
for d in col.find({}, {"code": 1, "art_code": 1, "pdf_type": 1, "name": 1, "pdf_time": 1, "doc_type": 1}):
    if d.get("code") and d.get("art_code"):
        meta_map[(d["code"], d["art_code"])] = d
print(f"MongoDB 元数据映射条数: {len(meta_map)}")

cur.execute("SELECT id, file_name FROM pdf_files")
rows = cur.fetchall()
updated = 0
no_match = []
for fid, fname in rows:
    m = re.match(r"^(\d{6})_(AN\w+)\.pdf$", fname)
    if not m:
        no_match.append(fname)
        continue
    code, art = m.group(1), m.group(2)
    d = meta_map.get((code, art))
    if not d:
        no_match.append(fname)
        continue
    cur.execute(
        "UPDATE pdf_files SET pdf_type=%s, name=%s, pdf_time=%s WHERE id=%s",
        (d.get("pdf_type") or None, d.get("name") or None,
         d.get("pdf_time") or None, fid))
    updated += 1
print(f"pdf_files 回填: {updated} | 未匹配: {len(no_match)} -> {no_match[:10]}")

# 同步已入库父块的 pdf_type
cur.execute("""UPDATE parent_docs p JOIN pdf_files f ON p.file_id = f.id
               SET p.pdf_type = f.pdf_type
               WHERE f.pdf_type IS NOT NULL AND f.pdf_type != ''""")
print("parent_docs 同步行数:", cur.rowcount)

# 验证
cur.execute("SELECT pdf_type, COUNT(*) FROM parent_docs GROUP BY pdf_type")
print("回填后 parent_docs.pdf_type 分布:", cur.fetchall())
cur.execute("SELECT COUNT(*) FROM pdf_files WHERE pdf_type IS NULL OR pdf_type = ''")
print("pdf_files 仍无 pdf_type:", cur.fetchone()[0])
conn.close()
