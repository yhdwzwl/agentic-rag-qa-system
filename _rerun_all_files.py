# -*- coding: utf-8 -*-
"""
全量重跑 v3 库已入库文件（修复 _safe_tail 重叠复制表格后，重建全部数据）。
流程：delete_by_file 清理 → process_pdf 重新解析 → add_documents 入库 → 更新状态。
用法：python _rerun_all_files.py
"""
import os
import sys

ROOT = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system"
sys.path.insert(0, ROOT)

from base import logger
from rag_qa.core.document_processor import process_pdf
from rag_qa.core.vector_store import VectorStore
from mysql_qa.db.mysql_client import MySQLClient

PDF_DIR = os.path.join(ROOT, "pdf_data")


def main():
    vs = VectorStore()  # 加载 BGE-M3（初始化较慢）
    mysql = MySQLClient()
    mysql.cursor.execute("SELECT id, file_name, file_path FROM pdf_files ORDER BY id")
    files = mysql.cursor.fetchall()
    logger.info(f"共 {len(files)} 个文件待重跑")
    ok = 0
    fail = 0
    for pf in files:
        fname = pf["file_name"]
        file_id = pf["id"]
        fpath = pf.get("file_path") or os.path.join(PDF_DIR, fname)
        if not os.path.exists(fpath):
            logger.error(f"文件不存在: {fpath}")
            fail += 1
            continue
        try:
            vs.delete_by_file(file_id)
            chunks, images = process_pdf(fpath)
            if not chunks:
                raise RuntimeError("解析结果为空")
            n_table = sum(1 for c in chunks if c.metadata.get("is_table"))
            n_pipe = sum(1 for c in chunks if "|" in c.page_content and not c.metadata.get("is_table"))

            for c in chunks:
                c.metadata["source"] = pf.get("pdf_type") or "unknown"
                c.metadata["code"] = pf.get("code") or ""
                c.metadata["name"] = pf.get("name") or ""
                c.metadata["pdf_time"] = str(pf.get("pdf_time") or "")
                c.metadata["doc_type"] = pf.get("doc_type") or ""
                c.metadata["file_id"] = file_id

            vs.add_documents(chunks, extracted_images=images, file_id=file_id)

            score = chunks[0].metadata.get("complexity_score", 0)
            method = chunks[0].metadata.get("parse_method", "unknown")
            n_parent = len(set(c.metadata.get("parent_id") for c in chunks))
            mysql.update_pdf_status(
                file_id, "done",
                parse_method=method, complexity_score=score,
                parent_count=n_parent, child_count=len(chunks),
            )
            ok += 1
            logger.info(f"✓ {fname} 重跑完成: {len(chunks)} 子块 | {n_table} 表格 | 文本含| {n_pipe} | {method}")
        except Exception as e:
            fail += 1
            logger.error(f"✗ {fname}: {e}")
    mysql.close()
    logger.info(f"\n===== 全量重跑完成：成功 {ok}，失败 {fail} =====")


if __name__ == "__main__":
    main()
