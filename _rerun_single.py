# -*- coding: utf-8 -*-
"""单独重跑单个 PDF 文件入库（修复 _is_fake_table 后验证 000016_935）"""
import os
import sys

ROOT = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system"
sys.path.insert(0, ROOT)

from base import logger
from rag_qa.core.document_processor import process_pdf
from rag_qa.core.vector_store import VectorStore
from mysql_qa.db.mysql_client import MySQLClient

FNAME = "000016_AN202608271828556935.pdf"
FID = 12


def main():
    vs = VectorStore()
    mysql = MySQLClient()
    fpath = os.path.join(ROOT, "pdf_data", FNAME)
    try:
        vs.delete_by_file(FID)
        chunks, images = process_pdf(fpath)
        if not chunks:
            raise RuntimeError("解析结果为空")
        n_table = sum(1 for c in chunks if c.metadata.get("is_table"))
        n_pipe = sum(1 for c in chunks if "|" in c.page_content and not c.metadata.get("is_table"))
        for c in chunks:
            c.metadata["source"] = "unknown"
            c.metadata["code"] = ""
            c.metadata["name"] = ""
            c.metadata["pdf_time"] = ""
            c.metadata["file_id"] = FID
        vs.add_documents(chunks, extracted_images=images, file_id=FID)
        score = chunks[0].metadata.get("complexity_score", 0)
        method = chunks[0].metadata.get("parse_method", "unknown")
        n_parent = len(set(c.metadata.get("parent_id") for c in chunks))
        mysql.update_pdf_status(
            FID, "done", parse_method=method, complexity_score=score,
            parent_count=n_parent, child_count=len(chunks))
        logger.info(f"✓ {FNAME}: {len(chunks)} 子块 | {n_table} 表格 | 管道符 {n_pipe} | {method}")
    except Exception as e:
        logger.error(f"✗ {FNAME}: {e}")
        raise
    finally:
        mysql.close()


if __name__ == "__main__":
    main()
