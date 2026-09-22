# -*- coding: utf-8 -*-
"""
重跑受伪表格/管道符污染的文件（v3 库）。
流程：delete_by_file 清理残留 → process_pdf 重新解析 → add_documents 入库 → 更新状态。
用法：python _rerun_polluted_files.py
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

# 含伪表格/管道符残留的文件（来自 v3 库 pipe_cnt > 0 统计）
POLLUTED_FILES = [
    "000019_AN202609141829380539.pdf",   # markitdown 16 伪表格
    "000011_AN202608311828829867.pdf",   # markitdown 11 伪表格（父块15所在文件）
    "000016_AN202609171829516535.pdf",   # markitdown 9 伪表格
    "000016_AN202609101829228626.pdf",   # markitdown 8 伪表格
    "000016_AN202608271828556932.pdf",   # markitdown 7 伪表格
    "000016_AN202608271828556936.pdf",   # markitdown 5 伪表格
    "000016_AN202609171829516625.pdf",   # markitdown 4 伪表格
    "000016_AN202608271828556935.pdf",   # mineru 空表格碎片
    "000010_AN202609111829268015.pdf",   # mineru 空表格碎片
    "000016_AN202609181829618242.pdf",   # mineru 管道符残留
    "000026_AN202609171829533488.pdf",   # mineru 管道符残留
    "000035_AN202609111829278389.pdf",   # mineru 管道符残留
    "000034_AN202608281828646440.pdf",   # mineru 管道符残留
    "000016_AN202609031828961330.pdf",   # mineru 管道符残留
]


def main():
    vs = VectorStore()  # 加载 BGE-M3（初始化较慢）
    mysql = MySQLClient()
    ok = 0
    fail = 0
    for fname in POLLUTED_FILES:
        fpath = os.path.join(PDF_DIR, fname)
        if not os.path.exists(fpath):
            logger.error(f"文件不存在: {fpath}")
            fail += 1
            continue
        try:
            # 1. 查 pdf_files 记录
            mysql.cursor.execute("SELECT * FROM pdf_files WHERE file_name = %s", (fname,))
            pf = mysql.cursor.fetchone()
            if not pf:
                logger.error(f"pdf_files 无记录，跳过: {fname}")
                fail += 1
                continue
            file_id = pf["id"]

            # 2. 清理上次入库残留（MySQL + Milvus）
            vs.delete_by_file(file_id)
            logger.info(f"清理完成: {fname} (file_id={file_id})")

            # 3. 重新解析 + 分块
            chunks, images = process_pdf(fpath)
            if not chunks:
                raise RuntimeError("解析结果为空")
            n_table = sum(1 for c in chunks if c.metadata.get("is_table"))
            n_pipe = sum(1 for c in chunks if "|" in c.page_content and not c.metadata.get("is_table"))

            # 4. 补充业务元数据
            for c in chunks:
                c.metadata["source"] = pf.get("pdf_type") or "unknown"
                c.metadata["code"] = pf.get("code") or ""
                c.metadata["name"] = pf.get("name") or ""
                c.metadata["pdf_time"] = str(pf.get("pdf_time") or "")
                c.metadata["doc_type"] = pf.get("doc_type") or ""
                c.metadata["file_id"] = file_id

            # 5. 入库
            vs.add_documents(chunks, extracted_images=images, file_id=file_id)

            # 6. 更新状态
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
    logger.info(f"\n===== 重跑完成：成功 {ok}，失败 {fail} =====")


if __name__ == "__main__":
    main()
