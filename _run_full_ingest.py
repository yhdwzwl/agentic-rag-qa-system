# -*- coding: utf-8 -*-
"""
全量入库 v3 库（4783 个 PDF）：
阶段1 登记：扫描 pdf_data 目录，按 MD5 去重登记进 pdf_files（缺失则 INSERT，已有则跳过）
阶段2 入库：逐文件 delete_by_file 清理 → process_pdf 解析 → add_documents 入库 → 更新状态
断点续跑：脚本可随时中断/重跑，自动跳过 status='done' 的文件，failed 的会重试
用法：python _run_full_ingest.py
"""
import os
import re
import sys
import hashlib
import time

ROOT = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system"
sys.path.insert(0, ROOT)

from base import logger
from rag_qa.core.document_processor import process_pdf
from rag_qa.core.vector_store import VectorStore
from mysql_qa.db.mysql_client import MySQLClient

PDF_DIR = os.path.join(ROOT, "pdf_data")
START = time.time()


def file_md5(path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_code(fname):
    """从文件名解析股票代码：000001_AN2026...pdf -> 000001"""
    m = re.match(r"^(\d{6})", fname)
    return m.group(1) if m else None


def register_all(mysql):
    """阶段1：登记全部 PDF 到 pdf_files（按 MD5 去重）"""
    files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    logger.info(f"目录发现 {len(files)} 个 PDF，开始登记...")
    # 已登记的 md5 集合
    mysql.cursor.execute("SELECT file_md5 FROM pdf_files")
    known = {r["file_md5"] for r in mysql.cursor.fetchall()}
    added = 0
    skipped = 0
    for i, fname in enumerate(files, 1):
        fpath = os.path.join(PDF_DIR, fname)
        try:
            md5v = file_md5(fpath)
            if md5v in known:
                skipped += 1
                continue
            size = os.path.getsize(fpath)
            file_id = mysql.insert_pdf_file(
                file_name=fname, file_path=fpath, file_md5=md5v,
                file_size=size, code=parse_code(fname))
            known.add(md5v)
            added += 1
            if added % 200 == 0:
                logger.info(f"  登记进度 {i}/{len(files)}：新增 {added}，跳过 {skipped}")
        except Exception as e:
            logger.error(f"  登记失败 {fname}: {e}")
    logger.info(f"登记完成：新增 {added}，已存在跳过 {skipped}")


def main():
    vs = VectorStore()  # 加载 BGE-M3（初始化较慢）
    mysql = MySQLClient()
    register_all(mysql)

    # 待处理 = 未 done 的全部文件
    mysql.cursor.execute(
        "SELECT id, file_name, file_path, code, name, pdf_type, pdf_time, status "
        "FROM pdf_files WHERE status != 'done' ORDER BY id")
    todo = mysql.cursor.fetchall()
    done0 = 0
    mysql.cursor.execute("SELECT COUNT(*) AS c FROM pdf_files WHERE status='done'")
    done0 = mysql.cursor.fetchone()["c"]
    logger.info(f"已 done {done0}，本次待处理 {len(todo)} 个文件")

    ok = 0
    fail = 0
    fail_list = []
    for n, pf in enumerate(todo, 1):
        fname = pf["file_name"]
        file_id = pf["id"]
        fpath = pf.get("file_path") or os.path.join(PDF_DIR, fname)
        if not os.path.exists(fpath):
            logger.error(f"✗ 文件不存在: {fpath}")
            fail += 1
            fail_list.append((fname, "文件不存在"))
            mysql.update_pdf_status(file_id, "failed", error_msg="文件不存在")
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
            if n % 50 == 0 or ok == 1:
                elapsed = time.time() - START
                logger.info(
                    f"[进度 {n}/{len(todo)}] ✓ {fname}: {len(chunks)} 子块 | {n_table} 表格 | "
                    f"管道符 {n_pipe} | {method} | 已用 {elapsed/60:.1f} 分钟")
            else:
                logger.info(f"✓ {fname}: {len(chunks)} 子块 | {n_table} 表格 | {method}")
        except Exception as e:
            fail += 1
            fail_list.append((fname, str(e)[:300]))
            mysql.update_pdf_status(file_id, "failed", error_msg=str(e)[:1000])
            logger.error(f"✗ {fname}: {e}")
            # 偶发瞬时错误（如显存/OOM）短暂退避后继续
            if "CUDA out of memory" in str(e):
                logger.warning("  检测到显存不足，等待 60s 后继续")
                time.sleep(60)

    mysql.close()
    elapsed = time.time() - START
    logger.info(f"\n===== 全量入库完成：成功 {ok}，失败 {fail}，耗时 {elapsed/60:.1f} 分钟 =====")
    if fail_list:
        logger.info("失败清单：")
        for fname, err in fail_list:
            logger.info(f"  {fname}: {err}")


if __name__ == "__main__":
    main()
