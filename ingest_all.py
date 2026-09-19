# -*- coding: utf-8 -*-
"""
全量PDF入库脚本（含MD5去重）- 多线程并发版

流程：
  1. 遍历 pdf_data 目录下所有PDF
  2. 算文件MD5，查 pdf_files 表
  3. 已处理过(status=done)的跳过
  4. 未处理的：打分→解析→分块→入MySQL+Milvus
  5. 更新 pdf_files 表状态

用法：
  python ingest_all.py              # 全量入库（2线程）
  python ingest_all.py --limit 3    # 只跑前3个（测试）
  python ingest_all.py --workers 3  # 3线程
"""
import os
import sys
import time
import hashlib
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 项目根目录
ROOT = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system"
sys.path.insert(0, ROOT)

from base import logger
from rag_qa.core.document_processor import process_pdf
from rag_qa.core.vector_store import VectorStore
from mysql_qa.db.mysql_client import MySQLClient

PDF_DIR = os.path.join(ROOT, "pdf_data")

# 全局计数器（线程安全）
lock = threading.Lock()
ok_count = 0
fail_count = 0
skip_count = 0
start_time = time.time()
total_files = 0


def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def process_one_file(fname, mongo_col, vs):
    """处理单个文件（每个线程调用）"""
    global ok_count, fail_count, skip_count

    fpath = os.path.join(PDF_DIR, fname)
    file_size = os.path.getsize(fpath)
    file_md5 = md5_of_file(fpath)

    # 每个线程用自己的 MySQL 连接
    mysql = MySQLClient()

    try:
        # 1. MD5去重：查 pdf_files 表
        existing = mysql.get_pdf_by_md5(file_md5)
        if existing and existing["status"] == "done":
            with lock:
                skip_count += 1
            return "skip", fname

        # 1.5 文件大小限制：只跑指定区间的文件
        # 分段跑：先跑 <1MB 的
        MIN_SIZE_MB = 0
        MAX_SIZE_MB = 1  # 1MB
        file_size_mb = file_size / 1024 / 1024
        if file_size_mb < MIN_SIZE_MB or file_size_mb > MAX_SIZE_MB:
            # 插入记录，标记为failed
            art_code = fname.replace(".pdf", "").split("_", 1)[1] if "_" in fname else ""
            mongo_doc = mongo_col.find_one({"art_code": art_code})
            doc_code = mongo_doc.get("code", "") if mongo_doc else ""
            doc_name = mongo_doc.get("name", "") if mongo_doc else ""
            doc_pdf_type = mongo_doc.get("pdf_type", "") if mongo_doc else ""
            doc_pdf_time = mongo_doc.get("pdf_time", "") if mongo_doc else ""

            file_id = mysql.insert_pdf_file(
                file_name=fname,
                file_path=fpath,
                file_md5=file_md5,
                file_size=file_size,
                code=doc_code,
                name=doc_name,
                pdf_type=doc_pdf_type,
                pdf_time=doc_pdf_time,
            )
            mysql.update_pdf_status(file_id, "failed", error_msg=f"文件大小{file_size_mb:.1f}MB，跳过（本轮只跑<1MB）")
            with lock:
                skip_count += 1
            return "skip", fname

        # 2. 从MongoDB查业务信息
        art_code = fname.replace(".pdf", "").split("_", 1)[1] if "_" in fname else ""
        mongo_doc = mongo_col.find_one({"art_code": art_code})
        doc_code = mongo_doc.get("code", "") if mongo_doc else ""
        doc_name = mongo_doc.get("name", "") if mongo_doc else ""
        doc_pdf_type = mongo_doc.get("pdf_type", "") if mongo_doc else ""
        doc_pdf_time = mongo_doc.get("pdf_time", "") if mongo_doc else ""
        doc_type = mongo_doc.get("doc_type", "") if mongo_doc else ""

        # 3. 插入 pdf_files 记录
        file_id = mysql.insert_pdf_file(
            file_name=fname,
            file_path=fpath,
            file_md5=file_md5,
            file_size=file_size,
            code=doc_code,
            name=doc_name,
            pdf_type=doc_pdf_type,
            pdf_time=doc_pdf_time,
        )
        mysql.update_pdf_status(file_id, "processing")

        # 4. 解析+分块
        chunks = process_pdf(fpath)
        if not chunks:
            raise RuntimeError("解析结果为空")

        # 给每个chunk补充业务元数据
        for c in chunks:
            c.metadata["source"] = doc_pdf_type or "unknown"
            c.metadata["code"] = doc_code
            c.metadata["name"] = doc_name
            c.metadata["pdf_time"] = doc_pdf_time
            c.metadata["doc_type"] = doc_type
            c.metadata["file_id"] = file_id

        # 5. 入库（共享 vs 实例，BGE-M3 模型线程安全）
        vs.add_documents(chunks)

        # 6. 更新状态
        score = chunks[0].metadata.get("complexity_score", 0)
        method = chunks[0].metadata.get("parse_method", "unknown")
        parent_ids = set(c.metadata.get("parent_id") for c in chunks)
        mysql.update_pdf_status(
            file_id, "done",
            parse_method=method,
            complexity_score=score,
            parent_count=len(parent_ids),
            child_count=len(chunks),
        )

        with lock:
            ok_count += 1
            done = ok_count + fail_count + skip_count
            logger.info(f"[{done}/{total_files}] ✓ {fname[:40]}  {len(chunks)}子块 {method}")

        return "ok", fname

    except Exception as e:
        with lock:
            fail_count += 1
            logger.error(f"[✗] {fname[:40]}: {e}")
        return "fail", fname
    finally:
        mysql.close()


def main(limit=None, workers=2):
    global total_files, start_time

    # 初始化共享组件
    logger.info("初始化共享组件...")
    vs = VectorStore()  # BGE-M3 模型共享

    from pymongo import MongoClient
    mongo_client = MongoClient("mongodb://localhost:27017/")
    mongo_col = mongo_client["rag_doc_data"]["doc_url"]

    # 遍历PDF
    pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    if limit:
        pdfs = pdfs[:limit]
    total_files = len(pdfs)

    logger.info(f"待处理PDF: {total_files} 个, {workers}线程并发")
    start_time = time.time()

    # 线程池并发
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one_file, fname, mongo_col, vs): fname for fname in pdfs}

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"线程异常: {e}")

            # 每20个打印进度
            done = ok_count + fail_count + skip_count
            if done % 20 == 0 or done == total_files:
                elapsed = (time.time() - start_time) / 60
                rate = done / (time.time() - start_time) if time.time() > start_time else 0
                remain = (total_files - done) / rate / 60 if rate > 0 else 0
                logger.info(f"--- 进度 {done}/{total_files}  成功{ok_count} 失败{fail_count} 跳过{skip_count}  "
                            f"已用{elapsed:.1f}分钟 预计剩余{remain:.0f}分钟 ---")

    elapsed = (time.time() - start_time) / 60
    logger.info(f"\n===== 完成：成功 {ok_count}，失败 {fail_count}，跳过 {skip_count}，"
                f"耗时 {elapsed:.1f} 分钟 =====")

    mongo_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全量PDF入库（多线程并发）")
    parser.add_argument("--limit", type=int, default=None, help="只处理前N个PDF")
    parser.add_argument("--workers", type=int, default=2, help="并发线程数")
    args = parser.parse_args()
    main(limit=args.limit, workers=args.workers)
