# -*- coding: utf-8 -*-
"""v4 批量入库：断点续跑 + 跳过已完成 + 自动质检门 + 模型额度耗尽即停。

用法：
  python batch_ingest.py                       # 跑全部未完成
  python batch_ingest.py --max-files 5         # 先跑 5 个试水
  python batch_ingest.py --only-failed         # 只重跑 failed
  python batch_ingest.py --pdf-dir pdf_data    # 指定 PDF 目录（默认 ./pdf_data）

行为：
  - 扫 pdf_dir/*.pdf，按 md5 查 v4.pdf_files：status='done' 且 child_count>0 直接跳过
  - 每个文件独立 try/except；单文件失败标 failed 继续下一个
  - 入库成功后自动跑 quality_gate，ERROR 级问题标 failed
  - TableLLMEnhancer 抛 RuntimeError（所有模型额度耗尽）时立即停止整批，等人工续费后重跑
  - 进度实时打印，日志落 logs/batch_ingest_YYYYMMDD.log
"""
import argparse
import hashlib
import io
import logging
import os
import sys
import time
import traceback
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pymysql
from base.config import Config
from base import logger as proj_logger
from rag_qa.core.document_processor import process_pdf
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.quality_gate import check_file
from mysql_qa.db.mysql_client import MySQLClient

conf = Config()
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(LOG_DIR, f"batch_ingest_{datetime.now():%Y%m%d_%H%M%S}.log")

# 文件 logger
_fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
proj_logger.addHandler(_fh)


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def v3_meta_for(fname: str):
    """从 MongoDB 拿登记元数据（code/name/pdf_type/pdf_time 等）。按文件名中的 art_code 匹配。"""
    try:
        from pymongo import MongoClient
        # 文件名格式: 001359_AN202609101829234605.pdf → art_code=AN202609101829234605
        base = os.path.splitext(fname)[0]
        art_code = base.split("_")[-1] if "_" in base else ""
        if not art_code.startswith("AN"):
            return {}
        mc = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
        col = mc["rag_doc_data"]["doc_url"]
        doc = col.find_one({"art_code": art_code})
        mc.close()
        if doc:
            return {
                "file_name": fname,
                "file_md5": None,
                "file_size": None,
                "code": doc.get("code"),
                "name": doc.get("name"),
                "pdf_type": doc.get("pdf_type"),
                "pdf_time": str(doc.get("pdf_time") or ""),
            }
    except Exception:
        pass
    return {}


def v4_status_by_md5(mysql: MySQLClient, md5: str):
    """返回 (file_id, status, child_count) 或 None。"""
    mysql.cursor.execute(
        "SELECT id, status, child_count FROM pdf_files WHERE file_md5 = %s", (md5,))
    return mysql.cursor.fetchone()


def run_one(pdf_path: str, mysql: MySQLClient, vs: VectorStore):
    """处理单个 PDF。返回 (file_id, n_child, n_table)。异常向上抛。"""
    fname = os.path.basename(pdf_path)
    md5 = md5_of(pdf_path)

    # 1) 跳过已完成
    existing = v4_status_by_md5(mysql, md5)
    if existing and existing["status"] == "done" and (existing["child_count"] or 0) > 0:
        print(f"[SKIP] {fname} (file_id={existing['id']}, already done, chunks={existing['child_count']})")
        return existing["id"], existing["child_count"], -1

    # 2) 登记 v4 pdf_files（复用 v3 元数据）
    meta = v3_meta_for(fname) or {}
    file_id = mysql.insert_pdf_file(
        file_name=fname, file_path=pdf_path, file_md5=md5,
        file_size=os.path.getsize(pdf_path),
        code=meta.get("code"), name=meta.get("name"),
        pdf_type=meta.get("pdf_type"), pdf_time=str(meta.get("pdf_time") or ""),
    )

    # 2.5) 重跑 failed 文件前清理上次残留（MySQL child_chunks/parent_docs + Milvus vectors）
    if existing and existing["status"] == "failed":
        print(f"  [CLEAN] 重跑 failed 文件 file_id={file_id}，清理上次残留")
        vs.delete_by_file(file_id)

    # 3) 解析
    t0 = time.time()
    chunks, images = process_pdf(pdf_path)
    if not chunks:
        raise RuntimeError("process_pdf 返回空 chunks")
    n_table = sum(1 for c in chunks if c.metadata.get("is_table"))

    # 4) 填 metadata
    for c in chunks:
        c.metadata["source"] = meta.get("pdf_type") or "unknown"
        c.metadata["code"] = meta.get("code") or ""
        c.metadata["name"] = meta.get("name") or ""
        c.metadata["pdf_time"] = str(meta.get("pdf_time") or "")
        c.metadata["file_id"] = file_id

    # 5) 写 MySQL + Milvus
    vs.add_documents(chunks, extracted_images=images, file_id=file_id)

    score = chunks[0].metadata.get("complexity_score", 0)
    method = chunks[0].metadata.get("parse_method", "unknown")
    n_parent = len(set(c.metadata.get("parent_id") for c in chunks))
    mysql.update_pdf_status(file_id, "done", parse_method=method,
                            complexity_score=score, parent_count=n_parent,
                            child_count=len(chunks))

    # 6) 自动质检门
    rep = check_file(file_id, mysql_client=mysql)
    if not rep.ok:
        raise RuntimeError(f"质检门未通过: {rep.errors}")

    elapsed = time.time() - t0
    print(f"[OK  ] {fname}  file_id={file_id}  chunks={len(chunks)} tables={n_table} "
          f"parents={n_parent} {method} {elapsed:.1f}s")
    return file_id, len(chunks), n_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=os.path.join(ROOT, "pdf_data"))
    ap.add_argument("--max-files", type=int, default=0, help="0=不限制")
    ap.add_argument("--only-failed", action="store_true", help="只跑 status=failed 的")
    args = ap.parse_args()

    pdfs = sorted(f for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf"))
    print(f"扫描到 {len(pdfs)} 个 PDF，日志: {_LOG_PATH}")

    mysql = MySQLClient()
    vs = VectorStore()

    done = failed = skipped = 0
    quota_hit = False
    for i, fname in enumerate(pdfs, 1):
        if args.max_files and (done + failed + skipped) >= args.max_files:
            print(f"达到 --max-files={args.max_files}，停止")
            break

        pdf_path = os.path.join(args.pdf_dir, fname)
        md5 = md5_of(pdf_path)
        existing = v4_status_by_md5(mysql, md5)

        # --only-failed：只重跑 failed
        if args.only_failed:
            if not existing or existing["status"] != "failed":
                continue
        else:
            # 普通模式：done 跳过；failed 重试
            if existing and existing["status"] == "done" and (existing["child_count"] or 0) > 0:
                skipped += 1
                continue

        print(f"\n[{i}/{len(pdfs)}] {fname}")
        try:
            run_one(pdf_path, mysql, vs)
            done += 1
        except RuntimeError as e:
            # TableLLMEnhancer 全部模型失败 → 整批暂停
            if "table_enhancer" in str(e) or "所有 LLM 模型" in str(e):
                print(f"\n!!! 检测到 LLM 模型额度耗尽，整批暂停: {e}")
                print("!!! 请续费后重新运行本脚本（断点自动续跑，已 done 的文件会跳过）")
                quota_hit = True
                break
            # 其他 RuntimeError：单文件失败
            failed += 1
            if existing:
                mysql.update_pdf_status(existing["id"], "failed", error_msg=str(e)[:500])
            print(f"[FAIL] {fname}: {e}")
        except Exception as e:
            failed += 1
            if existing:
                mysql.update_pdf_status(existing["id"], "failed", error_msg=f"{type(e).__name__}: {e}"[:500])
            print(f"[ERR ] {fname}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"批处理结束: done={done} failed={failed} skipped={skipped} quota_paused={quota_hit}")
    print(f"日志: {_LOG_PATH}")
    mysql.close()


if __name__ == "__main__":
    main()
