# -*- coding: utf-8 -*-
"""v4 批量入库：断点续跑 + 跳过已完成 + 自动质检门 + 模型额度耗尽即停。"""
import hashlib
import io
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pymongo import MongoClient

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from base import logger as proj_logger
from rag_qa.core.document_processor import process_pdf
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.quality_gate import check_file
from mysql_qa.db.mysql_client import MySQLClient

LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(LOG_DIR, f"batch_ingest_{datetime.now():%Y%m%d_%H%M%S}.log")

_fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
proj_logger.addHandler(_fh)


# ==================== 运行参数 ====================
PDF_DIR = os.path.join(ROOT, "pdf_data")
ONLY_FAILED = False
MAX_SIZE_MB = 1
# ================================================================


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def v3_meta_for(fname: str):
    try:
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
                "code": doc.get("code"),
                "name": doc.get("name"),
                "pdf_type": doc.get("pdf_type"),
                "pdf_time": str(doc.get("pdf_time") or ""),
            }
    except Exception:
        pass
    return {}


def v4_status_by_md5(mysql: MySQLClient, md5: str):
    mysql.cursor.execute(
        "SELECT id, status, child_count FROM pdf_files WHERE file_md5 = %s", (md5,))
    return mysql.cursor.fetchone()


def run_one(pdf_path: str, mysql: MySQLClient, vs: VectorStore):
    """处理单个 PDF。"""
    fname = os.path.basename(pdf_path)
    md5 = md5_of(pdf_path)

    meta = v3_meta_for(fname) or {}
    file_id = mysql.insert_pdf_file(
        file_name=fname, file_path=pdf_path, file_md5=md5,
        file_size=os.path.getsize(pdf_path),
        code=meta.get("code"), name=meta.get("name"),
        pdf_type=meta.get("pdf_type"), pdf_time=str(meta.get("pdf_time") or ""),
    )

    # 清理上次残留（failed重跑或新文件，delete不存在也不影响）
    print(f"  [CLEAN] file_id={file_id}，清理旧残留")
    vs.delete_by_file(file_id)

    t0 = time.time()
    chunks, images = process_pdf(pdf_path)
    if not chunks:
        raise RuntimeError("process_pdf 返回空 chunks")
    n_table = sum(1 for c in chunks if c.metadata.get("is_table"))

    for c in chunks:
        c.metadata["source"] = meta.get("pdf_type") or "unknown"
        c.metadata["code"] = meta.get("code") or ""
        c.metadata["name"] = meta.get("name") or ""
        c.metadata["pdf_time"] = str(meta.get("pdf_time") or "")
        c.metadata["file_id"] = file_id

    vs.add_documents(chunks, extracted_images=images, file_id=file_id)

    score = chunks[0].metadata.get("complexity_score", 0)
    method = chunks[0].metadata.get("parse_method", "unknown")
    n_parent = len(set(c.metadata.get("parent_id") for c in chunks))
    mysql.update_pdf_status(file_id, "done", parse_method=method,
                            complexity_score=score, parent_count=n_parent,
                            child_count=len(chunks))

    rep = check_file(file_id, mysql_client=mysql)
    if not rep.ok:
        raise RuntimeError(f"质检门未通过: {rep.errors}")

    elapsed = time.time() - t0
    print(f"[OK  ] {fname}  file_id={file_id}  chunks={len(chunks)} tables={n_table} "
          f"parents={n_parent} {method} {elapsed:.1f}s")


def main(max_files=0):
    """max_files: 本次最多处理多少个PDF，0=全部"""
    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    print(f"扫描到 {len(pdfs)} 个 PDF，日志: {_LOG_PATH}")
    print(f"参数: MAX_SIZE_MB={MAX_SIZE_MB}, max_files={max_files}, ONLY_FAILED={ONLY_FAILED}")

    mysql = MySQLClient()
    vs = VectorStore()

    done = failed = skipped = size_skipped = 0
    quota_hit = False
    for i, fname in enumerate(pdfs, 1):
        if max_files and (done + failed) >= max_files:
            print(f"达到 max_files={max_files}（已实际处理{done+failed}个），停止")
            break

        pdf_path = os.path.join(PDF_DIR, fname)

        file_size_mb = os.path.getsize(pdf_path) / 1024 / 1024
        if file_size_mb >= MAX_SIZE_MB:
            size_skipped += 1
            continue

        md5 = md5_of(pdf_path)
        existing = v4_status_by_md5(mysql, md5)

        if ONLY_FAILED:
            if not existing or existing["status"] != "failed":
                continue
        else:
            if existing and existing["status"] == "done" and (existing["child_count"] or 0) > 0:
                skipped += 1
                continue

        print(f"\n[{i}/{len(pdfs)}] {fname}")
        try:
            run_one(pdf_path, mysql, vs)
            done += 1
        except RuntimeError as e:
            if "table_enhancer" in str(e) or "所有 LLM 模型" in str(e):
                print(f"\n!!! LLM 额度耗尽，整批暂停: {e}")
                quota_hit = True
                break
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
    print(f"批处理结束: done={done} failed={failed} skipped={skipped} "
          f"size_skipped={size_skipped} quota_paused={quota_hit}")
    print(f"日志: {_LOG_PATH}")
    mysql.close()


if __name__ == "__main__":
    main(max_files=1)
