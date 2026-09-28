#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并发下载公告 PDF（直连 requests 版本）

特点：
  - 低并发 + UA 轮换 + 随机间隔，避免触发反爬
  - 断点续传：已下载的自动跳过
  - 失败自动重试，失败记录日志
  - 实时进度显示
"""
import os
import time
import random
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from pymongo import MongoClient

# ==================== 配置 ====================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "rag_doc_data"
COLLECTION_NAME = "doc_url"

PDF_SAVE_DIR = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system\pdf_data"

# 下载设置
MAX_WORKERS = 5              # 并发数（保守一点，避免被封）
PER_REQUEST_DELAY = (0.5, 1.5)  # 每个请求前随机延迟区间（秒）
MAX_RETRY = 3                # 单条失败重试次数
TIMEOUT = 15                 # 单条下载超时（秒）

# 随机 UA 池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==================== 下载单条 ====================
def download_one(doc):
    """下载单条公告PDF，返回 (code, status, msg)
    status: ok / skip / fail
    """
    code = doc["code"]
    art_code = doc["art_code"]
    save_path = os.path.join(PDF_SAVE_DIR, f"{code}_{art_code}.pdf")

    # 已存在 = 之前下过，直接标记
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
        return code, "skip", "已存在"

    url = f"https://pdf.dfcfw.com/pdf/H2_{art_code}_1.pdf"

    for attempt in range(1, MAX_RETRY + 1):
        try:
            # 随机延迟，避免频率太高
            delay = random.uniform(*PER_REQUEST_DELAY)
            time.sleep(delay)

            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Referer": "https://data.eastmoney.com/",
                "Accept": "application/pdf,*/*",
                "Accept-Language": "zh-CN,zh;q=0.9",
            }

            r = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)

            if r.status_code != 200:
                logger.warning(f"{code} HTTP {r.status_code}，第{attempt}次重试")
                time.sleep(2 * attempt)  # 退避
                continue

            # 校验 PDF 头
            content = r.content
            if content[:4] != b"%PDF":
                logger.warning(f"{code} 不是有效PDF，第{attempt}次重试")
                time.sleep(2 * attempt)
                continue

            # 保存文件
            with open(save_path, "wb") as f:
                f.write(content)

            # 更新 MongoDB
            MongoClient(MONGO_URI)[DB_NAME][COLLECTION_NAME].update_one(
                {"_id": doc["_id"]},
                {"$set": {"is_used": True, "updated_time": datetime.now()}},
            )

            return code, "ok", f"{len(content)//1024}KB"

        except Exception as e:
            logger.warning(f"{code} 第{attempt}次失败: {e}")
            time.sleep(2 * attempt)

    return code, "fail", "重试耗尽"


# ==================== 主流程 ====================
def main(limit=None, workers=MAX_WORKERS):
    os.makedirs(PDF_SAVE_DIR, exist_ok=True)

    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]

    # 查待下载的
    query = {"is_used": False}
    total = col.count_documents(query)
    logger.info(f"待下载记录（is_used=False）：{total} 条"
                + (f"，本次只处理前 {limit} 条" if limit else ""))

    if total == 0:
        logger.info("没有需要下载的记录，结束")
        return

    # 拉取待下载列表
    cursor = col.find(query)
    if limit:
        cursor = cursor.limit(limit)
    docs = list(cursor)

    ok = fail = skip = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, doc): doc for doc in docs}

        for i, future in enumerate(as_completed(futures), 1):
            code, status, msg = future.result()

            if status == "ok":
                ok += 1
                logger.info(f"[{i}/{len(docs)}] ✓ {code} 成功 ({msg})")
            elif status == "skip":
                skip += 1
                logger.info(f"[{i}/{len(docs)}] ⊙ {code} 跳过 ({msg})")
            else:
                fail += 1
                logger.error(f"[{i}/{len(docs)}] ✗ {code} 失败 ({msg})")

            # 每20条打印一次进度
            if i % 20 == 0 or i == len(docs):
                elapsed = time.time() - start
                speed = i / elapsed if elapsed > 0 else 0
                remain = (len(docs) - i) / speed if speed > 0 else 0
                logger.info(f"--- 进度 {i}/{len(docs)}  "
                            f"成功{ok} 失败{fail} 跳过{skip}  "
                            f"速度{speed:.1f}条/秒  预计剩余{remain/60:.1f}分钟 ---")

    elapsed = (time.time() - start) / 60
    logger.info(f"\n===== 完成：成功 {ok}，失败 {fail}，跳过 {skip}，"
                f"耗时 {elapsed:.1f} 分钟 =====")

    # 汇总剩余
    remain = col.count_documents({"is_used": False})
    logger.info(f"MongoDB 剩余待下载：{remain} 条")


if __name__ == "__main__":
    main()
