#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
东方财富 沪深京A股公告采集器

从东方财富公告 JSON 接口抓取公告元数据（标题/代码/分类/art_code），
写入 MongoDB rag_doc_data.doc_url。PDF 下载由 new_df_crawl_pdf.py 单独负责。

断点续爬：每个分类抓够 target 条即停，重跑自动跳过已达标分类。
用法：python df_crawl.py
"""
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from mongoengine import connect, Document, StringField, BooleanField, DateTimeField
from mongoengine.errors import NotUniqueError

# ==================== 配置 ====================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "rag_doc_data"
COLLECTION_NAME = "doc_url"

API_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
ANN_TYPE = "SHA,CYB,SZA,BJA,INV"
PAGE_SIZE = 50

# f_node 由页面实测确认：5=重大事项 3=风险提示 2=融资公告 4=信息变更 7=持股变动
TAG_TARGETS = [
    {"f_node": 5, "pdf_type": "重大事项", "target": 1500},
    {"f_node": 3, "pdf_type": "风险提示", "target": 1000},
    {"f_node": 2, "pdf_type": "融资公告", "target": 800},
    {"f_node": 4, "pdf_type": "信息变更", "target": 1000},
    {"f_node": 7, "pdf_type": "持股变动", "target": 700},
]

REQUEST_INTERVAL = (0.8, 1.8)
MAX_RETRY = 3
MAX_PAGES = 1000

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Referer": "https://data.eastmoney.com/notices/",
}


# ==================== 数据模型 ====================
class DocUrl(Document):
    meta = {
        "collection": COLLECTION_NAME,
        "indexes": [
            {"fields": ("code", "art_code", "pdf_type"), "unique": True},
            {"fields": ("pdf_type",)},
            {"fields": ("is_used",)},
        ],
        "ordering": ["-craw_time"],
    }
    code = StringField(required=True)
    name = StringField()
    pdf_type = StringField(required=True)
    doc_type = StringField()
    pdf_source_url = StringField(required=True)
    art_code = StringField(required=True)
    pdf_time = StringField()
    title = StringField()
    craw_time = DateTimeField(default=datetime.now)
    is_used = BooleanField(default=False)
    source = StringField(default="eastmoney")
    created_time = DateTimeField(default=datetime.now)
    updated_time = DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        if not self.created_time:
            self.created_time = datetime.now()
        self.updated_time = datetime.now()
        return super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.code} {self.name} | {self.pdf_type} | {self.pdf_time}"


# ==================== 采集器 ====================
class EastMoneyCrawler:
    def __init__(self):
        connect(db=DB_NAME, host=MONGO_URI)

    def fetch_json(self, params):
        """请求公告接口并返回 JSON 的 data 部分；失败自动重试"""
        url = API_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        for attempt in range(1, MAX_RETRY + 1):
            try:
                req = Request(url, headers=HEADERS)
                with urlopen(req, timeout=15) as resp:
                    text = resp.read().decode("utf-8", "ignore")
                data = json.loads(text)
                if data.get("success") != 1:
                    raise ValueError(f"接口返回失败: {text[:200]}")
                return data.get("data", {})
            except (HTTPError, URLError, ValueError, json.JSONDecodeError) as e:
                print(f"  请求失败(第{attempt}/{MAX_RETRY}次): {e}")
                if attempt < MAX_RETRY:
                    time.sleep(2 * attempt)
        return None

    def build_doc(self, item, pdf_type):
        """把接口返回的一条公告解析成入库字段"""
        codes = item.get("codes") or []
        code = codes[0].get("stock_code", "") if codes else ""
        name = codes[0].get("short_name", "") if codes else ""
        art_code = item.get("art_code", "")
        columns = item.get("columns") or []
        doc_type = columns[0].get("column_name", "") if columns else ""
        return {
            "code": code,
            "name": name,
            "pdf_type": pdf_type,
            "doc_type": doc_type,
            "art_code": art_code,
            "pdf_source_url": f"https://data.eastmoney.com/notices/detail/{code}/{art_code}.html",
            "pdf_time": item.get("notice_date", ""),
            "title": item.get("title", ""),
        }

    def save_doc(self, fields):
        """入库（依赖唯一索引去重）"""
        try:
            DocUrl(**fields).save()
            return "saved"
        except NotUniqueError:
            return "dup"

    def crawl_type(self, cfg):
        """按分类翻页采集，抓够 target 条新数据即停。

        不预先查库算差值：重复的靠唯一索引 (code, art_code, pdf_type) 自动跳过。
        """
        pdf_type, f_node, target = cfg["pdf_type"], cfg["f_node"], cfg["target"]
        print(f"\n===== [{pdf_type}] f_node={f_node} 目标新增{target}条 =====")

        saved_total = 0
        dup_total = 0
        page_index = 1
        while saved_total < target and page_index <= MAX_PAGES:
            params = {
                "page_size": PAGE_SIZE,
                "page_index": page_index,
                "ann_type": ANN_TYPE,
                "client_source": "web",
                "f_node": f_node,
            }
            data = self.fetch_json(params)
            if data is None:
                print(f"[{pdf_type}] 第{page_index}页请求失败，跳过本页")
                page_index += 1
                continue
            items = data.get("list") or []
            if not items:
                print(f"[{pdf_type}] 第{page_index}页无数据，翻页结束")
                break

            page_saved = 0
            for item in items:
                if saved_total >= target:
                    break
                fields = self.build_doc(item, pdf_type)
                if not fields["code"] or not fields["art_code"]:
                    continue
                result = self.save_doc(fields)
                if result == "saved":
                    saved_total += 1
                    page_saved += 1
                elif result == "dup":
                    dup_total += 1

            print(f"[{pdf_type}] 第{page_index}页：本页新增{page_saved}，"
                  f"累计新增{saved_total}，去重跳过{dup_total}")
            page_index += 1
            if saved_total < target and page_index <= MAX_PAGES:
                time.sleep(random.uniform(*REQUEST_INTERVAL))

        print(f"[{pdf_type}] 完成：新增{saved_total}条，去重跳过{dup_total}条（目标{target}条）")
        return saved_total

    def crawl(self):
        """5个分类并行爬，每个分类一个线程，线程内随机间隔不触发反爬"""
        total = 0
        try:
            with ThreadPoolExecutor(max_workers=len(TAG_TARGETS)) as pool:
                futures = {pool.submit(self.crawl_type, cfg): cfg["pdf_type"]
                           for cfg in TAG_TARGETS}
                for fut in as_completed(futures):
                    pdf_type = futures[fut]
                    try:
                        total += fut.result()
                    except Exception as e:
                        print(f"[{pdf_type}] 线程异常: {e}")
        except KeyboardInterrupt:
            print("\n[提示] 收到中断，进度已保留在 MongoDB 中，可再次运行续爬")
        print(f"\n===== 全部完成：本次新增 {total} 条 =====")


if __name__ == "__main__":
    EastMoneyCrawler().crawl()
