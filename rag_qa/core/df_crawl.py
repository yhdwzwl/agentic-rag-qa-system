#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
东方财富 沪深京A股公告采集器（完成版）

设计说明：
1. 不使用 selenium / pymongo：
   - 数据抓取：标准库 urllib 直接调用东方财富公告 JSON 接口（无需浏览器驱动、无需 requests）
   - 数据存储：mongoengine（MongoDB ODM），数据库 rag_doc_data / 集合 doc_url
2. 按需求用 subprocess 打开本机 Firefox（仅打开做展示，不做自动化控制）
3. 标签编号已实测校正：f_node=2 才是“融资公告”（原代码/草稿里的 6 是“资产重组”）

唯一依赖：mongoengine（pip install mongoengine），需要本机 MongoDB 已启动。

用法：
    python df_crawl.py            # 正式采集（打开 Firefox + 写入 MongoDB）
    python df_crawl.py --dry-run  # 试运行：每个类型抓 1 页并打印解析结果，不写库、不开浏览器
"""
import os
import json
import time
import random
import argparse
import subprocess
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ==================== 配置 ====================
# Firefox 路径（按需求直接用 subprocess 打开本机浏览器）
FIREFOX_PATH = r"C:\Users\旺仔\AppData\Local\Mozilla Firefox\firefox.exe"

# MongoDB 配置
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "rag_doc_data"          # 数据库名
COLLECTION_NAME = "doc_url"       # 集合名

# 东方财富公告接口
API_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
ANN_TYPE = "SHA,CYB,SZA,BJA,INV"  # 沪深京 A 股
PAGE_SIZE = 50                    # 每页条数

# f_node 标签编号 -> 类型名 -> 目标条数
# f_node 由页面 data-value 实测确认：5=重大事项 3=风险提示 2=融资公告 4=信息变更 7=持股变动
TAG_TARGETS = [
    {"f_node": 5, "pdf_type": "重大事项", "target": 1500},
    {"f_node": 3, "pdf_type": "风险提示", "target": 1000},
    {"f_node": 2, "pdf_type": "融资公告", "target": 800},
    {"f_node": 4, "pdf_type": "信息变更", "target": 1000},
    {"f_node": 7, "pdf_type": "持股变动", "target": 700},
]

REQUEST_INTERVAL = (0.8, 1.8)     # 每页请求间隔（秒），随机，避免请求过快
MAX_RETRY = 3                     # 单页请求失败重试次数
MAX_PAGES = 1000                  # 单个类型最多翻页上限（防止死循环）

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Referer": "https://data.eastmoney.com/notices/",
}


# ==================== 数据模型（mongoengine）====================
# mongoengine 是唯一第三方依赖：未安装时进入降级模式（dry-run 仍可运行）
try:
    from mongoengine import connect, Document, StringField, BooleanField, DateTimeField
    from mongoengine.errors import NotUniqueError

    class DocUrl(Document):
        meta = {
            "collection": COLLECTION_NAME,
            "indexes": [
                # 同一公司 + 同一公告 + 同一标签唯一：去重，同时允许同一公告跨标签入库
                {"fields": ("code", "art_code", "pdf_type"), "unique": True},
                {"fields": ("pdf_type",)},
                {"fields": ("is_used",)},
            ],
            "ordering": ["-craw_time"],
        }
        code = StringField(required=True)            # 股票代码
        name = StringField()                         # 股票名称
        pdf_type = StringField(required=True)        # 公告大类（重大事项/风险提示/融资公告/信息变更/持股变动）
        doc_type = StringField()                     # 公告类型（具体类型，如 投资理财/股东减持）
        pdf_source_url = StringField(required=True)  # 公告标题链接（东财详情页 URL）
        art_code = StringField(required=True)        # 公告唯一编号（东财 art_code）
        pdf_time = StringField()                     # 公告日期
        title = StringField()                        # 公告标题
        craw_time = DateTimeField(default=datetime.now)  # 入库时的系统时间
        is_used = BooleanField(default=False)        # 是否已被使用，默认 False
        source = StringField(default="eastmoney")    # 数据来源
        created_time = DateTimeField(default=datetime.now)
        updated_time = DateTimeField(default=datetime.now)

        def save(self, *args, **kwargs):
            if not self.created_time:
                self.created_time = datetime.now()
            self.updated_time = datetime.now()
            return super(DocUrl, self).save(*args, **kwargs)

        def __str__(self):
            return f"{self.code} {self.name} | {self.pdf_type} | {self.pdf_time}"

    MONGO_ENGINE_OK = True
except ImportError:
    MONGO_ENGINE_OK = False
    DocUrl = None
    NotUniqueError = Exception


# ==================== 采集器 ====================
class EastMoneyCrawler:
    def __init__(self, dry_run=False, max_pages=None):
        self.dry_run = dry_run
        self.max_pages = max_pages if max_pages else MAX_PAGES
        self.firefox_process = None
        if dry_run:
            print("[dry-run] 试运行模式：不写库、不开浏览器")
            return
        if not MONGO_ENGINE_OK:
            raise SystemExit("[错误] 未安装 mongoengine，请先执行：pip install mongoengine")
        connect(db=DB_NAME, host=MONGO_URI)
        self.firefox_process = self.open_firefox()

    # ---------- 按需求：subprocess 打开本机 Firefox ----------
    def open_firefox(self):
        if not os.path.exists(FIREFOX_PATH):
            print(f"[提示] 未找到 Firefox：{FIREFOX_PATH}，跳过打开浏览器（不影响采集）")
            return None
        try:
            # 打开第一个分类页，方便在浏览器里人工观察
            start_url = "https://data.eastmoney.com/notices/hsa/5.html"
            process = subprocess.Popen([FIREFOX_PATH, start_url])
            print(f"[提示] 已启动 Firefox（pid={process.pid}），采集结束后自动关闭")
            return process
        except Exception as e:
            print(f"[提示] 打开 Firefox 失败: {e}（不影响采集）")
            return None

    def close_firefox(self):
        if self.firefox_process:
            try:
                self.firefox_process.terminate()
                print("[提示] 已关闭 Firefox")
            except Exception:
                pass

    # ---------- 接口请求 ----------
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

    # ---------- 数据解析 ----------
    def build_doc(self, item, pdf_type):
        """把接口返回的一条公告解析成入库字段（纯字典，方便 dry-run 打印）"""
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
            # 公告标题链接（东财详情页 URL，由 代码 + art_code 拼出）
            "pdf_source_url": f"https://data.eastmoney.com/notices/detail/{code}/{art_code}.html",
            "pdf_time": item.get("notice_date", ""),
            "title": item.get("title", ""),
        }

    def save_doc(self, fields):
        """入库（依赖唯一索引去重）"""
        try:
            doc = DocUrl(**fields)
            doc.save()
            return "saved"
        except NotUniqueError:
            return "dup"

    # ---------- 单类型采集 ----------
    def crawl_type(self, cfg):
        pdf_type, f_node, target = cfg["pdf_type"], cfg["f_node"], cfg["target"]

        # 断点续爬：已有数量（dry-run 不查库）
        existing = 0
        needed = target
        if not self.dry_run:
            existing = DocUrl.objects(pdf_type=pdf_type).count()
            needed = target - existing
        print(f"\n===== [{pdf_type}] f_node={f_node} 目标{target}条 "
              f"库中已有{existing}条 还需{needed}条 =====")
        if needed <= 0:
            print(f"[{pdf_type}] 已达标，跳过")
            return 0

        saved_total = 0
        dup_total = 0
        page_index = 1
        while saved_total < needed and page_index <= self.max_pages:
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
                if saved_total >= needed:
                    break
                fields = self.build_doc(item, pdf_type)
                if not fields["code"] or not fields["art_code"]:
                    continue  # 缺关键字段，跳过
                if self.dry_run:
                    print("  示例:", json.dumps(fields, ensure_ascii=False))
                    page_saved += 1
                    continue
                result = self.save_doc(fields)
                if result == "saved":
                    saved_total += 1
                    page_saved += 1
                elif result == "dup":
                    dup_total += 1

            if self.dry_run:
                break  # 试运行只抓 1 页

            print(f"[{pdf_type}] 第{page_index}页：本页新增{page_saved}，"
                  f"累计新增{saved_total}，去重跳过{dup_total}")
            page_index += 1
            if saved_total < needed and page_index <= self.max_pages:
                time.sleep(random.uniform(*REQUEST_INTERVAL))

        print(f"[{pdf_type}] 完成：新增{saved_total}条，去重跳过{dup_total}条（目标{target}条）")
        return saved_total

    # ---------- 主流程 ----------
    def crawl(self):
        total = 0
        try:
            for cfg in TAG_TARGETS:
                total += self.crawl_type(cfg)
        except KeyboardInterrupt:
            print("\n[提示] 收到中断，进度已保留在 MongoDB 中（已入库的不重复），可再次运行续爬")
        finally:
            if not self.dry_run:
                self.close_firefox()
        print(f"\n===== 全部完成：本次新增 {total} 条 =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="东方财富沪深京A股公告采集器")
    parser.add_argument("--dry-run", action="store_true",
                        help="试运行：每个类型只抓1页并打印解析结果，不写库、不开浏览器")
    parser.add_argument("--pages", type=int, default=None,
                        help="每个类型最多翻页数（默认不限，用于小批量测试，如 --pages 1）")
    args = parser.parse_args()

    crawler = EastMoneyCrawler(dry_run=args.dry_run, max_pages=args.pages)
    crawler.crawl()
