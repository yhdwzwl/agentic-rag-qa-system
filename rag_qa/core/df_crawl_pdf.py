#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通过 Firefox 命令行批量下载公告 PDF 到本地。

方案：Firefox 配置了 pdfjs.disabled=true，打开 PDF URL 时自动下载到 Downloads 目录。
用 subprocess.Popen 逐个打开 URL，检测下载完成后移动文件到 pdf_data。

PDF 直链规律：https://pdf.dfcfw.com/pdf/H2_{art_code}_1.pdf
（Python 直连被 WAF 按 TLS 指纹封禁，但真实浏览器 Firefox 不受影响）

流程：
    1. 读取 MongoDB（rag_doc_data / doc_url）中 is_used=False 的记录
    2. 对每条记录：
       a. subprocess.Popen([firefox_path, pdf_url]) 打开 PDF 直链
       b. 等 Downloads 目录出现新 .pdf 文件
       c. 移动重命名为 pdf_data/{code}_{art_code}.pdf，校验 %PDF 头
       d. 标记 is_used=True
    3. 失败保持 is_used=False，下次重跑自动重试

依赖：mongoengine
"""
import os
import time
import glob
import shutil
import subprocess
import argparse
from datetime import datetime

from mongoengine import connect, Document, StringField, BooleanField, DateTimeField

# ==================== 配置 ====================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "rag_doc_data"
COLLECTION_NAME = "doc_url"
PDF_SAVE_DIR = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system\pdf_data"
DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
FIREFOX_PATH = r"C:\Users\旺仔\AppData\Local\Mozilla Firefox\firefox.exe"

# 等待时间（秒）
WAIT_DOWNLOAD = 60              # 下载完成超时（原来15秒太短）
BETWEEN = 1.5                   # 两条之间间隔
STABLE_CHECK_ROUNDS = 3         # 文件大小连续几次不变才算稳定
STABLE_CHECK_INTERVAL = 1.0     # 每次检查间隔（秒）


# ==================== 数据模型 ====================
class DocUrl(Document):
    meta = {"collection": COLLECTION_NAME, "ordering": ["-craw_time"]}
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


# ==================== 下载工具 ====================
def open_pdf_url(art_code):
    """用 Firefox 命令行打开 PDF 直链（配置为自动下载）"""
    url = f"https://pdf.dfcfw.com/pdf/H2_{art_code}_1.pdf"
    subprocess.Popen([FIREFOX_PATH, url])


def wait_for_download(timeout=WAIT_DOWNLOAD):
    """等 Downloads 目录出现新的 .pdf 文件，确认大小稳定后返回文件路径"""
    start_time = time.time()
    last_size = -1
    stable_count = 0

    while time.time() - start_time < timeout:
        # 找 Downloads 里最新的 PDF 文件
        pdfs = glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf"))
        if not pdfs:
            time.sleep(STABLE_CHECK_INTERVAL)
            continue

        newest = max(pdfs, key=os.path.getmtime)
        mtime = os.path.getmtime(newest)

        # 只看本次运行后新下载的文件
        if mtime < start_time:
            time.sleep(STABLE_CHECK_INTERVAL)
            continue

        current_size = os.path.getsize(newest)

        # 检查大小是否稳定
        if current_size == last_size and current_size > 0:
            stable_count += 1
        else:
            stable_count = 0
            last_size = current_size

        # 大小连续几次不变，且是有效 PDF
        if stable_count >= STABLE_CHECK_ROUNDS:
            try:
                with open(newest, "rb") as f:
                    if f.read(4) == b"%PDF":
                        return newest
            except Exception:
                pass

        time.sleep(STABLE_CHECK_INTERVAL)
    return None


# ==================== 单条处理 ====================
def process_one(doc):
    """处理一条：打开PDF直链→等下载→移文件→标记"""
    code = doc.code
    art_code = doc.art_code
    save_path = os.path.join(PDF_SAVE_DIR, f"{code}_{art_code}.pdf")

    # 文件已存在 = 之前下过，直接标记跳过
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
        doc.update(is_used=True, updated_time=datetime.now())
        return "skip"

    # 1. 打开 PDF 直链（Firefox 自动下载）
    open_pdf_url(art_code)

    # 2. 等下载完成
    downloaded = wait_for_download()
    if not downloaded:
        # 超时后再兜底检查一次：看看 Downloads 里有没有刚下载的新文件
        print(f"  [warn] {code} 下载超时，兜底检查...", flush=True)
        time.sleep(2)
        downloaded = wait_for_download(timeout=10)
        if not downloaded:
            print(f"  [warn] {code} 下载超时（兜底也没找到）", flush=True)
            return "fail"

    # 3. 移动文件
    shutil.move(downloaded, save_path)

    # 4. 标记 is_used=True
    doc.update(is_used=True, updated_time=datetime.now())

    return "ok"


# ==================== 主流程 ====================
def main(limit=None):
    os.makedirs(PDF_SAVE_DIR, exist_ok=True)
    connect(db=DB_NAME, host=MONGO_URI)

    qs = DocUrl.objects(is_used=False)
    total = qs.count()
    print(f"待处理记录（is_used=False）：{total} 条"
          + (f"，本次只处理前 {limit} 条" if limit else ""))
    if total == 0:
        print("没有需要下载的记录，结束")
        return

    docs = list(qs.limit(limit) if limit else qs)
    ok = fail = skip = 0
    start = time.time()

    for i, doc in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] {doc.code} {doc.name} [{doc.pdf_type}]", flush=True)
        try:
            result = process_one(doc)
        except Exception as e:
            print(f"  [异常] {type(e).__name__}: {e}", flush=True)
            result = "fail"

        if result == "ok":
            ok += 1
            print(f"  ✓ 成功", flush=True)
        elif result == "skip":
            skip += 1
        else:
            fail += 1
            print(f"  ✗ 失败（保持 is_used=False，重跑自动重试）", flush=True)

        if i % 10 == 0 or i == len(docs):
            elapsed = (time.time() - start) / 60
            rate = i / (time.time() - start) if time.time() > start else 0
            remain = (len(docs) - i) / rate / 60 if rate > 0 else 0
            print(f"  --- 进度 {i}/{len(docs)}  成功{ok} 失败{fail} 跳过{skip}  "
                  f"已用{elapsed:.1f}分钟 预计剩余{remain:.0f}分钟 ---", flush=True)

        time.sleep(BETWEEN)

    print(f"\n===== 完成：成功 {ok}，失败 {fail}，跳过 {skip}，"
          f"耗时 {(time.time() - start) / 60:.1f} 分钟 =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通过 Firefox 命令行下载公告 PDF")
    parser.add_argument("--limit", type=int, default=None,
                        help="最多处理多少条（默认全部）")
    args = parser.parse_args()
    main(limit=args.limit)
