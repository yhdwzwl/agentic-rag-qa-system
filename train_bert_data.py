# -*- coding: utf-8 -*-
"""
BERT二分类模型训练数据生成脚本

任务：判断用户query是【通用闲聊】还是【股票财经相关】
正样本(label=1): 股票财经相关的query
负样本(label=0): 通用无关的query

用法：
  1. 填好下方的 API_URL 和 API_KEY
  2. 运行：python train_bert_data.py
"""
import os
import sys
import time
import random
from datetime import datetime

# ========== 配置区（用户填充） ==========
API_URL = "https://api.example.com/v1/chat/completions"  # 大模型API地址
API_KEY = "your_api_key_here"  # 大模型API密钥
MODEL_NAME = "gpt-4o-mini"  # 模型名称

# 生成配置
TOTAL_COUNT = 5000  # 总条数
POS_RATIO = 0.5  # 正样本比例（财经相关）
BATCH_SIZE = 50  # 每次调用生成多少条

# MongoDB配置
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "rag_doc_data"
MONGO_COL = "bert_2_data"
# ======================================

import pymongo
import requests


# 正负样本比例
POS_COUNT = int(TOTAL_COUNT * POS_RATIO)
NEG_COUNT = TOTAL_COUNT - POS_COUNT

# 财经相关的query分类（基于现有公告类型）
FINANCE_CATEGORIES = [
    "融资公告", "持股变动", "重大事项", "风险提示", "信息变更",
    "股票分析", "财报解读", "股东减持", "股权激励", "并购重组",
    "业绩预告", "分红派息", "停牌复牌", "退市风险", "解禁上市",
]

# 通用闲聊的场景
GENERAL_SCENES = [
    "日常生活", "美食", "旅游", "天气", "电影", "音乐",
    "健康养生", "教育学习", "体育赛事", "娱乐八卦",
    "科技数码", "购物消费", "情感咨询", "职场工作", "历史文化",
]


def generate_prompt(is_positive: bool, batch_size: int) -> str:
    """生成prompt"""
    if is_positive:
        category = random.choice(FINANCE_CATEGORIES)
        prompt = f"""你是一个金融领域数据标注专家。请生成 {batch_size} 条【股票财经相关】的用户查询query。

要求：
1. 这些query应该是真实用户在搜索引擎或问答系统里会输入的口语化问题
2. 主题围绕：{category}
3. 每条query长度10-50字
4. 要多样化：包括问事实、问解释、问影响、问操作建议等不同类型
5. 要口语化，不要太书面

请严格按以下JSON格式返回，不要有多余解释：
{{
  "queries": [
    {{"text": "query内容", "category": "{category}"}},
    ...
  ]
}}

现在生成{batch_size}条："""
    else:
        scene = random.choice(GENERAL_SCENES)
        prompt = f"""你是一个数据标注专家。请生成 {batch_size} 条【与股票财经完全无关】的用户查询query。

要求：
1. 这些query应该是真实用户在日常生活中会问的问题
2. 主题围绕：{scene}
3. 每条query长度10-50字
4. 要多样化：包括各种日常生活场景
5. 绝对不能出现股票、基金、A股、财报、涨停等任何金融相关词汇

请严格按以下JSON格式返回，不要有多余解释：
{{
  "queries": [
    {{"text": "query内容", "category": "{scene}"}},
    ...
  ]
}}

现在生成{batch_size}条："""

    return prompt


def call_llm(prompt: str) -> list:
    """调用大模型API，返回解析后的queries列表"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # 解析JSON
    import json
    # 去掉可能的markdown代码块标记
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]
        content = content.strip()
    if content.start("json"):
        content = content[4:].strip()

    result = json.loads(content)
    return result["queries"]


def save_to_mongo(data_list: list, label: int):
    """保存到MongoDB"""
    client = pymongo.MongoClient(MONGO_URI)
    col = client[MONGO_DB][MONGO_COL]

    docs = []
    for item in data_list:
        docs.append({
            "text": item["text"],
            "label": label,  # 1=财经相关, 0=通用无关
            "category": item.get("category", ""),
            "source": "llm_generated",
            "created_time": datetime.now(),
        })

    col.insert_many(docs)
    client.close()
    return len(docs)


def main():
    print(f"=== 训练数据生成计划 ===")
    print(f"  总条数: {TOTAL_COUNT}")
    print(f"  正样本(财经): {POS_COUNT} 条")
    print(f"  负样本(通用): {NEG_COUNT} 条")
    print(f"  每批生成: {BATCH_SIZE} 条")
    print()

    client = pymongo.MongoClient(MONGO_URI)
    col = client[MONGO_DB][MONGO_COL]
    pos_done = col.count_documents({"label": 1})
    neg_done = col.count_documents({"label": 0})
    client.close()

    print(f"已完成: 正样本 {pos_done}/{POS_COUNT}, 负样本 {neg_done}/{NEG_COUNT}")
    print()

    # 生成正样本
    while pos_done < POS_COUNT:
        batch = min(BATCH_SIZE, POS_COUNT - pos_done)
        print(f"[正样本] 生成第 {pos_done+1}-{pos_done+batch} 条...")

        try:
            prompt = generate_prompt(is_positive=True, batch_size=batch)
            queries = call_llm(prompt)
            saved = save_to_mongo(queries, label=1)
            pos_done += saved
            print(f"  ✓ 成功生成 {saved} 条，累计 {pos_done}/{POS_COUNT}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            time.sleep(2)

        time.sleep(0.5)

    # 生成负样本
    while neg_done < NEG_COUNT:
        batch = min(BATCH_SIZE, NEG_COUNT - neg_done)
        print(f"[负样本] 生成第 {neg_done+1}-{neg_done+batch} 条...")

        try:
            prompt = generate_prompt(is_positive=False, batch_size=batch)
            queries = call_llm(prompt)
            saved = save_to_mongo(queries, label=0)
            neg_done += saved
            print(f"  ✓ 成功生成 {saved} 条，累计 {neg_done}/{NEG_COUNT}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            time.sleep(2)

        time.sleep(0.5)

    print(f"\n===== 完成！ =====")
    print(f"  正样本: {pos_done} 条")
    print(f"  负样本: {neg_done} 条")
    print(f"  总计: {pos_done + neg_done} 条")
    print(f"  存储位置: {MONGO_DB}.{MONGO_COL}")


if __name__ == "__main__":
    main()
