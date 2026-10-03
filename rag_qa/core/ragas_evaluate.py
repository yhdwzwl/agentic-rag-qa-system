# -*- coding: utf-8 -*-
"""
RAGAS评估脚本 - 支持版本管理、防重复、断点续跑

功能：
1. 支持版本号：每次改RAG系统，改版本号，重新跑评估
2. 同版本防重复：同一个query+同一个版本号，不重复跑
3. 断点续跑：边跑边存MongoDB，下次接着跑
4. 存MongoDB：结果存在rag_eval_results集合
5. 输出报告：跑完按query类型、业务分类统计平均分
"""
import os
import sys
import time
import pandas as pd
from datetime import datetime

ROOT = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system"
sys.path.insert(0, ROOT)

# ========== 配置区 ==========
# ★ 每次改RAG系统，改这里的版本号！！！
MODEL_VERSION = "v1.0"
# 跑多少条query（先小批量测试，改大了就全量跑）
EVAL_COUNT = 10
# 是否全量重跑（True：不管之前跑过什么，全部重跑；False：只跑没跑过这个版本的）
FORCE_RERUN = False
# RAGAS评估用的模型（2026-10-03：不再写死；import conf 后从 config.ini [llm] table_llm_models 读取）
RAGAS_LLM_MODEL = None
RAGAS_EMBEDDING_MODEL = "text-embedding-v3"
# ============================

from pymongo import MongoClient
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from base.config import conf
# 从 config.ini 读取 RAGAS 判分模型（table_llm_models 优先，未配置回退主模型；不写死）
RAGAS_LLM_MODEL = (conf.TABLE_LLM_MODELS or [conf.LLM_MODEL] or ["qwen-max"])[0]

# 连接MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/")
queries_col = mongo_client["rag_doc_data"]["rag_queries"]
eval_col = mongo_client["rag_doc_data"]["rag_eval_results"]

# 全局RAG系统实例（避免重复初始化）
rag_system = None


def init_rag_system():
    """初始化RAG系统"""
    global rag_system
    if rag_system is not None:
        return rag_system

    from rag_qa.core.vector_store import VectorStore
    from rag_qa.core.new_rag_system import RAGSystem
    from langchain_openai import ChatOpenAI

    print("初始化RAG系统...")
    vs = VectorStore()

    llm = ChatOpenAI(
        model=conf.LLM_MODEL,
        api_key=conf.DASHSCOPE_API_KEY,
        base_url=conf.DASHSCOPE_BASE_URL,
    )

    rag_system = RAGSystem(vs, llm)
    print("RAG系统初始化完成")
    return rag_system


def load_test_queries(count=10, force_rerun=False, model_version="v1.0"):
    """加载测试query，跳过已经评估过的"""
    print(f"加载测试query...")

    # 查已经评估过的query ID（当前版本）
    if not force_rerun:
        evaluated_ids = set(
            doc["query_id"] for doc in eval_col.find({"model_version": model_version}, {"query_id": 1})
        )
        print(f"当前版本 {model_version} 已评估过 {len(evaluated_ids)} 条")
    else:
        evaluated_ids = set()
        print(f"全量重跑模式，忽略之前的评估记录")

    # 从rag_queries加载，跳过已评估的
    queries = []
    for q in queries_col.find({}):
        q_id = str(q["_id"])
        if q_id not in evaluated_ids:
            queries.append(q)
        if len(queries) >= count:
            break

    print(f"本次要跑 {len(queries)} 条测试query")
    return queries


def run_rag_system(query_text):
    """调用RAG系统，生成答案，同时拿中间产物"""
    rag = init_rag_system()

    # 收集流式输出
    answer_parts = []
    for chunk in rag.generate_answer(query_text):
        answer_parts.append(chunk)

    answer = "".join(answer_parts)

    # 从RAG系统拿中间产物
    last_result = rag.last_query_result

    return {
        "answer": answer,
        "contexts": last_result["contexts"],
        "retrieved_doc_ids": last_result["retrieved_doc_ids"],
        "retrieved_scores": last_result["retrieved_scores"],
        "retrieval_strategy": last_result["retrieval_strategy"],
        "rewritten_query": last_result["rewritten_query"],
        "classify_time": last_result["classify_time"],
        "rewrite_time": last_result["rewrite_time"],
        "cache_time": last_result["cache_time"],
        "retrieval_time": last_result["retrieval_time"],
        "rerank_time": last_result["rerank_time"],
        "generation_time": last_result["generation_time"],
        "total_time": last_result["total_time"],
        "cache_hit": last_result["cache_hit"],
        "cache_level": last_result["cache_level"],
        "fallback": last_result["fallback"],
    }


def save_eval_result(query_doc, rag_result, ground_truth):
    """把评估结果存到MongoDB"""
    eval_doc = {
        "query_id": str(query_doc["_id"]),
        "query_text": query_doc["query_text"],
        "generated_answer": rag_result["answer"],
        "contexts": rag_result["contexts"],
        "ground_truth": ground_truth,
        "retrieved_doc_ids": rag_result["retrieved_doc_ids"],
        "retrieved_scores": rag_result["retrieved_scores"],
        "retrieval_strategy": rag_result["retrieval_strategy"],
        "rewritten_query": rag_result["rewritten_query"],
        "classify_time": rag_result["classify_time"],
        "rewrite_time": rag_result["rewrite_time"],
        "cache_time": rag_result["cache_time"],
        "retrieval_time": rag_result["retrieval_time"],
        "rerank_time": rag_result["rerank_time"],
        "generation_time": rag_result["generation_time"],
        "total_time": rag_result["total_time"],
        "cache_hit": rag_result["cache_hit"],
        "cache_level": rag_result["cache_level"],
        "fallback": rag_result["fallback"],
        "query_type": query_doc.get("query_type", ""),
        "category": query_doc.get("category", ""),
        "model_version": MODEL_VERSION,
        "eval_time": datetime.now(),
    }
    eval_col.insert_one(eval_doc)


def main():
    print("=" * 60)
    print(f"RAG系统评估（RAGAS） - 版本: {MODEL_VERSION}")
    print("=" * 60)

    # 1. 加载测试query
    queries = load_test_queries(EVAL_COUNT, FORCE_RERUN, MODEL_VERSION)

    if len(queries) == 0:
        print("没有要跑的query，全部评估完了！")
        return

    # 2. 批量跑RAG系统
    print(f"\n开始跑RAG系统，共 {len(queries)} 条query...")
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    success_count = 0
    fail_count = 0

    for i, q in enumerate(queries, 1):
        query_text = q["query_text"]
        ground_truth = q.get("expected_answer", "")

        print(f"\n[{i}/{len(queries)}] 处理: {query_text[:30]}...")

        try:
            rag_result = run_rag_system(query_text)

            # 存到MongoDB
            save_eval_result(q, rag_result, ground_truth)

            # 收集到RAGAS数据集
            eval_data["question"].append(query_text)
            eval_data["answer"].append(rag_result["answer"])
            eval_data["contexts"].append(rag_result["contexts"])
            eval_data["ground_truth"].append(ground_truth)

            success_count += 1
            print(f"  ✓ 成功，总耗时: {rag_result['total_time']:.2f}s")

        except Exception as e:
            fail_count += 1
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 每跑10条打印进度
        if i % 10 == 0:
            print(f"\n--- 进度: {i}/{len(queries)} ({i/len(queries)*100:.1f}%), 成功{success_count}, 失败{fail_count} ---")

        time.sleep(0.5)  # 别把CPU跑爆

    print(f"\n跑完了！成功 {success_count} 条，失败 {fail_count} 条")

    if len(eval_data["question"]) == 0:
        print("没有成功跑的query，跳过RAGAS评估")
        return

    # 3. 跑RAGAS算指标
    print(f"\n开始RAGAS评估，共 {len(eval_data['question'])} 条...")
    dataset = Dataset.from_dict(eval_data)

    # 配置RAGAS模型
    llm = ChatTongyi(
        model=RAGAS_LLM_MODEL,
        api_key=conf.DASHSCOPE_API_KEY,
    )
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=conf.DASHSCOPE_API_KEY,
        model=RAGAS_EMBEDDING_MODEL,
    )

    # 执行评估
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm,
        embeddings=embeddings
    )

    # 4. 把RAGAS指标更新回MongoDB
    print("\n更新RAGAS指标到MongoDB...")
    for i, q in enumerate(queries[:len(eval_data["question"])]):
        q_id = str(q["_id"])
        eval_col.update_one(
            {"query_id": q_id, "model_version": MODEL_VERSION},
            {
                "$set": {
                    "faithfulness": result["faithfulness"][i] if "faithfulness" in result else 0.0,
                    "answer_relevancy": result["answer_relevancy"][i] if "answer_relevancy" in result else 0.0,
                    "context_precision": result["context_precision"][i] if "context_precision" in result else 0.0,
                    "context_recall": result["context_recall"][i] if "context_recall" in result else 0.0,
                }
            }
        )

    # 5. 输出结果
    print("\n" + "=" * 60)
    print(f"RAGAS评估结果 - 版本 {MODEL_VERSION}：")
    print("=" * 60)
    print(result)

    # 6. 统计报告：按query类型分组
    print("\n" + "=" * 60)
    print("按query类型统计平均分：")
    print("=" * 60)
    df = pd.DataFrame(list(eval_col.find({"model_version": MODEL_VERSION})))
    if len(df) > 0 and "query_type" in df.columns:
        print(df.groupby("query_type")[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())

    # 7. 保存结果文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(ROOT, f"ragas_results_{MODEL_VERSION}_{timestamp}.csv")
    df.to_csv(result_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {result_file}")

    mongo_client.close()
    print("\n评估完成！")


if __name__ == "__main__":
    main()
