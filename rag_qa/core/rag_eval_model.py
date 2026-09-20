# -*- coding: utf-8 -*-
"""
RAG评估结果数据模型 - MongoDB
"""
from datetime import datetime
from mongoengine import (
    Document, StringField, ListField, FloatField,
    BooleanField, DateTimeField, IntField
)


class RagEvalResult(Document):
    """RAG评估结果模型"""
    meta = {
        'collection': 'rag_eval_results',
        'indexes': [
            'query_type',
            'category',
            'eval_time',
            'model_version',
        ]
    }

    # ========== 基础字段（RAGAS必须） ==========
    query_text = StringField(required=True)  # 原始用户问题
    generated_answer = StringField(default="")  # RAG系统生成的答案
    contexts = ListField(StringField())  # 检索到的上下文列表
    ground_truth = StringField(default="")  # 标准答案

    # ========== 检索中间产物 ==========
    retrieved_doc_ids = ListField(StringField())  # 召回的top k个文档ID
    retrieved_scores = ListField(FloatField())  # 每个文档的相似度分数
    retrieval_strategy = StringField(default="")  # 用了什么策略（direct/hyde/subquery）
    rewritten_query = StringField(default="")  # 改写后的query

    # ========== 性能指标（每一步单独记耗时） ==========
    classify_time = FloatField(default=0.0)  # 查询分类耗时（秒）
    rewrite_time = FloatField(default=0.0)  # query改写耗时（秒）
    cache_time = FloatField(default=0.0)  # 缓存查询耗时（秒）
    retrieval_time = FloatField(default=0.0)  # 混合检索耗时（秒）
    rerank_time = FloatField(default=0.0)  # 精排耗时（秒）
    generation_time = FloatField(default=0.0)  # 答案生成耗时（秒）
    total_time = FloatField(default=0.0)  # 总耗时（秒）

    # ========== 缓存与降级 ==========
    cache_hit = BooleanField(default=False)  # 有没有命中缓存
    cache_level = StringField(default="")  # 临时缓存/正式缓存/未命中
    fallback = BooleanField(default=False)  # 有没有降级

    # ========== 评估指标（RAGAS算完后填） ==========
    faithfulness = FloatField(default=0.0)  # 忠实度
    answer_relevancy = FloatField(default=0.0)  # 答案相关性
    context_precision = FloatField(default=0.0)  # 上下文精确率
    context_recall = FloatField(default=0.0)  # 上下文召回率

    # ========== 元数据 ==========
    query_type = StringField(default="")  # 测试query类型（simple/direct/multi_hop/hypothetical）
    category = StringField(default="")  # 业务分类（融资公告/持股变动等）
    eval_time = DateTimeField(default=datetime.now)  # 评估时间
    model_version = StringField(default="v1.0")  # 用的哪个版本的RAG系统

    def save(self, *args, **kwargs):
        self.eval_time = datetime.now()
        return super(RagEvalResult, self).save(*args, **kwargs)

    def __str__(self):
        return f"[{self.eval_time.strftime('%Y-%m-%d %H:%M')}] {self.query_text[:30]}..."
