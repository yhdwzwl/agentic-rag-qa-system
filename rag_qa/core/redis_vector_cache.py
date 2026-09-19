# -*- coding: utf-8 -*-
"""
Redis 向量缓存模块

功能：
- 用户query进来，先做向量相似度搜索
- 相似度 >= 阈值就直接返回缓存的上下文，跳过Milvus检索
- RAG跑完后自动回填缓存
- 自动判断高频：命中>=hot_threshold次升级为正式缓存，长TTL
- 自动淘汰：TTL过期 + LRU

依赖：redis-py + RediSearch 向量索引
"""
import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple

import redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from base import Config, logger


class RedisVectorCache:
    """RAG Query向量缓存"""

    def __init__(self):
        # 读配置
        conf = Config()
        self.host = conf.REDIS_HOST
        self.port = conf.REDIS_PORT
        self.password = conf.REDIS_PASSWORD or None
        self.db = conf.REDIS_DB

        self.index_name = conf.REDIS_VECTOR_CACHE_INDEX
        self.vector_dim = conf.REDIS_VECTOR_CACHE_DIM
        self.similarity_threshold = conf.REDIS_VECTOR_CACHE_SIMILARITY_THRESHOLD
        self.temporary_ttl = conf.REDIS_VECTOR_CACHE_TEMPORARY_TTL
        self.formal_ttl = conf.REDIS_VECTOR_CACHE_FORMAL_TTL
        self.hot_threshold = conf.REDIS_VECTOR_CACHE_HOT_THRESHOLD
        self.max_size = conf.REDIS_VECTOR_CACHE_MAX_SIZE

        # 连接Redis
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            db=self.db,
            decode_responses=False  # 向量要二进制存
        )
        self._ping()

        # 初始化索引
        self._create_index_if_not_exists()
        logger.info(f"Redis向量缓存初始化完成: {self.index_name}, 阈值={self.similarity_threshold}")

    def _ping(self):
        """测试连接"""
        try:
            self.client.ping()
            logger.info(f"Redis向量缓存连接成功: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise

    def _create_index_if_not_exists(self):
        """创建向量索引（如果不存在）"""
        try:
            # 先查索引是否存在
            self.client.ft(self.index_name).info()
            logger.info(f"RediSearch索引已存在: {self.index_name}")
            return
        except:
            # 不存在就创建
            schema = (
                VectorField(
                    "query_vector",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                    as_name="query_vector",
                ),
                TextField("query_text"),
                TextField("chunks_json"),
                TextField("parent_ids_json"),
                NumericField("hit_count"),
                NumericField("created_time"),
                NumericField("last_hit_time"),
                NumericField("is_formal"),  # 0=临时, 1=正式
            )

            self.client.ft(self.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH),
            )
            logger.info(f"RediSearch索引创建成功: {self.index_name}")

    def search_similar(self, query_vector: np.ndarray, top_k: int = 1) -> Optional[Dict]:
        """
        搜索最相似的query缓存
        返回: 命中的缓存内容，没命中返回None
        """
        # 把向量转成bytes
        query_vector_bytes = query_vector.astype(np.float32).tobytes()

        # 构造搜索query：KNN向量搜索
        q = (
            Query(f"*=>[KNN {top_k} @query_vector $vec AS score]")
            .sort_by("score")
            .paging(0, top_k)
            .dialect(2)
        )

        try:
            results = self.client.ft(self.index_name).search(
                q,
                query_params={"vec": query_vector_bytes},
            )
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return None

        if results.total == 0:
            return None

        # 取第一个结果
        doc = results.docs[0]
        # cosine距离转相似度：1 - distance
        score = 1 - float(doc.score)

        if score < self.similarity_threshold:
            logger.debug(f"最高相似度 {score:.3f} < 阈值 {self.similarity_threshold}，未命中")
            return None

        # 解析缓存内容
        cache_data = {
            "id": doc.id,
            "score": score,
            "query_text": doc.query_text,
            "chunks": json.loads(doc.chunks_json),
            "parent_ids": json.loads(doc.parent_ids_json),
            "hit_count": int(doc.hit_count),
            "is_formal": int(doc.is_formal),
        }

        # 更新命中次数和最后访问时间
        self._update_hit(cache_data["id"])

        logger.info(f"向量缓存命中 | 相似度={score:.3f} | 历史query='{doc.query_text[:30]}...'")
        return cache_data

    def save_cache(
        self,
        query_text: str,
        query_vector: np.ndarray,
        chunks: List[Dict],
        parent_ids: List[int],
    ):
        """
        RAG跑完后，把结果写入缓存
        自动判断是临时还是正式：
        - 第一次写入：临时缓存，短TTL
        - 之前命中过，累计到hot_threshold次：升级正式缓存，长TTL
        """
        # 先查一下之前有没有相似的
        existing = self.search_similar(query_vector, top_k=1)

        if existing:
            # 已经有缓存了，更新命中次数
            new_hit_count = existing["hit_count"] + 1
            new_is_formal = existing["is_formal"]

            # 命中次数够了，升级正式
            if new_hit_count >= self.hot_threshold and new_is_formal == 0:
                new_is_formal = 1
                # 延长TTL
                self.client.expire(existing["id"], self.formal_ttl)
                logger.info(f"升级为正式缓存 | 累计命中 {new_hit_count} 次")

            # 更新hit_count
            self.client.hset(existing["id"], mapping={
                "hit_count": new_hit_count,
                "last_hit_time": time.time(),
                "is_formal": new_is_formal,
            })
            return existing["id"]

        # 新写入：临时缓存
        cache_id = f"cache:{int(time.time() * 1000)}"
        cache_data = {
            "query_text": query_text,
            "query_vector": query_vector.astype(np.float32).tobytes(),
            "chunks_json": json.dumps(chunks, ensure_ascii=False),
            "parent_ids_json": json.dumps(parent_ids),
            "hit_count": 1,
            "created_time": time.time(),
            "last_hit_time": time.time(),
            "is_formal": 0,  # 临时
        }

        self.client.hset(cache_id, mapping=cache_data)
        # 设置临时TTL
        self.client.expire(cache_id, self.temporary_ttl)
        logger.debug(f"写入临时缓存 | query='{query_text[:30]}...' | TTL={self.temporary_ttl}s")
        return cache_id

    def _update_hit(self, cache_id: str):
        """更新命中时间"""
        self.client.hset(cache_id, "last_hit_time", time.time())

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        try:
            info = self.client.ft(self.index_name).info()
            num_docs = int(info.get("num_docs", 0))
            return {
                "total_cached": num_docs,
                "index_name": self.index_name,
                "similarity_threshold": self.similarity_threshold,
            }
        except:
            return {"total_cached": 0}


if __name__ == "__main__":
    # 简单测试
    cache = RedisVectorCache()
    print("缓存统计:", cache.get_stats())
