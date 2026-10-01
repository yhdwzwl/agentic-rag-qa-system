# -*- coding: utf-8 -*-
"""
Redis 向量缓存模块（v2，2026-10-01 重构）

功能：
- 用户（改写后）query 进来，先做向量相似度搜索
- 相似度 >= 阈值就直接返回缓存的上下文（chunks + parent_ids + source_metas），
  跳过 Milvus 混合检索 + Rerank（命中后仍由主流程调 LLM 生成答案，不缓存答案）
- RAG 跑完后自动回填缓存（只回填上下文，不含答案）
- 临时/正式两级缓存：每次真实命中 hit_count+1，累计 >= hot_threshold 升级为正式缓存（长 TTL）
- 淘汰以 TTL 为主：临时 temporary_ttl、正式 formal_ttl；临时命中可按配置滑动续期
- 命中率统计：Redis HASH cache:stats（hits/misses/upgraded/cache_time_ms 累计）+ MySQL 日表 redis_cache_stats

依赖：redis-py + RediSearch 向量索引
"""
import json
import time
import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple

import redis
from redis.commands.search.field import VectorField, TextField, NumericField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.commands.search.aggregation import AggregateRequest
from redis.commands.search.reducers import count as _ft_count

from base import Config, logger


def _decode(value) -> str:
    """bytes -> str；None -> ''。模块 decode_responses=False，FT 文本字段返回 bytes"""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class RedisVectorCache:
    """RAG Query向量缓存（v2）"""

    # 缓存条目的 source_filter 统一归一值：None / 空 / "ALL" 都表示不过滤
    FILTER_ALL = "ALL"

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
        self.sliding_ttl_enabled = getattr(conf, "SLIDING_TTL_ENABLED", True)
        self.stats_key = "cache_meta:stats"              # 注意：必须避开索引 prefix "cache:"，否则统计 HASH 会被 RediSearch 当成 1 个文档索引，污染 total_cached
        self.stats_prev_key = "cache_meta:stats:prev"
        self.stats_date_key = "cache_meta:stats:last_sync_date"

        # 连接Redis
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            db=self.db,
            decode_responses=False  # 向量要二进制存
        )
        self._ping()

        # 初始化索引（字段缺失时自动重建，当前生产索引为空库，重建零成本）
        self._create_index_if_not_exists()
        logger.info(f"Redis向量缓存初始化完成: {self.index_name}, 阈值={self.similarity_threshold}, "
                    f"hot_threshold={self.hot_threshold}")

    def _ping(self):
        """测试连接"""
        try:
            self.client.ping()
            logger.info(f"Redis向量缓存连接成功: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise

    # ---------- 索引 ----------

    def _schema(self):
        return (
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
            TextField("rewritten_query"),
            TextField("chunks_json"),
            TextField("parent_ids_json"),
            TextField("source_metas_json"),
            TagField("source_filter"),
            NumericField("hit_count"),
            NumericField("created_time"),
            NumericField("last_hit_time"),
            NumericField("is_formal"),  # 0=临时, 1=正式
        )

    def _create_index_if_not_exists(self):
        """创建向量索引（如果不存在）；存在但缺新字段则 DROP 重建（保证 schema 版本一致）"""
        try:
            info = self.client.ft(self.index_name).info()
        except Exception:
            info = None

        if info is not None:
            # 检查已有索引是否包含 v2 必需字段（缺任一则重建）
            attrs = info.get("attributes", [])
            attr_names = set()
            for attr in attrs:
                if isinstance(attr, (list, tuple)):
                    # FT.INFO attributes 为扁平 kv 列表：[b'identifier', name, b'attribute', name, ...]
                    name = None
                    for i in range(0, len(attr) - 1, 2):
                        if _decode(attr[i]) == "identifier":
                            name = _decode(attr[i + 1])
                            break
                    if name:
                        attr_names.add(name)
                elif isinstance(attr, dict):
                    attr_names.add(_decode(attr.get("identifier", "")))
            required = {"query_vector", "query_text", "rewritten_query", "chunks_json",
                        "parent_ids_json", "source_metas_json", "source_filter",
                        "hit_count", "created_time", "last_hit_time", "is_formal"}
            if required.issubset(attr_names):
                logger.info(f"RediSearch索引已存在且 schema 完整: {self.index_name}")
                return
            # schema 缺字段：重建（缓存可重建，DROP 安全）
            logger.warning(f"RediSearch索引 {self.index_name} 缺 v2 字段，重建索引")
            try:
                self.client.ft(self.index_name).dropindex()
            except Exception as e:
                logger.warning(f"DROP 旧索引失败（忽略，继续创建）: {e}")

        schema = self._schema()
        self.client.ft(self.index_name).create_index(
            fields=schema,
            definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH),
        )
        logger.info(f"RediSearch索引创建成功: {self.index_name}")

    # ---------- 内部工具 ----------

    @staticmethod
    def _escape_tag(value: str) -> str:
        """TAG 字段过滤值转义（RediSearch 特殊字符）"""
        return (value.replace("\\", "\\\\")
                     .replace(",", "\\,")
                     .replace("{", "\\{")
                     .replace("}", "\\}"))

    @staticmethod
    def _normalize_filter(source_filter) -> str:
        """source_filter 归一：None/空 -> 'ALL'；去首尾空白（防 UI 空格产生重复缓存键）"""
        if not source_filter:
            return RedisVectorCache.FILTER_ALL
        return str(source_filter).strip()

    @staticmethod
    def _cache_id(rewritten_query: str, source_filter: str) -> str:
        """确定性 id：md5(改写后query + 分隔符 + source_filter)。
        文本而非向量字节：确定性 + 幂等覆盖 + 不依赖浮点稳定性；语义去重交给向量检索。"""
        raw = f"{rewritten_query}\x1f{source_filter}"
        return f"cache:{hashlib.md5(raw.encode('utf-8')).hexdigest()}"

    def _raw_search(self, query_vector: np.ndarray, top_k: int = 1,
                    source_filter=None) -> Optional[Tuple[object, float]]:
        """纯 KNN 搜索，不做任何统计/命中副作用。返回 (doc, similarity_score) 或 None。
        内部供 search_similar / save_cache 共用，避免 save 路径污染 hits 统计。"""
        query_vector_bytes = query_vector.astype(np.float32).tobytes()
        tag = self._normalize_filter(source_filter)
        # 一律按 TAG 过滤：None 归一为 ALL，只命中 ALL 标签条目（隔离，防过滤串用）
        knn = f"@source_filter:{{{self._escape_tag(tag)}}}=>[KNN {top_k} @query_vector $vec AS score]"
        q = (
            Query(knn)
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
            # 异常上抛，由调用方区分"异常降级"与"真实未命中"（异常不计入 misses 统计）
            logger.error(f"向量搜索失败: {e}")
            raise

        if results.total == 0:
            return None
        doc = results.docs[0]
        # cosine 距离转相似度：1 - distance
        score = 1 - float(doc.score)
        if score < self.similarity_threshold:
            return None
        return doc, score

    def _hit_fields(self, doc, score: float) -> Dict:
        """把 FT 文档解析为返回 dict（文本字段统一 decode，消除 bytes 污染）"""
        return {
            "id": _decode(doc.id),
            "score": score,
            "query_text": _decode(getattr(doc, "query_text", "")),
            "rewritten_query": _decode(getattr(doc, "rewritten_query", "")),
            "chunks": json.loads(_decode(getattr(doc, "chunks_json", "[]"))),
            "parent_ids": json.loads(_decode(getattr(doc, "parent_ids_json", "[]"))),
            "source_metas": json.loads(_decode(getattr(doc, "source_metas_json", "[]"))),
            "hit_count": int(_decode(getattr(doc, "hit_count", "0")) or 0),
            "is_formal": int(_decode(getattr(doc, "is_formal", "0")) or 0),
        }

    def _promote_if_hot(self, cache_id: str, new_hit_count: int, is_formal: int) -> bool:
        """临时且累计命中 >= hot_threshold：升级正式 + 延长 TTL。返回是否发生升级"""
        if is_formal == 0 and new_hit_count >= self.hot_threshold:
            try:
                self.client.hset(cache_id, "is_formal", 1)
                self.client.expire(cache_id, self.formal_ttl)
                self.client.hincrby(self.stats_key, "upgraded", 1)
                logger.info(f"升级为正式缓存 | 累计命中 {new_hit_count} 次 | {cache_id}")
            except Exception as e:
                logger.warning(f"升级正式缓存失败（忽略）: {e}")
            return True
        return False

    # ---------- 对外接口 ----------

    def search_similar(self, query_vector: np.ndarray, top_k: int = 1,
                       source_filter=None) -> Optional[Dict]:
        """
        搜索最相似的 query 缓存（含命中副作用：hit_count+1、升级判断、TTL 滑动、hits 统计）
        返回: 命中的缓存内容，没命中返回 None；异常也返回 None（缓存绝不阻断主流程）
        """
        start = time.time()
        try:
            found = self._raw_search(query_vector, top_k=top_k, source_filter=source_filter)
        except Exception as e:
            # 防御：_raw_search 之外仍可能有意外异常，一律降级为未命中
            logger.warning(f"向量缓存查询异常，跳过缓存: {e}")
            return None
        finally:
            elapsed_ms = (time.time() - start) * 1000.0
            try:
                self.client.hincrbyfloat(self.stats_key, "cache_time_ms", elapsed_ms)
            except Exception:
                pass

        if found is None:
            # 未命中（含 total=0 与 score<threshold；搜索异常不在此列，已提前 return）
            try:
                self.client.hincrby(self.stats_key, "misses", 1)
            except Exception:
                pass
            logger.debug(f"向量缓存未命中 | 阈值={self.similarity_threshold}")
            return None

        doc, score = found
        cache_data = self._hit_fields(doc, score)
        cache_id = cache_data["id"]

        # 真实命中：hit_count+1（修复原语义错位——原实现只刷 last_hit_time，升级永不触发）
        new_hit_count = cache_data["hit_count"] + 1
        is_formal = cache_data["is_formal"]
        try:
            self._promote_if_hot(cache_id, new_hit_count, is_formal)
            mapping = {
                "hit_count": new_hit_count,
                "last_hit_time": time.time(),
            }
            if is_formal == 0:
                mapping["is_formal"] = 1 if new_hit_count >= self.hot_threshold else 0
            self.client.hset(cache_id, mapping=mapping)
            # 临时缓存滑动续期（给"低频→高频"留窗口；正式不再滑动）
            if is_formal == 0 and new_hit_count < self.hot_threshold and self.sliding_ttl_enabled:
                self.client.expire(cache_id, self.temporary_ttl)
            self.client.hincrby(self.stats_key, "hits", 1)
        except Exception as e:
            logger.warning(f"更新命中状态失败（不影响本次命中返回）: {e}")

        cache_data["hit_count"] = new_hit_count
        cache_data["is_formal"] = 1 if (is_formal == 1 or new_hit_count >= self.hot_threshold) else 0
        logger.info(f"向量缓存命中 | 相似度={score:.3f} | 级别={'正式' if cache_data['is_formal'] else '临时'} | "
                    f"历史query='{cache_data['query_text'][:30]}'")
        return cache_data

    def save_cache(
        self,
        query_text: str,
        rewritten_query: str,
        query_vector: np.ndarray,
        chunks: List[str],
        parent_ids: List[int],
        source_metas: List[Dict],
        source_filter=None,
    ) -> Optional[str]:
        """
        RAG 跑完后，把上下文结果写入缓存（不存答案）。
        - 已有相似条目（>=阈值）：覆盖更新（chunks/parent_ids/source_metas/rewritten_query/向量用最新）
          + hit_count+1 + 升级判断；不计入 hits 统计（它没从缓存拿东西）
        - 无相似条目：新建临时缓存，id = md5(rewritten_query + source_filter)（确定性，防并发撞 key）
        - 所有写操作 try/except，失败仅 warning，返回 None
        """
        filter_tag = self._normalize_filter(source_filter)
        try:
            found = self._raw_search(query_vector, top_k=1, source_filter=filter_tag)
        except Exception as e:
            logger.warning(f"缓存回填预检异常，跳过回填: {e}")
            return None

        try:
            if found is not None:
                # 合并到既有条目：覆盖更新 + 频次累加
                doc, score = found
                cache_id = _decode(doc.id)
                old_hit_count = int(_decode(getattr(doc, "hit_count", "0")) or 0)
                old_is_formal = int(_decode(getattr(doc, "is_formal", "0")) or 0)
                new_hit_count = old_hit_count + 1
                self._promote_if_hot(cache_id, new_hit_count, old_is_formal)
                self.client.hset(cache_id, mapping={
                    "query_text": query_text,
                    "rewritten_query": rewritten_query,
                    "query_vector": query_vector.astype(np.float32).tobytes(),
                    "chunks_json": json.dumps(chunks, ensure_ascii=False),
                    "parent_ids_json": json.dumps(parent_ids),
                    "source_metas_json": json.dumps(source_metas, ensure_ascii=False),
                    "hit_count": new_hit_count,
                    "last_hit_time": time.time(),
                    "is_formal": 1 if (old_is_formal == 1 or new_hit_count >= self.hot_threshold) else 0,
                })
                logger.debug(f"缓存条目已更新（同语义簇） | {cache_id} | 累计 {new_hit_count} 次")
                return cache_id

            # 新写入：临时缓存
            cache_id = self._cache_id(rewritten_query, filter_tag)
            cache_data = {
                "query_text": query_text,
                "rewritten_query": rewritten_query,
                "query_vector": query_vector.astype(np.float32).tobytes(),
                "chunks_json": json.dumps(chunks, ensure_ascii=False),
                "parent_ids_json": json.dumps(parent_ids),
                "source_metas_json": json.dumps(source_metas, ensure_ascii=False),
                "source_filter": filter_tag,
                "hit_count": 1,
                "created_time": time.time(),
                "last_hit_time": time.time(),
                "is_formal": 0,  # 临时
            }
            self.client.hset(cache_id, mapping=cache_data)
            self.client.expire(cache_id, self.temporary_ttl)
            logger.debug(f"写入临时缓存 | query='{rewritten_query[:30]}' | TTL={self.temporary_ttl}s")
            return cache_id
        except Exception as e:
            logger.warning(f"写入缓存失败（不影响主流程）: {e}")
            return None

    # ---------- 统计 ----------

    def _count_by_formal(self) -> Tuple[int, int]:
        """统计正式/临时条目数（FT.AGGREGATE），失败返回 (0,0)"""
        try:
            req = (AggregateRequest("*")
                   .group_by("@is_formal", _ft_count().alias("cnt")))
            res = self.client.ft(self.index_name).aggregate(req)
        except Exception as e:
            logger.warning(f"聚合统计失败: {e}")
            return 0, 0
        formal_cnt = 0
        temp_cnt = 0
        for row in res.rows:
            if isinstance(row, dict):
                # dict 行：key 可能是 bytes
                is_formal = _decode(row.get(b"is_formal", row.get("is_formal", "0")))
                cnt = int(_decode(row.get(b"cnt", row.get("cnt", "0"))) or 0)
            else:
                # 列表行：按投影顺序 [is_formal, cnt]
                vals = [_decode(x) for x in (row or [])]
                is_formal = vals[0] if vals else "0"
                try:
                    cnt = int(vals[1]) if len(vals) > 1 and vals[1] else 0
                except ValueError:
                    cnt = 0
            if is_formal in ("1", "true", "True"):
                formal_cnt += cnt
            else:
                temp_cnt += cnt
        return formal_cnt, temp_cnt

    def get_stats(self) -> Dict:
        """实时统计：条目数 + hits/misses/upgraded + 命中率"""
        try:
            info = self.client.ft(self.index_name).info()
            num_docs = int(info.get("num_docs", 0))
        except Exception:
            num_docs = 0
        try:
            raw = self.client.hgetall(self.stats_key)
            hits = int(_decode(raw.get(b"hits", raw.get("hits", "0"))) or 0)
            misses = int(_decode(raw.get(b"misses", raw.get("misses", "0"))) or 0)
            upgraded = int(_decode(raw.get(b"upgraded", raw.get("upgraded", "0"))) or 0)
        except Exception:
            hits, misses, upgraded = 0, 0, 0
        total = hits + misses
        formal_cnt, temp_cnt = self._count_by_formal()
        return {
            "total_cached": num_docs,
            "index_name": self.index_name,
            "similarity_threshold": self.similarity_threshold,
            "hits": hits,
            "misses": misses,
            "query_volume": total,
            "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
            "upgraded": upgraded,
            "formal_count": formal_cnt,
            "temp_count": temp_cnt,
        }

    def reset_stats(self):
        """清零运行时统计（测试用；MySQL 日表不受影响）"""
        try:
            self.client.delete(self.stats_key)
            self.client.delete(self.stats_date_key)
            self.client.delete(self.stats_prev_key)
        except Exception as e:
            logger.warning(f"重置统计失败: {e}")

    def sync_daily_stats(self):
        """每日把累计统计增量 upsert 进 MySQL 日表 redis_cache_stats（当天只同步一次）。
        口径：Redis 计数器为累计值，与上次同步值做差得到当日增量；
        threshold/hot_threshold 记录当日生效值，保证调参后历史命中率可比。
        失败仅 warning，不影响主流程。"""
        try:
            from datetime import date, datetime
            today = date.today().isoformat()
            # 判重：当天已同步过则跳过
            try:
                if _decode(self.client.get(self.stats_date_key)) == today:
                    return
            except Exception:
                pass

            # 当前累计值
            raw = self.client.hgetall(self.stats_key)
            cur_hits = int(_decode(raw.get(b"hits", raw.get("hits", "0"))) or 0)
            cur_misses = int(_decode(raw.get(b"misses", raw.get("misses", "0"))) or 0)
            cur_upgraded = int(_decode(raw.get(b"upgraded", raw.get("upgraded", "0"))) or 0)
            cur_time_ms = float(_decode(raw.get(b"cache_time_ms", raw.get("cache_time_ms", "0"))) or 0)

            # 上次同步后的累计值
            prev = self.client.hgetall(self.stats_prev_key)
            prev_hits = int(_decode(prev.get(b"hits", prev.get("hits", "0"))) or 0)
            prev_misses = int(_decode(prev.get(b"misses", prev.get("misses", "0"))) or 0)
            prev_upgraded = int(_decode(prev.get(b"upgraded", prev.get("upgraded", "0"))) or 0)
            prev_time_ms = float(_decode(prev.get(b"cache_time_ms", prev.get("cache_time_ms", "0"))) or 0)

            dh = max(cur_hits - prev_hits, 0)
            dmiss = max(cur_misses - prev_misses, 0)
            dupgraded = max(cur_upgraded - prev_upgraded, 0)
            dtime_ms = max(cur_time_ms - prev_time_ms, 0.0)
            volume = dh + dmiss
            hit_rate = round(dh / volume, 4) if volume > 0 else 0.0
            avg_ms = round(dtime_ms / volume, 4) if volume > 0 else 0.0
            formal_cnt, temp_cnt = self._count_by_formal()

            # 懒加载 MySQL（防循环导入与启动开销）
            from mysql_qa.db.mysql_client import MySQLClient
            mysql = MySQLClient()
            try:
                mysql.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS redis_cache_stats (
                      id INT AUTO_INCREMENT PRIMARY KEY,
                      stat_date DATE NOT NULL UNIQUE,
                      query_volume INT DEFAULT 0,
                      hits INT DEFAULT 0, misses INT DEFAULT 0,
                      hit_rate DOUBLE DEFAULT 0,
                      upgraded INT DEFAULT 0,
                      total_cached INT DEFAULT 0,
                      formal_count INT DEFAULT 0, temp_count INT DEFAULT 0,
                      threshold DOUBLE DEFAULT 0,
                      hot_threshold INT DEFAULT 0,
                      avg_cache_time_ms DOUBLE DEFAULT 0,
                      updated_at DATETIME
                    )
                """)
                mysql.cursor.execute("""
                    INSERT INTO redis_cache_stats
                      (stat_date, query_volume, hits, misses, hit_rate, upgraded,
                       total_cached, formal_count, temp_count, threshold, hot_threshold,
                       avg_cache_time_ms, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                      query_volume=VALUES(query_volume), hits=VALUES(hits), misses=VALUES(misses),
                      hit_rate=VALUES(hit_rate), upgraded=VALUES(upgraded),
                      total_cached=VALUES(total_cached), formal_count=VALUES(formal_count),
                      temp_count=VALUES(temp_count), threshold=VALUES(threshold),
                      hot_threshold=VALUES(hot_threshold), avg_cache_time_ms=VALUES(avg_cache_time_ms),
                      updated_at=VALUES(updated_at)
                """, (
                    today, volume, dh, dmiss, hit_rate, dupgraded,
                    self._count_by_formal_total(), formal_cnt, temp_cnt,
                    self.similarity_threshold, self.hot_threshold, avg_ms,
                    datetime.now(),
                ))
                mysql.connection.commit()
                # 记录本次同步基准 + 日期（下次按增量继续）
                self.client.hset(self.stats_prev_key, mapping={
                    "hits": cur_hits, "misses": cur_misses,
                    "upgraded": cur_upgraded, "cache_time_ms": cur_time_ms,
                })
                self.client.set(self.stats_date_key, today)
                logger.info(f"缓存日统计已同步: {today} 命中率={hit_rate} 升级={dupgraded}")
            finally:
                mysql.close()
        except Exception as e:
            logger.warning(f"缓存日统计同步失败（不影响主流程）: {e}")

    def _count_by_formal_total(self) -> int:
        """日表 total_cached 用（FT.INFO num_docs）"""
        try:
            info = self.client.ft(self.index_name).info()
            return int(info.get("num_docs", 0))
        except Exception:
            return 0


if __name__ == "__main__":
    # 简单测试
    cache = RedisVectorCache()
    print("缓存统计:", cache.get_stats())
