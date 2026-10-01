# -*- coding: utf-8 -*-
"""
[12] 三层记忆模块：MemoryManager

分层（对应业界 Mem0/Zep/LangGraph 记忆范式）：
- 窗口记忆：MySQL conversations 表最近 N 轮（由 new_main 维护，本模块不重复实现）
- 工作记忆：Redis `rag:wm:{session_key}` 会话级状态（当前实体/过滤偏好/近期问题），TTL 过期即弃
- 长期记忆：MySQL user_memory 表（常问公司计数 / 偏好公告类型计数 / query_count），规则抽取

设计原则（最小修改、不引入新 bug）：
1. 全链路 fail-open：Redis/MySQL 任何异常 → 返回 ""/None/False，只记日志，绝不阻断主链路
2. 注入与提取分离：build_memory_context（读+拼注入文本） / update_memory（写）
3. 长期记忆抽取用规则（首版）：matched_companies 计数 + source_filter 计数，零 LLM 成本、确定性
4. user_id 尚未接入真实用户体系：先写死 10 个 DUMMY_USER_IDS，按 session_id 稳定取模映射，
   便于验证跨会话长期记忆；接入真实用户体系后在 DUMMY_USER_IDS 处更换
5. 注入文本限长 memory_max_chars，配合 new_rag_system 分层截断后的兜底硬截，保证不超 prompt 预算
"""
import hashlib
import json
import time

from base import logger


# —— 写死的 10 个用户 id（临时验证用；接入真实用户体系后在此更换/删除）——
DUMMY_USER_IDS = [
    "789d0d76-564b-4ac3-b42e-577c2267d5fd",
    "c55dbba5-0e90-4b6f-a023-078e17352481",
    "444df0e3-e68c-44a1-804c-b5bce06a2957",
    "ce637c8e-0c46-4200-903a-100e0d466973",
    "dba6d294-8e46-41f2-88fb-5c782f79900e",
    "98dc03d0-ab7e-4849-8296-07c58ae120d9",
    "b93eb582-01cd-4e09-959b-9d8156c01336",
    "01f41740-3b52-4fee-899d-313033bf4e2d",
    "790e3810-eae9-43e1-8377-0f80d60bf26b",
    "9c1b6596-8c18-4aff-bfa5-589a96245269",
]


def resolve_user_id(user_id=None, session_id=None):
    """解析记忆用的 user_id：显式传入优先；否则按 session_id 稳定映射到写死列表
    （同一会话恒映射到同一 user_id，确定性，便于验证长期记忆；session_id 为空回退列表第一个）。"""
    if user_id:
        return user_id
    if not DUMMY_USER_IDS:
        return None
    if not session_id:
        return DUMMY_USER_IDS[0]
    digest = hashlib.md5(str(session_id).encode("utf-8")).hexdigest()
    return DUMMY_USER_IDS[int(digest, 16) % len(DUMMY_USER_IDS)]


class MemoryManager:
    """三层记忆读写器。redis_client/mysql_client 可空（None 时对应层自动降级），初始化不抛异常。"""

    # 工作记忆 key 前缀（与向量缓存 index rag_query_vector_cache 完全隔离，不串数据）
    WM_PREFIX = "rag:wm:"

    def __init__(self, redis_client=None, mysql_client=None):
        # 兼容：RedisClient.client 是 StrictRedis（decode_responses=True）；传入 None 则该层降级
        self.redis = getattr(redis_client, "client", None) if redis_client is not None else None
        self.mysql = mysql_client
        try:
            from base import Config
            conf = Config()
            self.enabled = conf.MEMORY_ENABLED
            self.working_ttl = conf.WORKING_TTL
            self.memory_max_chars = conf.MEMORY_MAX_CHARS
            self.long_term_max_companies = conf.LONG_TERM_MAX_COMPANIES
        except Exception as e:
            logger.warning(f"MemoryManager 配置读取失败，使用默认值: {e}")
            self.enabled = True
            self.working_ttl = 7200
            self.memory_max_chars = 400
            self.long_term_max_companies = 10
        # 长期记忆表自动建表（IF NOT EXISTS 幂等，与项目建表风格一致）
        self._ensure_table()

    # ==================== 内部存储 ====================

    def _ensure_table(self):
        """user_memory 表不存在则创建（幂等；失败降级不影响主流程）"""
        if self.mysql is None:
            return
        try:
            self.mysql.cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL,
                    companies JSON COMMENT '常问公司 [{code,name,count}]',
                    source_filters JSON COMMENT '偏好公告类型 [{type,count}]',
                    preferences JSON COMMENT '长期偏好（预留 LLM 抽取）',
                    query_count INT DEFAULT 0,
                    created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_user_id (user_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户长期记忆表'
            """)
            self.mysql.connection.commit()
        except Exception as e:
            logger.warning(f"user_memory 表初始化失败（记忆降级，不影响主流程）: {e}")

    @staticmethod
    def _parse_json_field(value, default):
        """MySQL JSON 列经 pymysql 可能返回 str；兼容 str/list/dict"""
        if value is None or value == "":
            return default
        if isinstance(value, (list, dict)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return default

    def _get_working(self, session_key):
        """读工作记忆（Redis）；异常/不存在 → None"""
        if self.redis is None or not self.enabled or not session_key:
            return None
        try:
            raw = self.redis.get(self.WM_PREFIX + str(session_key))
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning(f"读取工作记忆失败: {e}")
            return None

    def _set_working(self, session_key, data):
        """写工作记忆（Redis，TTL=working_ttl）；异常只记日志"""
        if self.redis is None or not self.enabled or not session_key:
            return
        try:
            self.redis.set(self.WM_PREFIX + str(session_key),
                           json.dumps(data, ensure_ascii=False),
                           ex=self.working_ttl)
        except Exception as e:
            logger.warning(f"写入工作记忆失败: {e}")

    def _get_long_term(self, user_id):
        """读长期记忆（MySQL）；异常/不存在 → None"""
        if self.mysql is None or not self.enabled or not user_id:
            return None
        try:
            self.mysql.cursor.execute(
                "SELECT companies, source_filters, preferences, query_count "
                "FROM user_memory WHERE user_id=%s", (user_id,))
            row = self.mysql.cursor.fetchone()
            if not row:
                return None
            return {
                "companies": self._parse_json_field(row.get("companies"), []) or [],
                "source_filters": self._parse_json_field(row.get("source_filters"), []) or [],
                "query_count": int(row.get("query_count") or 0),
            }
        except Exception as e:
            logger.warning(f"读取长期记忆失败: {e}")
            return None

    # ==================== 对外：写 ====================

    def update_memory(self, session_key, user_id, question, answer,
                      matched_companies=None, source_filter=None):
        """流式回答完成后调用：写工作记忆（Redis）+ 长期记忆（MySQL）。
        任何异常只记日志，不影响主流程。
        :param session_key: 会话 id（工作记忆按会话维度）
        :param user_id: 用户 id（长期记忆按用户维度；None 时按 session_id 稳定映射到写死列表）
        :param matched_companies: 分类器输出的公司实体列表 [{code,name},...]，规则抽取的输入
        :param source_filter: 公告类型过滤（偏好统计输入）
        """
        if not self.enabled:
            return
        uid = None
        try:
            uid = resolve_user_id(user_id, session_key)
        except Exception as e:
            logger.warning(f"user_id 解析失败，跳过长期记忆: {e}")
        # —— 工作记忆（Redis，会话级，TTL）——
        try:
            wm = self._get_working(session_key) or {}
            entity_map = {e.get("code"): e for e in (wm.get("entities") or []) if e.get("code")}
            for c in (matched_companies or []):
                code = str(c.get("code") or "")
                if not code:
                    continue
                prev = entity_map.get(code, {})
                entity_map[code] = {"code": code,
                                    "name": c.get("name") or prev.get("name", "")}
            recent = (wm.get("recent_questions") or [])
            if question:
                recent.append(str(question)[:200])
            recent = recent[-5:]
            self._set_working(session_key, {
                "entities": list(entity_map.values()),
                "source_filter": source_filter or wm.get("source_filter", ""),
                "recent_questions": recent,
                "updated_at": int(time.time()),
            })
        except Exception as e:
            logger.warning(f"工作记忆更新失败（不影响主流程）: {e}")
        # —— 长期记忆（MySQL，用户级，规则计数）——
        if uid:
            self._upsert_long_term(uid, matched_companies or [],
                                   [source_filter] if source_filter else [])

    def clear_working(self, session_key):
        """清除指定会话的工作记忆（Redis）；异常只记日志"""
        if self.redis is None or not session_key:
            return
        try:
            self.redis.delete(self.WM_PREFIX + str(session_key))
        except Exception as e:
            logger.warning(f"清除工作记忆失败: {e}")

    def _upsert_long_term(self, user_id, companies, source_filters):
        """长期记忆 upsert：合并现有计数 → 按 count 降序取前 N → INSERT ... ON DUPLICATE KEY UPDATE。
        任何异常只记日志。"""
        if self.mysql is None or not self.enabled:
            return
        try:
            row = self._get_long_term(user_id) or {}
            merged_companies = {}
            for c in (row.get("companies") or []):
                code = str(c.get("code") or "")
                if not code:
                    continue
                merged_companies[code] = {"code": code, "name": c.get("name", ""),
                                          "count": int(c.get("count") or 0)}
            for c in (companies or []):
                code = str(c.get("code") or "")
                if not code:
                    continue
                prev = merged_companies.get(code, {"code": code, "name": "", "count": 0})
                prev["name"] = c.get("name") or prev.get("name") or ""
                prev["count"] = int(prev.get("count") or 0) + 1
                merged_companies[code] = prev
            top_companies = sorted(merged_companies.values(),
                                   key=lambda x: -int(x.get("count") or 0))[:self.long_term_max_companies]

            merged_filters = {}
            for f in (row.get("source_filters") or []):
                t = str(f.get("type") or "") if isinstance(f, dict) else str(f or "")
                if not t:
                    continue
                merged_filters[t] = int(f.get("count") or 0) if isinstance(f, dict) else merged_filters.get(t, 0)
            for sf in (source_filters or []):
                t = sf.get("type") if isinstance(sf, dict) else str(sf or "")
                if not t:
                    continue
                merged_filters[t] = merged_filters.get(t, 0) + 1
            top_filters = [{"type": k, "count": v}
                           for k, v in sorted(merged_filters.items(), key=lambda x: -x[1])]

            query_count = int(row.get("query_count") or 0) + 1
            companies_json = json.dumps(top_companies, ensure_ascii=False)
            filters_json = json.dumps(top_filters, ensure_ascii=False)
            self.mysql.cursor.execute("""
                INSERT INTO user_memory (user_id, companies, source_filters, query_count)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    companies=%s, source_filters=%s, query_count=%s, updated_time=CURRENT_TIMESTAMP
            """, (user_id, companies_json, filters_json, query_count,
                  companies_json, filters_json, query_count))
            self.mysql.connection.commit()
        except Exception as e:
            logger.warning(f"写入长期记忆失败: {e}")

    # ==================== 对外：读 + 拼注入文本 ====================

    def build_memory_context(self, session_key, user_id=None):
        """读工作记忆 + 长期记忆 → 拼注入文本（限长 memory_max_chars）。
        任何异常/无记忆 → 返回 ""（绝不阻断主链路）。
        注入位置由调用方决定（new_rag_system 拼到 prompt 的 history 段最前）。"""
        if not self.enabled or not session_key:
            return ""
        parts = []
        try:
            uid = resolve_user_id(user_id, session_key)
        except Exception as e:
            logger.warning(f"user_id 解析失败，跳过长期记忆注入: {e}")
            uid = None
        # —— 长期记忆（用户偏好/常问公司）——
        if uid:
            lt = self._get_long_term(uid)
            if lt:
                cs = lt.get("companies") or []
                if cs:
                    parts.append("（用户长期关注）常问公司：" + "、".join(
                        f"{c.get('name','')}({c.get('code','')})×{c.get('count',0)}"
                        for c in cs[:5]))
                sfs = lt.get("source_filters") or []
                if sfs:
                    parts.append("偏好公告：" + "、".join(
                        str(f.get("type", "")) if isinstance(f, dict) else str(f)
                        for f in sfs[:5]))
        # —— 工作记忆（本会话关注）——
        wm = self._get_working(session_key)
        if wm:
            ents = wm.get("entities") or []
            if ents:
                parts.append("（本会话关注）当前关注：" + "、".join(
                    f"{e.get('name','')}({e.get('code','')})" for e in ents[:5]))
        text = "；".join(parts)
        if len(text) > self.memory_max_chars:
            text = text[:self.memory_max_chars]
        return text
