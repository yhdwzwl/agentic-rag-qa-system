# -*- coding: utf-8 -*-
"""
standard_qa 一级缓存生成器（锚点驱动五段式管线）

目标：为 MySQL standard_qa 生成与金融公告强关联的高质量问答对，
     作为 BM25 一级缓存数据源（命中=直接返回标准答案，未命中=进 RAG）。

管线：
    Stage0 数据治理（分层抽样）   —— 剔除封面/备查文件/短块，按 pdf_type 配额抽样，断点续跑
    Stage1 事实锚点抽取          —— LLM 抽"可核验的事实"(entity/fact_type/fact_value/evidence)
    Stage2 动态提问              —— 按公告类型骨架 + 锚点值生成口语化问题与标准答案
    Stage3 双重校验              —— 规则硬校验(数字/日期/代码必须落在原文) + LLM 软校验
    Stage4 去重 + 配额 + 落库    —— 问题去重、限量、写入 standard_qa + standard_qa_meta(溯源)
用法（代码内调用，不走命令行）：

    from generate_test_queries import generate_standard_qa

    # 生成 500 条（默认覆盖五类公告，配额见 pdf_type_weights）
    result = generate_standard_qa(target_count=500)

    # 只跑某一类公告
    result = generate_standard_qa(target_count=200, pdf_type_filter="融资公告")

    # 先试跑：只打印结果不写库
    result = generate_standard_qa(target_count=30, dry_run=True)
"""

import configparser
import json
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pymysql
from openai import OpenAI

# ============================================================
# 常量与默认配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "..", "..", "config.ini")  # rag_qa/core/ -> 项目根
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.ini")  # 兜底：直接在根目录运行时

# 低价值块 heading 黑名单（这些块不产生 QA）
HEADING_BLACKLIST = {"文件封面", "封面", "备查文件", "目录", "落款"}

# 每类公告的默认配额（按 parent_docs.pdf_type 分布校正：融资公告占比过高，压到 50%）
DEFAULT_PDF_TYPE_WEIGHTS = {
    "融资公告": 0.50,
    "重大事项": 0.20,
    "信息变更": 0.10,
    "风险提示": 0.10,
    "持股变动": 0.10,
}

# 问题骨架：按公告类型约束提问方向（真实投资者问法），生成时由 LLM 结合锚点填充
QA_SKELETONS = {
    "融资公告": "发行价格、发行数量、募资总额、上市日期、募资用途、承销商",
    "重大事项": "事项具体内容、涉及金额、对公司的影响、董事会/股东大会决议结果",
    "信息变更": "变更前后对比、变更原因、生效日期、涉及主体",
    "风险提示": "风险类型、风险具体内容、公司应对措施、对投资者的提示",
    "持股变动": "股东名称、变动方向(增持/减持)、变动数量、比例、价格区间、变动后持股情况",
}

# 归一化时忽略的标点/空白（用于去重与数字比对）
_STRIP_RE = re.compile(r"[\s\u3000,，。；;：:、！!？?\"'“”‘’（）()\[\]【】\-—\t\n]+")

# 数字+单位 token：用于"答案中的数字必须能在原文 evidence 中找到"的硬校验
# 注意：复合单位（元/股/亿元/万股…）必须排在简单单位前，否则交替匹配会截断
# group(1)=数字部分（含千分位/小数），供边界感知比对
_NUM_UNIT_RE = re.compile(
    r"(\d[\d,，.]*)\s*(?:元/股|万元/股|元/吨|亿元|万元|万股|港元|人民币|美元|亿|万|元|股|％|%)",
)
_DATE_RE = re.compile(r"\d{4}[年\-\.]\d{1,2}[月\-\.]\d{1,2}日?")

# 区间日期（如 2026年1-8月 / 2026年1月至8月 中的 "2026年1-8"）：表示一段范围，
# 不是"单个事实日期"，格式变体多（1-8月/1月至8月），不做整体硬比对，只校验年份出现。
_RANGE_DATE_RE = re.compile(r"\d{4}年\d{1,2}[-至]\d{1,2}")

# 边界感知独立数字：千分位/小数，但不跨空白/标点拼接
# 关键：normalize_text 会删除空格和逗号，若直接对整个 evidence 归一化，
#       '17,840,000 8.92' 会被拼成 '178400008.92'（两个数字变一个，破坏校验），
#       因此必须先在原文上按边界切出独立数字，再逐个归一化。
_NUM_VALUE_RE = re.compile(r"\d[\d,，]*(?:\.\d+)?")


# ============================================================
# 配置读取（密钥只从 config.ini 读，不硬编码）
# ============================================================
def load_config() -> Dict:
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH, encoding="utf-8")

    mysql = cfg["mysql"] if cfg.has_section("mysql") else {}
    llm = cfg["llm"] if cfg.has_section("llm") else {}

    # 多模型轮换列表：table_llm_models（逗号分隔，额度用尽自动切换下一个），
    # 未配置时退化为单模型 [model]
    raw_models = llm.get("table_llm_models", "") or ""
    models = [m.strip() for m in raw_models.split(",") if m.strip()]
    if not models:
        models = [llm.get("model", "glm-5.3")]
    elif llm.get("model", "") and llm["model"] not in models:
        models.insert(0, llm["model"])  # 主模型优先

    return {
        "mysql": {
            "host": mysql.get("host", "127.0.0.1"),
            "port": mysql.getint("port", 3306),
            "user": mysql.get("user", "root"),
            "password": mysql.get("password", ""),
            "database": mysql.get("database", "df_rag_v4"),
        },
        "llm": {
            "model": llm.get("model", "glm-5.3"),
            "models": models,
            "api_key": llm.get("dashscope_api_key", ""),
            "base_url": llm.get("dashscope_base_url", ""),
            "timeout": llm.getint("table_llm_timeout", 60) or 60,
        },
    }


# ============================================================
# LLM 多模型轮换（额度用尽自动切换下一个）
# ============================================================
# 额度用尽/欠费错误关键词（与 table_enhancer.TableLLMEnhancer 同一套判定口径）
_QUOTA_KEYWORDS = (
    "quota", "insufficient", "balance", "额度", "欠费", "余额不足",
    "payment required", "out of quota", "credit", "accepted balance",
    "超过限制", "调用额度", "trial expired", "expired", "账户余额",
)


def _is_quota_error(exc: Exception) -> bool:
    """判定是否为额度/欠费类错误（这类错误 = 该模型永久不可用，切换下一个）。"""
    msg = str(exc).lower()
    status = None
    for attr in ("status_code", "http_status", "code"):
        v = getattr(exc, attr, None)
        if v is not None:
            status = v
            break
    if isinstance(status, int) and status in (402, 429):
        return True
    return any(k in msg for k in _QUOTA_KEYWORDS)


class LLMRouter:
    """多模型轮换调用器：额度用尽自动切换下一个；线程安全 + 失败模型共享熔断。

    - 环形游标：同一时刻多个 worker 共享，下次从下一个模型开始，避免重复命中失败模型；
    - _failed 集合共享：一个线程发现某模型额度用尽即拉黑，其他线程立即跳过；
    - 全部模型拉黑后，_next_model 直接返回 None，后续调用零网络开销（熔断）；
    - 错误按类型计数（insufficient_quota / network_or_api / empty_response），
      供 stats["llm_error"] 透传，解决"静默失败看不出原因"的问题。
    """
    def __init__(self, api_key: str, base_url: str, models: List[str], timeout: int = 60):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.models = [m for m in (models or []) if m]
        self._cursor = 0
        self._failed: set = set()
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = {}

    def _next_model(self) -> Optional[str]:
        with self._lock:
            for _ in range(len(self.models)):
                model = self.models[self._cursor % len(self.models)]
                self._cursor += 1
                if model not in self._failed:
                    return model
            return None  # 全部已拉黑 -> 熔断，零调用

    def _record(self, err_type: str) -> None:
        with self._lock:
            self._stats[err_type] = self._stats.get(err_type, 0) + 1

    def chat(self, messages: List[Dict], temperature: float = 0.2) -> Optional[str]:
        """轮换模型发起一次对话，返回文本内容；所有模型均失败返回 None。"""
        for _ in range(len(self.models)):
            model = self._next_model()
            if model is None:
                return None  # 熔断：所有模型已拉黑，不再发起调用
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
                self._record("empty_response")
            except Exception as e:
                if _is_quota_error(e):
                    self._failed.add(model)  # 额度类错误：永久拉黑该模型
                    self._record("insufficient_quota")
                else:
                    self._record("network_or_api")
        return None

    def error_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ============================================================
# 文本归一化工具
# ============================================================
def normalize_text(text: str) -> str:
    """全角转半角 + 去空白标点，用于问题去重与数字比对。 避免因排版差异误判"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        out.append(chr(code))
    return _STRIP_RE.sub("", "".join(out))


def extract_num_tokens(text: str) -> List[str]:
    """提取'数字+单位'token 的数字部分，归一化后用于硬校验（答案数字必须落在原文）。

    注意：只返回 group(1) 数字部分（去掉单位），单位不参与子串比对——
          evidence 原文中数字后可能无单位（如 '17,840,000 8.92'），
          带单位整体比对会导致正确答案被误拒。
    """
    return [normalize_text(m.group(1)) for m in _NUM_UNIT_RE.finditer(text)]


def extract_numeric_values(text: str) -> List[str]:
    """边界感知提取独立数字（千分位/小数），归一化后用于硬校验。

    关键：不跨空白/标点拼接。normalize_text 会删除空格和逗号，
          若直接对整个 evidence 归一化，'17,840,000 8.92' 会被拼成
          '178400008.92'（两个数字变一个），导致答案 17840000 找不到（误拒），
          或幻觉拼接数字 178400008.92 反而命中（漏放）。
          因此必须先按边界切出独立数字，再逐个归一化。
    """
    return [normalize_text(m.group()) for m in _NUM_VALUE_RE.finditer(text)]


def extract_num_unit_pairs(text: str) -> Dict[str, set]:
    """提取 '数字 -> 该数字带过的单位集合' 映射（只统计带单位的数字出现）。

    用于单位量级一致性校验：evidence 中某数字若带单位出现过（如 '5.68元/股'），
    则答案携带的单位必须与该集合一致，防止 '元' 被篡改成 '万元'（差 10000 倍）。
    若该数字在 evidence 中从未带单位出现（如 '17,840,000 8.92'），集合为空，
    调用方应放宽单位校验（原文未提供单位，无从比对）。
    """
    pairs: Dict[str, set] = {}
    for m in _NUM_UNIT_RE.finditer(text):
        num = normalize_text(m.group(1))
        unit = normalize_text(m.group(0)[len(m.group(1)):])
        if num:
            pairs.setdefault(num, set()).add(unit or "无单位")
    return pairs


def extract_dates(text: str) -> List[str]:
    return [m.group() for m in _DATE_RE.finditer(text)]


def _date_key(s: str) -> str:
    """把日期字符串转为可比键 YYYYMMDD（兼容 '2026-09-01' / '2026年9月1日' / '2026.9.1'）。

    用于 pdf_time 兜底比较：答案日期与公告日期格式常不一致（中文 vs 连字符），
    直接子串比对会误拒真实一致的日期。
    """
    m = re.search(r"(\d{4})[年\-\.](\d{1,2})[月\-\.](\d{1,2})", s)
    if not m:
        return ""
    return f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}"


def bigram_jaccard(a: str, b: str) -> float:
    """轻量问题去重：字符 bigram Jaccard（不引入嵌入模型，避免 GPU 依赖）。"""
    sa, sb = set(), set()
    a, b = normalize_text(a), normalize_text(b)
    sa = {a[i:i + 2] for i in range(len(a) - 1)}
    sb = {b[i:i + 2] for i in range(len(b) - 1)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ============================================================
# MySQL 访问
# ============================================================
def get_conn(mysql_cfg: Dict) -> pymysql.Connection:
    return pymysql.connect(
        host=mysql_cfg["host"],
        port=mysql_cfg["port"],
        user=mysql_cfg["user"],
        password=mysql_cfg["password"],
        database=mysql_cfg["database"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def ensure_meta_table(conn: pymysql.Connection) -> None:
    """standard_qa_meta：QA 生成溯源表（记录来源父块/锚点/校验结果，支持断点续跑与重跑）。"""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS standard_qa_meta (
            id INT AUTO_INCREMENT PRIMARY KEY,
            standard_qa_id INT NULL COMMENT '关联 standard_qa.id（NULL=校验未通过未入库）',
            batch_id VARCHAR(40) COMMENT '生成批次',
            source_doc_id INT COMMENT '来源父块 parent_docs.id',
            doc_code VARCHAR(20),
            doc_name VARCHAR(200),
            pdf_type VARCHAR(50),
            heading VARCHAR(500),
            question TEXT COMMENT '生成的问题（含被拒样本，便于人工抽检）',
            answer TEXT COMMENT '生成的答案（含被拒样本）',
            anchor_json MEDIUMTEXT COMMENT '事实锚点 JSON',
            rule_check VARCHAR(20) COMMENT '规则硬校验: pass/reject',
            llm_check VARCHAR(20) COMMENT 'LLM软校验: pass/reject/na',
            check_note VARCHAR(500),
            created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            KEY idx_source_doc (source_doc_id),
            KEY idx_batch (batch_id),
            KEY idx_std_qa (standard_qa_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )
    conn.commit()


def get_used_doc_ids(conn: pymysql.Connection) -> set:
    """断点续跑：已处理过（无论是否入库）的父块 id 集合。"""
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT source_doc_id FROM standard_qa_meta")
    return {r["source_doc_id"] for r in cur.fetchall()}


def ensure_qa_evidence_col(conn: pymysql.Connection, database: str) -> None:
    """幂等确保 standard_qa 存在 evidence 列（冗余存储答案依据原文，便于 BM25 命中后溯源展示）。"""
    cur = conn.cursor()
    cur.execute(
        """SELECT COUNT(*) AS n FROM information_schema.COLUMNS
           WHERE TABLE_SCHEMA=%s AND TABLE_NAME='standard_qa' AND COLUMN_NAME='evidence'""",
        (database,),
    )
    if cur.fetchone()["n"] == 0:
        cur.execute(
            """ALTER TABLE standard_qa
               ADD COLUMN evidence TEXT NULL COMMENT '答案依据原文片段' AFTER answer"""
        )
        conn.commit()
        print("[Schema] standard_qa 已新增 evidence 列")


def insert_qa_meta(
    conn: pymysql.Connection,
    qa: Dict,
    doc: Dict,
    batch_id: str,
    rule_check: str,
    llm_check: str,
    check_note: str,
    qa_id: Optional[int] = None,
) -> int:
    """统一写 standard_qa_meta 溯源表（同时控制是否写主表）。

    规则：
        qa_id is None 且 rule_check == 'pass' 且 llm_check in ('pass','na')
            -> 先 INSERT standard_qa 主表（自增 id 由数据库生成，cur.lastrowid 取回），再写 meta 关联；
        其余情况 -> 只写 meta：qa_id 保持 None 时写入 standard_qa_id=NULL
            （规则被拒/LLM被拒样本绝不进主表，但留痕便于人工抽检）。

    返回：写主表时返回自增 qa_id；否则返回 None/传入值。
    """
    cur = conn.cursor()

    # 通过校验（规则+LLM 双通过）：先写主表 standard_qa，拿自增 id 用于关联
    if qa_id is None and rule_check == "pass" and llm_check in ("pass", "na"):
        cur.execute(
            """INSERT INTO standard_qa (question, answer, category, status, evidence, created_time, updated_time)
               VALUES (%s, %s, %s, 'active', %s, NOW(), NOW())""",
            (qa["question"], qa["answer"], doc["pdf_type"], qa.get("evidence", "") or ""),
        )
        qa_id = cur.lastrowid

    # 再写溯源表（qa_id=None 时 pymysql 自动转 NULL）
    cur.execute(
        """INSERT INTO standard_qa_meta
           (standard_qa_id, batch_id, source_doc_id, doc_code, doc_name, pdf_type, heading,
            question, answer, anchor_json, rule_check, llm_check, check_note)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            qa_id, batch_id, doc["id"], doc.get("code"), doc.get("name"),
            doc["pdf_type"], doc.get("heading"),
            qa["question"], qa["answer"],
            json.dumps(qa.get("anchors", []), ensure_ascii=False),
            rule_check, llm_check, check_note[:500],
        ),
    )
    conn.commit()
    return qa_id


# ============================================================
# Stage 0：数据治理（分层抽样 + 低价值块剔除 + 断点续跑）
# ============================================================
def sample_parent_docs(
    conn: pymysql.Connection,
    target_count: int,
    max_queries_per_doc: int,
    pdf_type_weights: Dict[str, float],
    pdf_type_filter: Optional[str] = None,
    min_block_len: int = 100,
    redundancy_factor: float = 1.3,
) -> List[Dict]:
    """按 pdf_type 配额分层抽样 parent_docs。

    目标 QA 条数 -> 需要的父块数 ≈ target / max_per_doc * 冗余系数（去重/校验失败损耗）。
    """
    cur = conn.cursor()

    # 计算每类需要的块数
    total_docs = max(1, int(target_count / max_queries_per_doc * redundancy_factor) + 1)
    weights = dict(pdf_type_weights or DEFAULT_PDF_TYPE_WEIGHTS)
    if pdf_type_filter:
        weights = {pdf_type_filter: 1.0}
    # 过滤掉库里没有的 pdf_type
    placeholders = ",".join(["%s"] * len(weights))
    cur.execute(f"SELECT pdf_type, COUNT(*) AS n FROM parent_docs WHERE pdf_type IN ({placeholders}) GROUP BY pdf_type", list(weights.keys()))
    exist_counts = {r["pdf_type"]: r["n"] for r in cur.fetchall()}

    # 归一化权重（只保留存在的类型）
    weights = {k: v for k, v in weights.items() if k in exist_counts}
    wsum = sum(weights.values()) or 1.0
    weights = {k: v / wsum for k, v in weights.items()}

    used_ids = get_used_doc_ids(conn)
    docs: List[Dict] = []

    blacklist_sql = "heading NOT IN (" + ",".join(["%s"] * len(HEADING_BLACKLIST)) + ")"
    for pdf_type, weight in weights.items():
        quota = max(1, int(total_docs * weight + 0.5))
        quota = min(quota, exist_counts.get(pdf_type, 0))
        if quota <= 0:
            continue
        # 注意参数顺序必须与 SQL 占位符顺序一致：pdf_type -> heading黑名单 -> min_block_len -> quota
        params = [pdf_type] + list(HEADING_BLACKLIST) + [min_block_len]
        where = f"pdf_type=%s AND {blacklist_sql} AND LENGTH(parent_content) >= %s"
        if used_ids:
            ph = ",".join(["%s"] * len(used_ids))
            where += f" AND id NOT IN ({ph})"
            params += list(used_ids)
        params.append(quota)
        cur.execute(
            f"SELECT id, file_id, code, name, pdf_type, heading, heading_path, complexity_score, parent_content "
            f"FROM parent_docs WHERE {where} ORDER BY RAND() LIMIT %s",
            params,
        )
        docs.extend(cur.fetchall())

    random.shuffle(docs)
    return docs


# ============================================================
# Stage 1：事实锚点抽取
# ============================================================
ANCHOR_PROMPT = """你是金融公告信息抽取助手。给定一段股票公告正文，抽取其中**可以回答投资者问题的事实锚点**。

要求：
1. 只抽取公告中**明确陈述的事实**（数字/日期/比例/金额/事项/条款），不得推断、不得猜测。
2. 每个锚点必须给出 evidence：原文中的确切表述片段（**必须包含关键数字/日期/代码**）。
3. 忽略封面、落款、免责声明、无信息量的表述。
4. 每段最多抽取 5 个锚点；锚点之间不要重复（同一事实只取一次）。
5. 若公告正文无实质事实（如纯格式性文字），返回空列表。

【公告信息】
公司：{doc_name}（代码 {doc_code}）
公告类型：{pdf_type}
章节：{heading}

【公告正文】
{doc_content}

只返回 JSON（不要任何解释），格式：
{{"anchors": [{{"fact_type": "发行价格", "entity": "公司简称", "fact_value": "5.68元/股", "evidence": "发行价格为5.68元/股"}}]}}"""


def extract_anchors(
    router: LLMRouter,
    doc: Dict,
    max_anchors: int = 5,
    retries: int = 1,
) -> List[Dict]:
    """LLM 抽取事实锚点；解析失败重试 retries 次，最终失败返回 []（该块跳过，不中断整体）。

    模型轮换由 router 完成：额度用尽自动切下一个，全部失败熔断返回 None。
    """
    prompt = ANCHOR_PROMPT.format(
        doc_name=doc.get("name", ""),
        doc_code=doc.get("code", ""),
        pdf_type=doc.get("pdf_type", ""),
        heading=doc.get("heading", ""),
        doc_content=doc["parent_content"][:2500],
    )
    for _ in range(retries + 1):
        content = router.chat([{"role": "user", "content": prompt}], temperature=0.2)
        if content is None:
            return []  # 所有模型均失败（router 已记录 error_stats）
        try:
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.S).strip()
            data = json.loads(content)
            anchors = data.get("anchors", [])[:max_anchors]
            # 过滤掉没有 evidence 的锚点（无法校验 = 不用）
            return [a for a in anchors if a.get("evidence") and a.get("fact_type")]
        except Exception:
            continue
    return []


# ============================================================
# Stage 2：动态提问（类型骨架 + 锚点填充）
# ============================================================
QA_PROMPT = """你是金融领域的投资者问答生成助手。基于给定公告和事实锚点，生成**真实散户会问的问题**与标准答案。

要求：
1. 问题口语化、长度 10-50 字，像真实投资者提问（例如"天博智能的发行价格是多少？""XX这次募资准备用在什么地方？"）。
2. 答案必须**严格基于锚点事实**，简洁准确、独立成句、可直接回答；不得超出锚点/原文内容。
3. **答案中的数字/日期/证券代码必须与 evidence 完全一致**，不得改写、不得四舍五入出新的数值。
4. 每个锚点最多生成 1 个问题；全部锚点合计最多 {max_per_doc} 个问题；优先选信息量大的锚点。
5. 不要生成"公告里写了什么/请介绍一下"这类伪问题，不要生成与锚点无关的开放问题。

【公告信息】
公司：{doc_name}（代码 {doc_code}）
公告类型：{pdf_type}（提问方向参考：{skeleton}）

【事实锚点】
{anchors_json}

只返回 JSON（不要任何解释），格式：
{{"qas": [{{"question": "...", "answer": "...", "fact_type": "发行价格", "evidence": "..."}}]}}"""


def generate_qa(
    router: LLMRouter,
    doc: Dict,
    anchors: List[Dict],
    max_queries_per_doc: int,
) -> List[Dict]:
    """基于锚点生成 QA；解析失败返回 []。模型轮换由 router 完成。"""
    if not anchors:
        return []
    anchors_json = json.dumps(anchors, ensure_ascii=False)
    skeleton = QA_SKELETONS.get(doc.get("pdf_type"), "公告的核心事实、数字、日期、影响")
    prompt = QA_PROMPT.format(
        doc_name=doc.get("name", ""),
        doc_code=doc.get("code", ""),
        pdf_type=doc.get("pdf_type", ""),
        skeleton=skeleton,
        max_per_doc=max_queries_per_doc,
        anchors_json=anchors_json,
    )
    content = router.chat([{"role": "user", "content": prompt}], temperature=0.3)
    if content is None:
        return []
    try:
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.S).strip()
        data = json.loads(content)
        qas = data.get("qas", [])
        out = []
        for q in qas:
            if q.get("question") and q.get("answer"):
                q["anchors"] = anchors  # 溯源
                out.append(q)
        return out[:max_queries_per_doc]
    except Exception:
        return []


# ============================================================
# Stage 3：双重校验
# ============================================================
def rule_check(doc: Dict, qa: Dict) -> Tuple[bool, str]:
    """规则硬校验（防幻觉的核心门禁）。

    1) 答案中的每个数字必须能在 evidence 的数字集合中找到（边界感知，防 normalize 拼接）；
    2) 答案中的日期必须出现在 evidence（区间日期如"2026年1-8月"只校验年份）；
    3) 公司代码（doc.code，若有）必须出现在 question/answer/evidence；
    4) 公司主体：锚点 entity 或 doc.name 任一命中 question/answer/evidence 即放行；
    5) answer 长度 5-500。
    """
    question, answer = qa.get("question", ""), qa.get("answer", "")
    evidence = qa.get("evidence", "") or ""
    evidence_norm = normalize_text(evidence)

    if not (5 <= len(answer) <= 500):
        return False, f"answer 长度异常({len(answer)})"

    # 1) 数字硬校验：答案每个"数字+单位"token 的数字部分必须在 evidence 的数字集合中
    #    不直接对整个 evidence 做归一化子串匹配——normalize 会删空格/逗号，
    #    '17,840,000 8.92' 会被拼成 '178400008.92'，导致正确答案误拒、幻觉数字漏放。
    #    同时做单位量级一致性：evidence 中该数字若带单位出现过，答案单位必须一致
    #    （防 '元'→'万元'、'万'→'亿' 这类量级篡改幻觉）。
    evidence_values = set(extract_numeric_values(evidence))
    ev_num_units = extract_num_unit_pairs(evidence)
    for m in _NUM_UNIT_RE.finditer(answer):
        num_part = normalize_text(m.group(1))
        if num_part and num_part not in evidence_values:
            return False, f"答案数字 {num_part} 未在原文 evidence 中找到"
        # 单位量级一致性（evidence 中该数字无单位记录时放宽）
        unit_part = normalize_text(m.group(0)[len(m.group(1)):])
        ev_units = ev_num_units.get(num_part)
        if unit_part and ev_units and unit_part not in ev_units:
            return False, f"答案单位 {unit_part} 与原文 evidence 不一致（原文: {sorted(ev_units)}）"

    # 2) 日期校验：标准日期归一化后必须命中 evidence（normalize 会删 '-'，
    #    '2026-09-22' 需与原文同形式归一化后比对）；区间日期（如 2026年1-8月）
    #    格式变体多（1-8月/1月至8月），只要求其年份（如 '2026年'）出现在 evidence；
    #    evidence 找不到时再用结构化日期键比较公告日期 pdf_time（兼容中英格式）。
    pdf_time_norm = normalize_text(doc.get("pdf_time", ""))
    pdf_time_key = _date_key(doc.get("pdf_time", ""))
    for d in extract_dates(answer):
        if not d:
            continue
        if _RANGE_DATE_RE.search(d):
            year = d[:5]  # '2026年'
            if year not in evidence_norm and year not in pdf_time_norm:
                return False, f"答案区间日期 {d} 的年份未在 evidence 中找到"
        else:
            d_norm = normalize_text(d)
            if d_norm not in evidence_norm and d_norm not in pdf_time_norm and (not pdf_time_key or _date_key(d) != pdf_time_key):
                return False, f"答案日期 {d} 未在 evidence 中找到"

    # 3) 证券代码一致性：若问题/答案提到代码，则 evidence 中必须能找到（散户提问常不带代码，故不强制出现）
    code = doc.get("code")
    if code:
        qa_text_norm = normalize_text(question + answer)
        if code in qa_text_norm and code not in evidence_norm:
            return False, f"问题/答案提到证券代码 {code} 但原文 evidence 中未找到"

    # 4) 公司主体校验：锚点 entity、doc.name、doc.name 前 2 字前缀
    #    任一命中问题/答案/evidence 即放行。
    #    旧逻辑只查 doc.name，误杀率高：LLM 生成的 Q/A 常用简称变体（"中集"vs"中集集团"）
    #    或锚点主体名，原文 evidence 也可能省略主语（如"减持期间为…"）。
    #    锚点来自本文档证据，其 entity 命中即代表 QA 与文档强关联；全不命中才拒（兜底）。
    candidates = {a.get("entity", "") for a in (qa.get("anchors") or []) if a.get("entity") and len(a.get("entity")) >= 2}
    name = doc.get("name")
    if name and len(name) >= 2:
        candidates.add(name)
        prefix = name[:2]
        if len(prefix) >= 2 and prefix != name:
            candidates.add(prefix)
    if candidates:
        hay = normalize_text(question + answer + evidence)
        if not any(c in hay for c in candidates if c):
            return False, f"公司相关实体 {sorted(candidates)[:3]} 未出现在问题/答案/evidence"

    return True, "pass"


LLM_CHECK_PROMPT = """你是金融问答质量审核员。判断以下问答对是否**可以直接作为标准答案**回答投资者。

判定标准：
1. 答案与公告原文 evidence 一致，无虚构/篡改的数字、日期、金额。
2. 问题真实、可答、口语化（不是"请介绍一下"这类伪问题）。
3. 答案独立成立、简洁准确，不依赖上下文也能理解。
4. 问题具有实质性信息价值：应涉及可核验的具体信息（数字/日期/金额/事项/条款/影响/风险等）；
   仅涉及"备查文件是否存在/在哪里""某部门或岗位负责什么业务""流程性、程序性表述"等
   低信息量的提问，判为不通过。

【公告信息】公司：{doc_name}（代码 {doc_code}）｜类型：{pdf_type}

【问题】{question}
【答案】{answer}
【原文证据】{evidence}

只返回 JSON（不要解释）：{{"pass": true|false, "reason": "不超过50字的原因"}}"""


def llm_check(router: LLMRouter, doc: Dict, qa: Dict) -> Tuple[bool, str]:
    """LLM 软校验：判答案可否直接作为标准答案。失败按 reject 处理（宁可少不可错）。"""
    prompt = LLM_CHECK_PROMPT.format(
        doc_name=doc.get("name", ""),
        doc_code=doc.get("code", ""),
        pdf_type=doc.get("pdf_type", ""),
        question=qa.get("question", ""),
        answer=qa.get("answer", ""),
        evidence=qa.get("evidence", "") or "",
    )
    content = router.chat([{"role": "user", "content": prompt}], temperature=0.0)
    if content is None:
        return False, "所有 LLM 模型不可用（见 stats.llm_error）"
    try:
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.S).strip()
        data = json.loads(content)
        return bool(data.get("pass")), str(data.get("reason", ""))[:200]
    except Exception as e:
        return False, f"llm_check 解析失败: {str(e)[:120]}"


# ============================================================
# Stage 4：去重 + 配额 + 落库（含主函数）
# ============================================================
def _is_dup(question: str, existing_norms: List[str], similarity_threshold: float) -> bool:
    q_norm = normalize_text(question)
    if not q_norm:
        return True
    if q_norm in existing_norms:
        return True
    for old in existing_norms:
        if bigram_jaccard(q_norm, old) >= similarity_threshold:
            return True
    return False


def generate_standard_qa(
    target_count: int = 500,
    max_queries_per_doc: int = 2,
    pdf_type_weights: Optional[Dict[str, float]] = None,
    pdf_type_filter: Optional[str] = None,
    similarity_threshold: float = 0.85,
    llm_workers: int = 4,
    enable_llm_check: bool = False,
    dry_run: bool = False,
    min_block_len: int = 100,
) -> Dict:
    """锚点驱动五段式管线主函数。

    参数：
        target_count: 目标生成的 QA 条数（控制总量，最终实际入库 ≈ 该值，允许略少）
        max_queries_per_doc: 每个父块最多生成的问题数
        pdf_type_weights: 五类公告配额 dict，默认 DEFAULT_PDF_TYPE_WEIGHTS
        pdf_type_filter: 只生成某一类公告（如 '融资公告'），None=全部
        similarity_threshold: 问题去重阈值（字符 bigram Jaccard，0.85≈严格去重；0.93≈更严格）
        llm_workers: 并发线程数（每个线程独立 LLM/DB 连接）
        enable_llm_check: 是否启用 LLM 软校验（默认 False：省一半 LLM 调用、生成提速；
                          质量门只剩规则硬校验，弱 QA 会入库，需后续人工/回扫治理。
                          需要时传 True 恢复双门禁）
        dry_run: True=只统计与打印样例，不写库
        min_block_len: 抽样块的最小字符数（过滤短块）
    """
    config = load_config()
    mysql_cfg = config["mysql"]
    llm_cfg = config["llm"]
    router = LLMRouter(api_key=llm_cfg["api_key"], base_url=llm_cfg["base_url"],
                       models=llm_cfg["models"], timeout=llm_cfg["timeout"])
    batch_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]
    print(f"[LLM] 模型轮换列表: {llm_cfg['models']}")

    # Stage 0：抽样
    conn = get_conn(mysql_cfg)
    try:
        ensure_meta_table(conn)
        ensure_qa_evidence_col(conn, mysql_cfg["database"])
        docs = sample_parent_docs(
            conn, target_count, max_queries_per_doc,
            pdf_type_weights or DEFAULT_PDF_TYPE_WEIGHTS,
            pdf_type_filter, min_block_len,
        )
    finally:
        conn.close()

    if not docs:
        return {"status": "no_docs", "reason": "抽样为空", "batch_id": batch_id}

    print(f"[Stage0] 抽样父块 {len(docs)} 个 | batch={batch_id} | target={target_count} | dry_run={dry_run}")
    for d in docs[:10]:
        print(f"  - {d['pdf_type']} | {d.get('name')} ({d.get('code')}) | {str(d.get('heading'))[:20]} | 块长{len(d['parent_content'])}")

    # 并发处理每个块：抽锚点 -> 生成 QA -> 校验
    stats = {"docs_ok": 0, "docs_fail": 0, "qa_generated": 0, "rule_reject": 0, "llm_reject": 0, "dup_reject": 0, "inserted": 0}
    def process_doc(doc: Dict) -> Tuple[List[Tuple[Dict, Dict, str, str, str]], int, int, int, bool]:
        """单块全流程。返回 (通过条目列表, 规则拒绝数, LLM拒绝数, 生成QA数, 是否失败)。"""
        local_conn = get_conn(mysql_cfg) if not dry_run else None
        n_rule_reject = 0
        n_llm_reject = 0
        n_generated = 0
        results = []
        try:
            anchors = extract_anchors(router, doc)
            if not anchors:
                return results, n_rule_reject, n_llm_reject, n_generated, False
            qas = generate_qa(router, doc, anchors, max_queries_per_doc)
            n_generated = len(qas)
            for qa in qas:
                ok, note = rule_check(doc, qa)
                if not ok:
                    n_rule_reject += 1
                    if not dry_run:
                        # 规则被拒也留痕（rule_check='reject' -> 只写 meta，standard_qa_id=NULL，便于人工抽检）
                        insert_qa_meta(local_conn, qa, doc, batch_id, "reject", "na", note)
                    continue
                llm_ok, llm_note = ("pass", "") if not enable_llm_check else llm_check(router, doc, qa)
                if not llm_ok:
                    n_llm_reject += 1
                    if not dry_run:
                        insert_qa_meta(local_conn, qa, doc, batch_id, "pass", "reject", llm_note)
                    continue
                results.append((doc, qa, "pass", "pass" if enable_llm_check else "na", ""))
            return results, n_rule_reject, n_llm_reject, n_generated, False
        except Exception:
            return results, n_rule_reject, n_llm_reject, n_generated, True
        finally:
            if local_conn:
                local_conn.close()

    all_passed: List[Tuple[Dict, Dict, str, str, str]] = []
    with ThreadPoolExecutor(max_workers=llm_workers) as pool:
        futures = {pool.submit(process_doc, d): d for d in docs}
        for fut in as_completed(futures):
            results, n_rule, n_llm, n_generated, failed = fut.result()
            all_passed.extend(results)
            stats["qa_generated"] += n_generated
            stats["rule_reject"] += n_rule
            stats["llm_reject"] += n_llm
            if failed:
                stats["docs_fail"] += 1
            else:
                stats["docs_ok"] += 1

    # Stage 4：去重 + 配额截断 + 落库
    existing_norms: List[str] = []
    inserted = 0
    conn = get_conn(mysql_cfg)
    try:
        ensure_meta_table(conn)
        if not dry_run:
            cur = conn.cursor()
            cur.execute("SELECT question FROM standard_qa WHERE status='active'")
            existing_norms = [normalize_text(r["question"]) for r in cur.fetchall()]

        for doc, qa, rule_c, llm_c, note in all_passed:
            if inserted >= target_count:
                break
            if _is_dup(qa["question"], existing_norms, similarity_threshold):
                stats["dup_reject"] += 1
                continue
            if dry_run:
                inserted += 1
                existing_norms.append(normalize_text(qa["question"]))
                continue
            insert_qa_meta(conn, qa, doc, batch_id, rule_c, llm_c, note)
            inserted += 1
            existing_norms.append(normalize_text(qa["question"]))
    finally:
        conn.close()

    stats["inserted"] = inserted
    stats["llm_error"] = router.error_stats()  # 轮换过程中的失败归因（额度用尽/网络/空响应）

    print(f"\n[Stage3/4] 通过规则校验 {len(all_passed)} 条 | 去重拒绝 {stats['dup_reject']} | 最终入库/试跑 {inserted} 条")
    print(f"统计: {stats}")
    if stats.get("llm_error"):
        print(f"[LLM] 错误归因: {stats['llm_error']}")
    if dry_run and all_passed:
        print("\n===== dry-run 样例（前 5 条）=====")
        for doc, qa, *_ in all_passed[:5]:
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answer']}")
            print(f"  evidence: {qa.get('evidence','')[:80]}")

    return {"status": "ok", "batch_id": batch_id, "stats": stats}


# ============================================================
# 直接调用示例（本文件仅提供函数，不走命令行）
# ============================================================
if __name__ == "__main__":
    print(time.time())
    generate_standard_qa(target_count=100)
    print(time.time())

