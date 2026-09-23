# -*- coding: utf-8 -*-
"""
表格语义化增强模块（双轨方案：检索层增强，保真层不动）

背景
----
Milvus 中表格子块直接向量化原始 Markdown 表格（"| 单元格 | 单元格 |"），
BGE-M3 对其语义理解差，导致表格检索命中率低（"表格语意差"）。

方案
----
- L1 规则转写（TableRuleTranscriber）：markdown 表格 -> 行级"表头: 值"语义句子，
  零成本、数字 100% 保真、可单测。
- L2 LLM 按需增强（TableLLMEnhancer）：仅对 L1 无法语义化的复杂表调用 LLM，
  复用生成层 API Key / Base URL；模型名列表轮换，额度用尽自动切下一个；
  输出做数字校验（防幻觉），失败自动降级 L1。
- 统一输出带 "[表格] " 前缀，仅用于 Milvus 向量化文本，MySQL 原文不动。

用法
----
    from rag_qa.core.table_enhancer import TableEnhancer
    enhancer = TableEnhancer(conf)          # use_llm 默认 True
    vec_text = enhancer.enhance(md_table)   # -> "[表格] ..."
"""
import hashlib
import os
import re
import sys

# 路径保护：支持被 vector_store 导入与独立运行（python rag_qa/core/table_enhancer.py）
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_RAG_QA_DIR = os.path.dirname(_CORE_DIR)
_PROJECT_ROOT = os.path.dirname(_RAG_QA_DIR)
for _p in (_RAG_QA_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base import logger

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:  # openai 库缺失时降级为纯 L1
    OpenAI = None
    _HAS_OPENAI = False


# ============================================================
# HTML 表格 -> Markdown 表格（MinerU 复杂表输出 HTML 时使用）
# ============================================================
class HtmlTableConverter:
    """HTML 表格 -> Markdown 表格（规则转换，零成本）。

    MinerU 对复杂表格（合并单元格等）有时只输出紧凑 HTML
    （`<table><tbody><tr><td>...`），旧代码把原始 HTML 当向量文本入库，
    导致 Milvus 里出现"HTML 语言填充"。此转换器把 HTML 表格结构化为
    Markdown，使后续 L1 规则转写/LLM 都能正确处理。
    - colspan：内容横向重复展开（列数对齐）
    - rowspan：内容纵向延续填充（信息不丢）
    - 内嵌标签/实体（<br>、&nbsp; 等）清理为纯文本
    """

    _ENTITY = {"nbsp": " ", "amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "#160": " "}

    @staticmethod
    def _clean_cell(html: str) -> str:
        text = re.sub(r'<br\s*/?>', ' ', html, flags=re.I)
        text = re.sub(r'<[^>]+>', '', text)

        def ent(m):
            key = m.group(1).lower()
            return HtmlTableConverter._ENTITY.get(key, m.group(0))

        text = re.sub(r'&([a-zA-Z#0-9]+);', ent, text)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _parse_grid(table_html: str) -> list:
        """HTML 表格 -> 单元格二维数组（展开 colspan/rowspan）"""
        rows_html = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.S)
        grid = []
        pending = {}  # col -> [remaining, value]  rowspan 待延续单元格
        for row_html in rows_html:
            cells = re.findall(r'<t[dh]([^>]*)>(.*?)</t[dh]>', row_html, re.S)
            row = []
            col = 0
            for attrs, content in cells:
                # 先填充因上一行 rowspan 延续到本行的单元格
                while col in pending and pending[col][0] > 0:
                    pending[col][0] -= 1
                    row.append(pending[col][1])
                    col += 1
                text = HtmlTableConverter._clean_cell(content)
                attrs_l = attrs.lower()
                m_col = re.search(r'colspan\s*=\s*"?(\d+)"?', attrs_l)
                m_row = re.search(r'rowspan\s*=\s*"?(\d+)"?', attrs_l)
                colspan = int(m_col.group(1)) if m_col else 1
                rowspan = int(m_row.group(1)) if m_row else 1
                start_col = col
                for _ in range(colspan):
                    row.append(text)
                    col += 1
                if rowspan > 1:
                    pending[start_col] = [rowspan - 1, text]
            # 行尾补齐 rowspan 延续
            while col in pending and pending[col][0] > 0:
                pending[col][0] -= 1
                row.append(pending[col][1])
                col += 1
            grid.append(row)
        return grid

    @staticmethod
    def to_markdown(html: str) -> str:
        """HTML 表格块 -> Markdown 表格；未找到 <table> 时返回原文本"""
        m = re.search(r'<table.*?</table>', html, re.S)
        if not m:
            return html
        grid = HtmlTableConverter._parse_grid(m.group(0))
        if not grid:
            return html
        max_cols = max(len(r) for r in grid)
        grid = [r + [""] * (max_cols - len(r)) for r in grid]
        lines = ["| " + " | ".join(grid[0]) + " |",
                 "| " + " | ".join(["---"] * max_cols) + " |"]
        for r in grid[1:]:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)


# ============================================================
# L1：规则转写
# ============================================================
class TableRuleTranscriber:
    """L1：规则转写 —— markdown/HTML 表格 -> 行级"表头: 值"语义句子（数字保真）"""

    _SEP_RE = re.compile(r'^\s*\|?[\s:|-]+\|?\s*$')
    _NUM_RE = re.compile(r'^-?\d+(?:[.,]\d+)*[%％]?$|^-?[0-9,，.]+[%％]?$')
    _CN_EN_RE = re.compile(r'[\u4e00-\u9fffA-Za-z]')

    @staticmethod
    def _is_separator(line: str) -> bool:
        """表格分隔线行：| --- | :---: | 等"""
        t = line.strip().strip('|').strip()
        if not t:
            return False
        return TableRuleTranscriber._SEP_RE.match(line) is not None

    def parse_rows(self, md_table: str) -> list:
        """解析 markdown 表格为单元格二维数组（跳过分隔线行、非表格行）"""
        rows = []
        for raw in md_table.splitlines():
            line = raw.strip()
            if not line.startswith('|') or self._is_separator(line):
                continue
            cells = [c.strip() for c in line.strip('|').split('|')]
            # 丢弃行尾空单元格（markdown 行末 | 产生的空串）
            while cells and cells[-1] == '':
                cells.pop()
            rows.append(cells)
        return rows

    @staticmethod
    def _is_numeric(s: str) -> bool:
        t = s.replace(',', '').replace('，', '').replace('%', '').replace('％', '').strip()
        if not t:
            return False
        try:
            float(t)
            return True
        except ValueError:
            return False

    def transcribe(self, md_table: str) -> str:
        """表格 -> 行级语义句子（HTML 表格先转 Markdown）。

        规则：每行 = "首列指标名：第2列表头 值，第3列表头 值"；
        首列为数字/空时用第一列表头做主语；
        表头整行内容相同（colspan 大标题行，如"易平方系列案统计表"）识别为表格标题。
        解析失败保底返回原文。
        """
        text = (md_table or "").strip()
        if not text:
            return ""
        if "<table" in text.lower():
            text = HtmlTableConverter.to_markdown(text)
        rows = self.parse_rows(text)
        if not rows:
            return text  # 保底原文

        header = rows[0]
        title = ""
        # colspan 大标题行：表头所有单元格内容相同且列数>1 → 提取为表格标题
        if len(header) > 1 and header[0] and all(h == header[0] for h in header):
            title = header[0]
            header = rows[1] if len(rows) > 1 else []
            body = rows[2:]
        else:
            body = rows[1:]
        if not body:
            head_text = "、".join(h for h in header if h)
            return f"{title}：{head_text}" if title else head_text

        sentences = []
        for row in body:
            cells = (list(row) + [""] * len(header))[:len(header)]
            subject = cells[0].strip()
            start = 1
            if not subject or self._is_numeric(subject):
                subject = header[0].strip() if header else ""
            parts = []
            for i in range(start, len(header)):
                h = header[i].strip()
                v = cells[i].strip()
                if not h or not v or v == h:
                    continue
                parts.append(f"{h} {v}")
            if parts:
                sentences.append(f"{subject}：" + "，".join(parts))
            elif subject:
                sentences.append(subject)
        out = "；".join(s for s in sentences if s)
        if title and out:
            out = f"{title}：" + out
        return out or text

    def needs_llm(self, md_table: str, transcribed: str) -> bool:
        """判断 L1 转写后是否仍需要 LLM 增强（只对少数复杂表启用，控制 API 成本）。

        触发条件（任一）：
        1. 规则解析失败（含无法转换的 HTML）；
        2. 无表头；
        3. 表头无中英文语义（纯数字/符号矩阵）；
        4. 多列纯数字矩阵（列数>=4 且数据单元格几乎无文字）。
        HTML 表格会先转 Markdown 再判定，转换后能正常转写的不再强制走 LLM。
        """
        text = (md_table or "").strip()
        if "<table" in text.lower():
            text = HtmlTableConverter.to_markdown(text)
        rows = self.parse_rows(text)
        if not rows:
            return True
        header = rows[0]
        # colspan 大标题行 → 提为标题，用下一行判定
        if len(header) > 1 and header[0] and all(h == header[0] for h in header):
            if len(rows) > 1:
                header = rows[1]
            else:
                return True
        header_clean = [h.strip() for h in header if h.strip()]
        if not header_clean:
            return True
        if not any(self._CN_EN_RE.search(h) for h in header_clean):
            return True
        body = rows[1:]
        if len(header_clean) >= 4 and body:
            total = 0
            text_cells = 0
            for r in body:
                for c in (list(r) + [""] * len(header))[:len(header)]:
                    total += 1
                    if self._CN_EN_RE.search(c.strip()):
                        text_cells += 1
            if total and text_cells / total < 0.15:
                return True
        return False


# ============================================================
# L2：LLM 按需增强（模型列表轮换 + 额度用尽自动切换）
# ============================================================
_QUOTA_KEYWORDS = (
    "quota", "insufficient", "balance", "额度", "欠费", "余额不足",
    "payment required", "out of quota", "credit", "accepted balance",
    "超过限制", "调用额度", "trial expired", "expired", "账户余额",
)

_TABLE_SYSTEM_PROMPT = (
    "你是金融公告表格解析助手。请把用户给出的 Markdown 表格改写为一段自然语言描述，用于向量检索。要求：\n"
    "1. 只描述表格内容，不得添加表格中不存在的数字、指标、日期或事实；\n"
    "2. 关键数字（金额、比例、日期、数量）必须与表格原文完全一致；\n"
    "3. 用 1-3 句话概括表格核心信息，语义通顺；\n"
    "4. 直接输出描述文本，不要输出 Markdown 表格、不要解释。"
)


class TableLLMEnhancer:
    """L2：LLM 按需增强 —— 模型名列表轮换，额度用尽自动切换下一个"""

    _NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')
    _UNIT_RE = re.compile(r'(-?\d+(?:\.\d+)?)\s*([万亿])')

    def __init__(self, api_key: str, base_url: str, models, timeout: int = 30):
        if not _HAS_OPENAI or not api_key:
            logger.warning("[table_enhancer] openai 库或 API Key 不可用，L2 LLM 增强禁用")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.models = [m for m in (models or []) if m]
        self._cursor = 0          # 环形轮换游标（下次从下一个模型开始，避免重复命中失败模型）
        self._failed = set()      # 已确认不可用的模型（额度用尽/调用失败）
        self._cache = {}          # 表格内容 hash -> 总结文本（跨文件相同表格不重复调用）

    # ---------- 额度用尽判定 ----------
    @staticmethod
    def _is_quota_error(exc) -> bool:
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

    # ---------- 模型轮换 ----------
    def _next_model(self):
        """返回下一个未失败的模型；全部失败返回 None"""
        if not self.models:
            return None
        for _ in range(len(self.models)):
            model = self.models[self._cursor % len(self.models)]
            self._cursor += 1
            if model not in self._failed:
                return model
        return None

    # ---------- 数字校验（防幻觉） ----------
    @classmethod
    def _extract_numeric_values(cls, text: str) -> set:
        """提取文本中的纯数值（不做万/亿换算，因为金融表格的单位在表头不在单元格）。
        LLM 总结时会把表头单位加回去（如 10.00 万份），这里只对比数字本身。"""
        return {float(m.group(0)) for m in cls._NUM_RE.finditer(text)}

    @staticmethod
    def _verify_numbers(summary: str, md_table: str) -> bool:
        """防幻觉：总结中的大额数字（|值| >= 1万）必须能在原文中找到（含万/亿换算，1% 容差）。

        只校验大额绝对值：金融表格里最危险的幻觉是编造金额；
        百分比/小额派生指标（如"同比增长 10%"）不校验，避免误伤 LLM 的合理计算表达。
        """
        src = TableLLMEnhancer._extract_numeric_values(md_table)
        dst = TableLLMEnhancer._extract_numeric_values(summary)
        for d in dst:
            if abs(d) < 10000:
                continue
            if not any(abs(d - s) <= max(1.0, abs(s) * 0.01) for s in src):
                return False
        return True

    # ---------- 主调用 ----------
    def summarize(self, md_table: str) -> str:
        """调用 LLM 生成表格语义化描述；失败/额度用尽自动切换模型；全部失败 raise RuntimeError（批处理应暂停等人工续费）"""
        if self.client is None or not self.models:
            return ""
        cache_key = hashlib.md5(md_table.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        import time as _time
        max_rounds = 3       # 临时错误（网络/超时）最多重试 3 轮
        wait_sec = 60        # 每轮之间等 60 秒，等网络恢复

        for round_no in range(1, max_rounds + 1):
            temp_failed_this_round = []
            for _ in range(len(self.models)):
                model = self._next_model()
                if model is None:
                    break
                try:
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": _TABLE_SYSTEM_PROMPT},
                            {"role": "user", "content": md_table},
                        ],
                        timeout=30,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    if not text:
                        logger.warning(f"[table_enhancer] 模型 {model} 返回空内容，切换下一个模型")
                        continue
                    if not self._verify_numbers(text, md_table):
                        logger.warning(f"[table_enhancer] 模型 {model} 输出数字校验未通过（疑似幻觉），降级 L1")
                        return ""
                    self._cache[cache_key] = text
                    return text
                except Exception as e:
                    if self._is_quota_error(e):
                        logger.warning(f"[table_enhancer] 模型 {model} 额度用尽，自动切换下一个模型: {e}")
                        self._failed.add(model)
                    else:
                        # 临时错误（网络/超时），不永久拉黑，记下来等下轮重试
                        logger.warning(f"[table_enhancer] 模型 {model} 临时失败（网络/超时），记下等下轮重试: {e}")
                        temp_failed_this_round.append(model)

            # 一轮跑完，看是不是还有可用模型
            if self._next_model() is not None:
                # 还有未失败的模型，继续试（应该不会走到这里，因为上面 for 已经遍历完）
                continue
            # 所有模型都试过了，如果还有临时失败的，等下一轮
            if temp_failed_this_round and round_no < max_rounds:
                logger.warning(f"[table_enhancer] 第 {round_no}/{max_rounds} 轮全部临时失败，{wait_sec}s 后重试...")
                _time.sleep(wait_sec)
                # 清空本轮临时失败标记，让下轮重新试这些模型
                for m in temp_failed_this_round:
                    self._failed.discard(m)
                continue
            break

        failed_models = sorted(self._failed)
        raise RuntimeError(
            f"[table_enhancer] 所有 LLM 模型均不可用（已尝试 {len(failed_models)} 个：{failed_models}，"
            f"临时错误重试 {max_rounds} 轮后仍失败）。"
            f"请检查 API Key 额度/网络后手动重跑断点；本次入库已暂停。"
        )


# ============================================================
# 组合入口
# ============================================================
class TableEnhancer:
    """表格语义化组合入口：L1 规则转写 + L2 LLM 按需，统一加 [表格] 前缀"""

    def __init__(self, config, use_llm: bool = True):
        self.rule = TableRuleTranscriber()
        self.llm = None
        if use_llm:
            models = getattr(config, "TABLE_LLM_MODELS", None) or []
            if models:
                self.llm = TableLLMEnhancer(
                    api_key=getattr(config, "DASHSCOPE_API_KEY", ""),
                    base_url=getattr(config, "DASHSCOPE_BASE_URL", ""),
                    models=models,
                    timeout=getattr(config, "TABLE_LLM_TIMEOUT", 30),
                )
            else:
                logger.info("[table_enhancer] 未配置 TABLE_LLM_MODELS，仅使用 L1 规则转写")

    def enhance(self, md_table: str) -> str:
        """表格文本 -> "[表格] " + 语义化描述（LLM 失败自动降级 L1）。

        HTML 表格（MinerU 复杂表输出）先统一转 Markdown，保证 L1/LLM 输入干净。
        """
        text = (md_table or "").strip()
        if not text:
            return ""
        if "<table" in text.lower():
            text = HtmlTableConverter.to_markdown(text)
        transcribed = self.rule.transcribe(text)
        if self.llm is not None and self.rule.needs_llm(text, transcribed):
            summary = self.llm.summarize(text)
            if summary:
                return "[表格] " + summary
        return "[表格] " + transcribed


if __name__ == "__main__":
    # 手动冒烟
    demo = (
        "| 指标 | 2023年 | 2024年 |\n"
        "| --- | --- | --- |\n"
        "| 营业收入 | 100亿 | 120亿 |\n"
        "| 净利润 | 10亿 | 15亿 |"
    )
    e = TableEnhancer(type("C", (), {
        "TABLE_LLM_MODELS": [],
        "DASHSCOPE_API_KEY": "",
        "DASHSCOPE_BASE_URL": "",
        "TABLE_LLM_TIMEOUT": 30,
    })(), use_llm=False)
    print(e.enhance(demo))
