"""入库后自动质检门（Quality Gate）。

每个 PDF 文件入库完成后调用 `check_file(file_id)`，对 MySQL child_chunks
与 Milvus 双向对账，输出结构化报告。ERROR 级问题抛出，让批处理把该文件标 failed。

检查项：
  Milvus 侧：HTML 标签残留、LaTeX 命令残留、空块、纯符号块、page_no=0 占比、
            子块长度分布、表格块占比
  MySQL 侧：child_chunks 空 content、page_no NULL 占比
  对账：MySQL child_chunks 数 vs Milvus 子块数
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------- 阈值 ----------
HTML_TAG_RE = re.compile(r"<\s*/?\s*(table|tbody|tr|td|th|div|span|p|br|h\d)[^>]*>", re.IGNORECASE)
LATEX_CMD_RE = re.compile(r"\\(?:mathrm|mathbf|text|times|div|frac|left|right|operatorname|boldsymbol)")
SYMBOLS_ONLY_RE = re.compile(r"^[\s\W_]*$")

ERROR_HTML = 0          # Milvus HTML 残留 > 0 → ERROR
ERROR_LATEX = 0         # Milvus LaTeX 残留 > 0 → ERROR
ERROR_EMPTY_CHUNK = 0   # Milvus 空块(<10字) > 0 → ERROR
WARN_PAGE0_PCT = 20.0   # page_no=0 占比 > 20% → WARNING
WARN_AVG_LEN_MIN = 100
WARN_AVG_LEN_MAX = 400


@dataclass
class QGReport:
    file_id: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary_line(self) -> str:
        m = self.metrics
        flag = "OK " if self.ok else "ERR"
        return (
            f"[QG] file_id={self.file_id} {flag} | "
            f"chunks={m.get('milvus_chunks')} tables={m.get('table_chunks')} "
            f"html={m.get('html_residual')} latex={m.get('latex_residual')} "
            f"p0={m.get('page0_pct')}% avg_len={m.get('avg_len')}"
            + (f" WARN={len(self.warnings)}" if self.warnings else "")
            + (f" ERR={len(self.errors)}" if self.errors else "")
        )


def check_file(file_id: int,
               mysql_client=None,
               milvus_client=None,
               milvus_db: str = "df_finance_rag_v4",
               milvus_collection: str = "df_announce_chunks") -> QGReport:
    """对指定 file_id 的入库结果跑质检。客户端不传则自建。"""
    report = QGReport(file_id=file_id)
    own_mysql = own_milvus = False

    try:
        # ---------- MySQL 侧 ----------
        if mysql_client is None:
            from mysql_qa.db.mysql_client import MySQLClient
            mysql_client = MySQLClient()
            own_mysql = True
        mc = mysql_client

        # 该 file 的所有 parent_docs id
        mc.cursor.execute(
            "SELECT id FROM parent_docs WHERE file_id = %s", (file_id,))
        parent_ids = [r["id"] for r in mc.cursor.fetchall()]
        if not parent_ids:
            report.errors.append(f"file_id={file_id} 在 parent_docs 表无记录")
            return report

        # 查 child_chunks
        fmt = ",".join(["%s"] * len(parent_ids))
        mc.cursor.execute(
            f"SELECT id, parent_id, content, is_table, page_no, CHAR_LENGTH(content) AS clen "
            f"FROM child_chunks WHERE parent_id IN ({fmt})", parent_ids)
        rows = mc.cursor.fetchall()
        mysql_chunks = len(rows)
        empty_mysql = sum(1 for r in rows if not (r["content"] or "").strip())
        page_no_null = sum(1 for r in rows if r["page_no"] is None)

        # ---------- Milvus 侧 ----------
        if milvus_client is None:
            from pymilvus import MilvusClient
            milvus_client = MilvusClient(uri="http://127.0.0.1:19530", db_name=milvus_db)
            own_milvus = True
        mc2 = milvus_client
        mc2.flush(milvus_collection)

        pid_strs = [str(p) for p in parent_ids]
        pid_fmt = ",".join(f'"{p}"' for p in pid_strs)
        mv_rows = mc2.query(
            collection_name=milvus_collection,
            filter=f"parent_id in [{pid_fmt}]",
            output_fields=["id", "text", "page_no"],
            limit=16384,
        )
        mv_texts = [r.get("text", "") or "" for r in mv_rows]
        mv_pages = [r.get("page_no", 0) or 0 for r in mv_rows]

        # ---------- 指标 ----------
        lengths = [len(t) for t in mv_texts]
        report.metrics.update({
            "mysql_chunks": mysql_chunks,
            "milvus_chunks": len(mv_texts),
            "table_chunks": sum(1 for t in mv_texts if t.startswith("[表格]")),
            "html_residual": sum(1 for t in mv_texts if HTML_TAG_RE.search(t)),
            "latex_residual": sum(1 for t in mv_texts if LATEX_CMD_RE.search(t)),
            "empty_chunks": sum(1 for t in mv_texts if len(t.strip()) < 10),
            "symbol_only": sum(1 for t in mv_texts if SYMBOLS_ONLY_RE.match(t.strip())),
            "page0_pct": round(100.0 * sum(1 for p in mv_pages if p == 0) / max(1, len(mv_pages)), 1),
            "avg_len": round(sum(lengths) / max(1, len(lengths)), 1),
            "max_len": max(lengths) if lengths else 0,
            "mysql_empty": empty_mysql,
            "mysql_page_no_null": page_no_null,
        })

        # ---------- 判定 ----------
        if report.metrics["html_residual"] > ERROR_HTML:
            report.errors.append(
                f"Milvus HTML 残留 {report.metrics['html_residual']} 条")
        if report.metrics["latex_residual"] > ERROR_LATEX:
            report.errors.append(
                f"Milvus LaTeX 残留 {report.metrics['latex_residual']} 条")
        if report.metrics["empty_chunks"] > ERROR_EMPTY_CHUNK:
            report.errors.append(
                f"Milvus 空块 {report.metrics['empty_chunks']} 条")
        if report.metrics["mysql_chunks"] != report.metrics["milvus_chunks"]:
            report.errors.append(
                f"双库条数不一致: MySQL={mysql_chunks} Milvus={len(mv_texts)}")
        if report.metrics["mysql_empty"] > 0:
            report.errors.append(
                f"MySQL child_chunks 空 content {report.metrics['mysql_empty']} 条")

        if report.metrics["page0_pct"] > WARN_PAGE0_PCT:
            report.warnings.append(
                f"page_no=0 占比 {report.metrics['page0_pct']}% 偏高")
        if report.metrics["avg_len"] < WARN_AVG_LEN_MIN or report.metrics["avg_len"] > WARN_AVG_LEN_MAX:
            report.warnings.append(
                f"平均子块长度 {report.metrics['avg_len']} 超出 [{WARN_AVG_LEN_MIN},{WARN_AVG_LEN_MAX}]")
        if report.metrics["symbol_only"] > 0:
            report.warnings.append(
                f"纯符号块 {report.metrics['symbol_only']} 条")

        logger.info(report.summary_line())
        for w in report.warnings:
            logger.warning(f"[QG][WARN] file_id={file_id}: {w}")
        for e in report.errors:
            logger.error(f"[QG][ERR ] file_id={file_id}: {e}")
        return report

    finally:
        if own_mysql:
            try:
                mysql_client.close()
            except Exception:
                pass
        if own_milvus:
            try:
                milvus_client.close()
            except Exception:
                pass


if __name__ == "__main__":
    # 命令行自检：python quality_gate.py <file_id>
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    fid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rep = check_file(fid)
    print(rep.summary_line())
    print("metrics:", rep.metrics)
    sys.exit(0 if rep.ok else 1)
