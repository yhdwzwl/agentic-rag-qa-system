# -*- coding:utf-8 -*-
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
import torch.cuda
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
import time
from collections import defaultdict
import sys, os

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

from base import logger, Config
from mysql_qa.db.mysql_client import MySQLClient
# 表格语义化增强（双轨：只改 Milvus 向量化文本，MySQL 原文不动；模型额度用尽自动切换）
from rag_qa.core.table_enhancer import TableEnhancer

import re as _re

# 数字上下标 Unicode 映射
_SUBS = {'0':'\u2080','1':'\u2081','2':'\u2082','3':'\u2083','4':'\u2084',
         '5':'\u2085','6':'\u2086','7':'\u2087','8':'\u2088','9':'\u2089'}
_SUPS = {'0':'\u2070','1':'\u00b9','2':'\u00b2','3':'\u00b3','4':'\u2074',
         '5':'\u2075','6':'\u2076','7':'\u2077','8':'\u2078','9':'\u2079'}
# 常用 LaTeX 运算符 -> Unicode
_LATEX_OPS = {
    'times':'\u00d7','div':'\u00f7','cdot':'\u00b7','pm':'\u00b1','mp':'\u2213',
    'leq':'\u2264','leqslant':'\u2266','geq':'\u2265','geqslant':'\u2267','neq':'\u2260',
    'approx':'\u2248','equiv':'\u2261','in':'\u2208','notin':'\u2209',
    'rightarrow':'\u2192','leftarrow':'\u2190','Rightarrow':'\u21d2','Leftarrow':'\u21d0',
    'sum':'\u2211','prod':'\u220f','infty':'\u221e',
    'alpha':'\u03b1','beta':'\u03b2','gamma':'\u03b3','delta':'\u03b4','theta':'\u03b8',
    'pi':'\u03c0','lambda':'\u03bb','mu':'\u03bc','sigma':'\u03c3','phi':'\u03c6','omega':'\u03c9',
    'Delta':'\u0394','Sigma':'\u03a3','Omega':'\u03a9',
}

def latex_to_text(text: str) -> str:
    """把 LaTeX 数学式轻量转成可读 Unicode 文本（仅用于 Milvus 向量化层；MySQL 原文保真）。
    处理：去 $$/$ 包裹、\mathrm/\mathbf/\text 去壳、常用运算符->Unicode、数字上下标->Unicode。
    未识别命令保留原样，不做激进替换，避免改错语义。"""
    if not text or ('\\' not in text and '$' not in text):
        return text
    s = text.replace('$$', '').replace('$', '')
    # 数字上下标先转 Unicode（必须先于去壳：\mathrm{P_{0}} 中 {0} 嵌套在花括号内，
    # 先把 _{0} 转成 ₀，去壳正则才不会因嵌套花括号失败）
    s = _re.sub(r'_\s*\{\s*(\d+)\s*\}', lambda m: ''.join(_SUBS[d] for d in m.group(1)), s)
    s = _re.sub(r'_\s*(\d)', lambda m: _SUBS[m.group(1)], s)
    s = _re.sub(r'\^\s*\{\s*(\d+)\s*\}', lambda m: ''.join(_SUPS[d] for d in m.group(1)), s)
    s = _re.sub(r'\^\s*(\d)', lambda m: _SUPS[m.group(1)], s)
    # \command{arg} -> arg（去字体壳；此时花括号内已无嵌套上下标）
    s = _re.sub(r'\\(?:mathrm|mathbf|text|boldsymbol|mathcal|mathit|operatorname)\s*\{([^{}]*)\}', r'\1', s)
    # 运算符替换
    for cmd, sym in _LATEX_OPS.items():
        s = s.replace('\\' + cmd + ' ', sym).replace('\\' + cmd, sym)
    # 去 \left \right \, \; \! \  等空白/分组命令
    s = _re.sub(r'\\(?:left|right|,|;|!|\s)\b', '', s)
    # 清理残留空格
    s = _re.sub(r'[ \t]{2,}', ' ', s)
    s = _re.sub(r' *\n *', '\n', s)
    return s.strip()


def _median(values):
    """取中位数（仅用于 rerank 分数统计；空序列返回 0.0）"""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


conf = Config()


# core/vector_store.py
# 定义 VectorStore 类，封装向量存储和检索功能
# 架构：Milvus 存子块向量 + 元数据，MySQL 存父块内容 + 子块元数据
class VectorStore:
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"使用设备：{self.device}")

        # 初始化重排序模型
        reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        self.reranker = CrossEncoder(reranker_path, device=self.device)

        # 初始化 BGE-M3 嵌入
        m3_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,
            use_fp16=False,
            device=self.device
        )
        self.dense_dim = self.embedding_function.dim["dense"]

        # 初始化 Milvus：先确保 database 存在
        self._ensure_database_exists()
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        self._create_or_load_collection()

        # 初始化 MySQL
        self.mysql = MySQLClient()
        # 最近一次混合检索的分段耗时（milvus_search/mysql_fetch/rerank/total，秒）
        # 由 hybrid_search_with_rerank 每次调用时重置并填充，供 new_rag_system 落评估库
        self.last_timings = {}
        # 表格语义化增强器（懒加载：首次遇到表格子块才初始化，避免启动时网络开销）
        self._table_enhancer = None
        self.logger.info("VectorStore 初始化完成")

    def _ensure_database_exists(self):
        """确保目标 database 存在，不存在则自动创建"""
        temp_client = MilvusClient(uri=f"http://{self.host}:{self.port}")
        try:
            existing_dbs = temp_client.list_databases()
            if self.database not in existing_dbs:
                temp_client.create_database(self.database)
                self.logger.info(f"自动创建 Milvus 数据库: {self.database}")
            else:
                self.logger.info(f"Milvus 数据库已存在: {self.database}")
        finally:
            temp_client.close()

    def _create_or_load_collection(self):
        """创建或加载 Milvus Collection"""
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # parent_id 改为关联 MySQL 的数字 ID（字符串形式）
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=50)
            # 不再存 parent_content（移到 MySQL）
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)
            # 子块在原始 PDF 中的页码（1 基；解析无法定位时为 0）
            schema.add_field(field_name="page_no", datatype=DataType.INT32)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"已创建集合 {self.collection_name}")
        else:
            logger.info(f"已加载集合 {self.collection_name}")

        self.client.load_collection(self.collection_name)

    def add_documents(self, documents, extracted_images=None, file_id=None):
        """
        写入文档：先写 MySQL 拿 parent_id，再写 Milvus
        流程：
            1. 按 parent_id 分组子块
            2. 对每个父块：写 MySQL parent_docs → 拿 doc_id
            3. 写 MySQL child_chunks → 关联 parent_id
            4. 写 Milvus → parent_id = str(doc_id)
        """
        if not documents:
            logger.warning("没有文档需要写入")
            return

        # 1. 按 parent_id 分组
        parent_groups = defaultdict(list)
        parent_meta = {}
        for doc in documents:
            pid = doc.metadata.get("parent_id", "unknown")
            parent_groups[pid].append(doc)
            parent_meta[pid] = {
                "parent_content": doc.metadata.get("parent_content", ""),
                "heading": doc.metadata.get("heading", ""),
                "heading_level": doc.metadata.get("heading_level", 0),
                "heading_path": doc.metadata.get("heading_path", ""),
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp") or doc.metadata.get("processed_at", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "file_path": doc.metadata.get("file_path", ""),
                "file_id": doc.metadata.get("file_id", None),
                "parse_method": doc.metadata.get("parse_method", ""),
                "complexity_score": doc.metadata.get("complexity_score", 0),
                "code": doc.metadata.get("code", ""),
                "name": doc.metadata.get("name", ""),
                "pdf_time": doc.metadata.get("pdf_time", ""),
                "doc_type": doc.metadata.get("doc_type", ""),
            }

        logger.info(f"共 {len(parent_groups)} 个父块，{len(documents)} 个子块待写入")

        # 2. 逐父块写入
        milvus_data = []
        pending_doc_ids = []  # 已写 MySQL 的父块 doc_id，Milvus upsert 成功后才标记 indexed（2026-10-01 修复时序）
        written_img_idx = set()  # 文件内已写入的图片索引（防父块重叠导致同一图重复入库）
        for orig_pid, children in parent_groups.items():
            meta = parent_meta[orig_pid]

            # 写 MySQL parent_docs
            doc_id = self.mysql.insert_parent_doc(
                file_id=meta.get("file_id"),
                file_name=meta["file_name"],
                file_path=meta.get("file_path", ""),
                pdf_type=meta.get("source", ""),
                code=meta.get("code", ""),
                name=meta.get("name", ""),
                pdf_time=meta.get("pdf_time", ""),
                doc_type=meta.get("doc_type", ""),
                parse_method=meta["parse_method"],
                complexity_score=meta["complexity_score"],
                parent_content=meta["parent_content"],  # 父块完整内容
                heading=meta.get("heading"),
                heading_level=meta.get("heading_level"),
                heading_path=meta.get("heading_path"),
            )
            logger.info(f"  MySQL parent_docs 写入成功, doc_id={doc_id}")

            # 写 MySQL child_chunks
            for idx, child in enumerate(children):
                content = child.page_content
                # 自动识别子块特征：优先用分块器的表格判定（已做伪表格过滤），
                # 兜底再按内容特征重算，避免伪表格/普通文本被误判为表格
                chunk_type, is_table, has_number = self._detect_chunk_features(
                    content, child.metadata.get("is_table"))
                chunk_id = self.mysql.insert_child_chunk(
                    parent_id=doc_id,
                    chunk_index=idx,
                    content=content,
                    chunk_type=chunk_type,
                    is_table=is_table,
                    has_number=has_number,
                    heading=child.metadata.get("heading"),
                    heading_level=child.metadata.get("heading_level"),
                    heading_path=child.metadata.get("heading_path"),
                    page_no=child.metadata.get("page_no"),
                )
                # 处理子块里的图片占位符
                if extracted_images:
                    import re
                    image_ids = []
                    placeholders = re.findall(r'\[IMAGE_(\d+)\]', content)
                    for ph in placeholders:
                        img_idx = int(ph)
                        if img_idx in written_img_idx:
                            continue  # 同一图片已在前面子块写入（父块重叠区域），跳过
                        if img_idx < len(extracted_images):
                            img = extracted_images[img_idx]
                            image_id = self.mysql.insert_image(
                                file_id=file_id,
                                parent_id=doc_id,
                                chunk_id=chunk_id,
                                chunk_index=idx,
                                image_data=img['image_data'],
                                image_type=img['image_type'],
                                image_caption=img.get('image_caption', ''),
                                image_size_kb=len(img['image_data']) / 1024
                            )
                            written_img_idx.add(img_idx)
                            image_ids.append(image_id)
                    if image_ids:
                        self.mysql.update_chunk_image_ids(chunk_id, image_ids)
                        logger.info(f"    子块 {idx} 关联 {len(image_ids)} 张图片")
                # 写 Milvus
                text = child.page_content
                # 表格子块语义化增强（双轨：只改向量化文本；MySQL content 保持原文，保证生成阶段数字保真）
                if is_table and conf.TABLE_ENHANCE_ENABLED:
                    text = self._get_table_enhancer().enhance(text)
                # LaTeX 轻量去命令化（双轨：只改向量化文本；MySQL content 保留原始 LaTeX）
                text = latex_to_text(text)
                embeddings = self.embedding_function([text])
                dense_vec = embeddings["dense"][0]
                # scipy 稀疏矩阵不支持 [0] 切片，用 [[0]]
                sparse_vec = self._to_sparse_dict(embeddings["sparse"][[0]])

                # 用 file_name + parent_id + text 组合算 hash，保证不同文件的相同文本不会冲突
                id_raw = f"{meta['file_name']}_{orig_pid}_{idx}_{text}"
                text_hash = hashlib.md5(id_raw.encode('utf-8')).hexdigest()
                milvus_data.append({
                    "id": text_hash,
                    "text": text,
                    "dense_vector": dense_vec,
                    "sparse_vector": sparse_vec,
                    "parent_id": str(doc_id),  # 关联 MySQL 的 doc_id
                    "source": meta["source"],
                    "timestamp": meta["timestamp"],
                    "page_no": child.metadata.get("page_no") or 0,
                    # 标量字段（collection 已开 enable_dynamic_field，自动进动态字段；
                    # 供表格类 query 用 is_table == 1 过滤；存量记录由回填脚本补齐）
                    # 注：chunk_type 与 is_table 完全同义（纯冗余），按用户要求不写入 Milvus
                    "is_table": is_table,
                    "has_number": has_number,
                })

            # 状态延后：Milvus upsert 成功后再统一标记 indexed（见第 3 步）
            pending_doc_ids.append(doc_id)

        # 3. 批量写 Milvus —— 必须先成功再标记 indexed；
        #    若 upsert 失败则标记 failed 并抛出，避免"MySQL 已入库但 Milvus 无向量"的静默丢失
        #    （2026-10-01 修复：原实现先标 indexed 后 upsert，Milvus 写入失败时状态失真）
        try:
            if milvus_data:
                self.client.upsert(collection_name=self.collection_name, data=milvus_data)
                logger.info(f"Milvus 已插入 {len(milvus_data)} 条子块")
        except Exception as e:
            for doc_id in pending_doc_ids:
                try:
                    self.mysql.update_doc_status(doc_id, "failed", error_msg=str(e)[:500])
                except Exception:
                    pass  # 状态标记失败不掩盖原始异常
            raise
        # upsert 成功（或无子块）后统一标记 indexed
        for doc_id in pending_doc_ids:
            self.mysql.update_doc_status(doc_id, "indexed")

    def delete_by_file(self, file_id: int):
        """
        按 file_id 清理该文件的全部残留数据（MySQL + Milvus），用于断点续跑/崩溃恢复。
        清理后该文件可安全重新入库，避免重复或孤儿数据。
        """
        # 1. 先按 file_id 查父块；历史数据 file_id 可能为 None，再用 file_name 兜底
        parent_ids = []
        fname = None
        self.mysql.cursor.execute("SELECT file_name FROM pdf_files WHERE id=%s", (file_id,))
        row = self.mysql.cursor.fetchone()
        if row:
            fname = row["file_name"]
        if fname is None:
            logger.warning(f"delete_by_file: pdf_files 无记录 id={file_id}，跳过清理")
            return

        self.mysql.cursor.execute(
            "SELECT id FROM parent_docs WHERE file_id=%s OR file_name=%s", (file_id, fname))
        parent_ids = [str(r["id"]) for r in self.mysql.cursor.fetchall()]

        # 2. 删 Milvus 向量（按 parent_id）
        if parent_ids:
            expr = "parent_id in [" + ",".join(f'"{p}"' for p in parent_ids) + "]"
            try:
                self.client.delete(collection_name=self.collection_name, filter=expr)
                logger.info(f"  Milvus 清理 {len(parent_ids)} 个父块对应向量: {expr}")
            except Exception as e:
                logger.warning(f"  Milvus 清理失败（忽略，重新入库会覆盖）: {e}")

            # 3. 删 MySQL：先子表后父表
            ph = ",".join(["%s"] * len(parent_ids))
            self.mysql.cursor.execute(f"DELETE FROM child_chunks WHERE parent_id IN ({ph})", parent_ids)
            self.mysql.cursor.execute(f"DELETE FROM doc_status WHERE parent_id IN ({ph})", parent_ids)
            self.mysql.cursor.execute(f"DELETE FROM parent_docs WHERE id IN ({ph})", parent_ids)
            # 3.5 删 doc_images（按 file_id；旧实现漏删导致孤儿图片随重跑累积）
            self.mysql.cursor.execute("DELETE FROM doc_images WHERE file_id=%s", (file_id,))
            self.mysql.connection.commit()
            logger.info(f"  MySQL 清理完成: {len(parent_ids)} 个父块（file_id={file_id}），含 doc_images")
        else:
            logger.info(f"  delete_by_file: 无残留父块（file_id={file_id}）")

    def _to_sparse_dict(self, row):
        """将稀疏向量转成 Milvus 需要的 {index: value} 格式"""
        sparse_vector = {}
        try:
            if hasattr(row, 'col'):  # coo_array
                indices = row.col
                values = row.data
            else:  # csr_matrix
                indices = row.indices
                values = row.data
            for idx, value in zip(indices, values):
                sparse_vector[int(idx)] = float(value)
        except Exception as e:
            logger.warning(f"稀疏向量转换失败: {e}")
        return sparse_vector

    def _get_table_enhancer(self):
        """懒加载表格语义化增强器（只在启用且首次遇到表格子块时初始化）"""
        if self._table_enhancer is None:
            self._table_enhancer = TableEnhancer(conf, use_llm=conf.TABLE_ENHANCE_ENABLED)
        return self._table_enhancer

    def _detect_chunk_features(self, content: str, splitter_is_table=None):
        """
        自动识别子块特征
        返回: (chunk_type, is_table, has_number)
        splitter_is_table: 分块器在伪表格过滤后给出的表格判定（0/1/None）。
        有判定时优先采用；无判定时兜底要求"存在标准表格分隔线"才算表格，
        避免 markitdown 伪表格（大量 | 但无表格结构）入库被误判。
        """
        if not content:
            return 'text', 0, 0

        if splitter_is_table is not None:
            is_table = 1 if splitter_is_table else 0
        else:
            # 兜底判定：标准 Markdown 表格必须有分隔线行（| --- | --- |）
            lines = content.split('\n')
            has_sep = any(('---' in l) and re.search(r'\|.*\|', l) for l in lines)
            pipe_count = content.count('|')
            line_count = content.count('\n')
            is_table = 1 if (has_sep and pipe_count >= 5 and line_count >= 2) else 0
        chunk_type = 'table' if is_table else 'text'

        # 2. 判断有没有数字/金额
        import re
        has_number = 1 if re.search(r'\d+\.?\d*', content) else 0

        return chunk_type, is_table, has_number

    def get_query_vector(self, query: str):
        """生成query的稠密向量（给Redis缓存用）"""
        query_embeddings = self.embedding_function([query])
        return query_embeddings["dense"][0]

    def _build_filter_expr(self, source_filter: str = None, table_only: bool = False) -> str:
        """构建 Milvus 检索过滤表达式：source 与 is_table 条件 AND 叠加。

        - source_filter: 公告类型（如 '重大事项'），原样拼 `source == '...'`
        - table_only: 表格类 query 时叠加 `is_table == 1`（标量字段，动态字段同样可过滤）
        无任何条件时返回空字符串（Milvus 表示不过滤）。
        """
        parts = []
        if source_filter:
            parts.append(f"source == '{source_filter}'")
        if table_only:
            parts.append("is_table == 1")
        return " and ".join(parts)

    def _hybrid_search(self, dense_query_vector, sparse_query_vector, k, filter_expr, output_fields):
        """Milvus 稠密+稀疏混合检索（WeightedRanker 1.0/0.7），返回 hits 列表（可能为空）。"""
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )
        return self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=WeightedRanker(1.0, 0.7),
            limit=k,
            output_fields=output_fields
        )[0]

    # Milvus 不可用时的关键词降级检索停用词（仅用于兜底，勿与检索主词表混淆）
    _FALLBACK_STOPWORDS = {
        "的", "了", "吗", "呢", "啊", "吧", "么", "呀", "哦", "嘛",
        "怎么", "什么", "哪些", "多少", "几个", "哪种", "如何", "怎样", "为什么", "是否", "请问",
        "有", "是", "在", "和", "与", "及", "或", "都", "也", "还", "很", "更", "最",
        "公司", "公告", "一下", "目前", "现在", "最近", "这个", "那个", "有关", "情况",
    }

    def _fallback_search_mysql(self, query, k, source_filter=None, table_only=False, max_rows=300):
        """Milvus 不可用时的 MySQL 关键词降级检索（纯兜底，召回质量低于向量检索）。

        返回与 _hybrid_search 同构的 hits 列表：[{"entity": {"parent_id", "text", ...}}, ...]，
        使 hybrid_search_with_rerank 的后续步骤（父块回取/rerank）完全复用，返回结构一致。

        匹配策略：jieba 分词 → 滤停用词/单字 → 取前 5 个关键词 → content LIKE 匹配
        （OR 命中任一词即可，按命中词数降序）→ 每父块聚合由调用方处理。
        source_filter 走 parent_docs.pdf_type 子查询（child_chunks 无 source 字段）。
        """
        try:
            from mysql_qa.utils.preprocess import preprocess_text  # jieba 懒加载，正常路径零开销
            words = [w for w in preprocess_text(query)
                     if len(w) >= 2 and w not in self._FALLBACK_STOPWORDS]
            if not words:
                logger.warning("降级检索：query 分词后无有效关键词，返回空")
                return []
            words = words[:5]
            params = []
            where_ors = " OR ".join(["content LIKE %s"] * len(words))
            score_expr = " + ".join(["(content LIKE %s)"] * len(words))
            sql = (f"SELECT parent_id, chunk_index, content, is_table "
                   f"FROM child_chunks WHERE ({where_ors})")
            params.extend([f"%{w}%" for w in words])
            if source_filter:
                sql += " AND parent_id IN (SELECT id FROM parent_docs WHERE pdf_type=%s)"
                params.append(source_filter)
            if table_only:
                sql += " AND is_table = 1"
            sql += f" ORDER BY {score_expr} DESC, char_count DESC LIMIT %s"
            params.extend([f"%{w}%" for w in words])
            params.append(max_rows)

            self.mysql.cursor.execute(sql, params)
            rows = self.mysql.cursor.fetchall()
            if not rows and table_only:
                # 与正常路径同语义：表格过滤空结果 → 去过滤再查
                logger.info("降级检索：is_table=1 无召回，去掉表格过滤重试")
                self.last_timings["table_filter_fallback"] = True
                sql = (f"SELECT parent_id, chunk_index, content, is_table "
                       f"FROM child_chunks WHERE ({where_ors})")
                params = [f"%{w}%" for w in words]
                if source_filter:
                    sql += " AND parent_id IN (SELECT id FROM parent_docs WHERE pdf_type=%s)"
                    params.append(source_filter)
                sql += f" ORDER BY {score_expr} DESC, char_count DESC LIMIT %s"
                params.extend([f"%{w}%" for w in words])
                params.append(max_rows)
                self.mysql.cursor.execute(sql, params)
                rows = self.mysql.cursor.fetchall()

            logger.info(f"降级检索完成: 关键词={words}, 命中子块={len(rows)}")
            return [{"entity": {"parent_id": str(r["parent_id"]), "text": r["content"],
                                "source": "", "timestamp": "",
                                "is_table": r["is_table"]}}
                    for r in rows]
        except Exception as e:
            logger.error(f"降级检索自身失败，返回空结果: {e}")
            return []

    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None, table_only=False, rerank_k=None):
        """
        混合检索 + 重排序
        流程：
            1. Milvus 混合检索召回子块（table_only=True 时叠加 is_table==1 过滤，
               空结果自动去掉表格过滤回退重试，避免表格类 query 漏检）
            2. 提取 parent_id 列表（去重）
            3. MySQL 批量查父块内容
            4. 重排序返回
        分段耗时写入 self.last_timings：milvus_search / mysql_fetch / rerank / total（秒）
        rerank_k: 精排后保留数，None 时用 conf.RERANK_K（默认 3）；
                  [11] 自愈扩大 topk 的唯一生效入口（传更大值如 5/10），默认行为不变
        """
        # 每次调用重置分段耗时（调用方在方法返回后读取）
        self.last_timings = {"milvus_search": 0.0, "mysql_fetch": 0.0, "rerank": 0.0, "total": 0.0,
                             "table_filter_fallback": False, "table_hit_count": 0,
                             "fallback_to_mysql": False}
        search_start = time.time()

        # 1. 生成查询向量
        query_embeddings = self.embedding_function([query])
        dense_query_vector = query_embeddings["dense"][0]
        sparse_query_vector = self._to_sparse_dict(query_embeddings["sparse"][[0]])

        # 2. Milvus 混合检索（source 与 is_table 过滤 AND 叠加）；
        #    Milvus 异常（容器不稳/超时/网络断）时降级 MySQL 关键词检索，保证有上下文可答
        filter_expr = self._build_filter_expr(source_filter, table_only)
        t0 = time.time()
        try:
            results = self._hybrid_search(
                dense_query_vector, sparse_query_vector, k, filter_expr,
                ["text", "parent_id", "source", "timestamp", "is_table"]
            )
        except Exception as e:
            logger.error(f"Milvus 检索异常，降级 MySQL 关键词检索: {e}")
            self.last_timings["fallback_to_mysql"] = True
            results = self._fallback_search_mysql(query, k, source_filter, table_only)
        if not results and table_only and not self.last_timings.get("fallback_to_mysql"):
            # 兜底：表格类 query 在 is_table==1 下无召回（如存量库未回填标量字段，
            # 或该 source 下无表格块）→ 去掉表格过滤重试，避免漏检
            logger.info("表格过滤无召回，回退不带 is_table 过滤的混合检索")
            self.last_timings["table_filter_fallback"] = True
            filter_expr = self._build_filter_expr(source_filter, False)
            results = self._hybrid_search(
                dense_query_vector, sparse_query_vector, k, filter_expr,
                ["text", "parent_id", "source", "timestamp", "is_table"]
            )
        self.last_timings["milvus_search"] = time.time() - t0

        if not results:
            self.last_timings["total"] = time.time() - search_start
            return []

        # 3. 提取子块 + parent_id 列表，并收集表格块的 [表格] 摘要
        sub_chunks = []
        parent_ids = set()
        table_summaries_by_parent = {}
        for hit in results:
            entity = hit["entity"]
            pid = entity.get("parent_id")
            text = entity.get("text", "")
            if pid:
                parent_ids.add(int(pid))
                # 表格块（is_table==1，结构化标量字段；存量已回填 20,976 条）的摘要单独聚合，供上下文分流用
                # 2026-10-02 迁移：原按 text 前缀 "[表格]" 约定判定，现改纯 is_table==1（用户拍板）
                if entity.get("is_table") == 1:
                    table_summaries_by_parent.setdefault(int(pid), []).append(text)
            sub_chunks.append(Document(
                page_content=text,
                metadata={
                    "parent_id": pid,
                    "source": entity.get("source"),
                    "timestamp": entity.get("timestamp")
                }
            ))

        # 4. MySQL 批量查父块内容
        t1 = time.time()
        parent_docs = self._get_parent_contents(list(parent_ids))
        self.last_timings["mysql_fetch"] = time.time() - t1

        # 4.1 把表格摘要挂到对应父块的 metadata（供 generate_answer 上下文分流）
        for doc in parent_docs:
            pid = int(doc.metadata.get("parent_id", 0) or 0)
            if pid in table_summaries_by_parent:
                doc.metadata["table_summaries"] = table_summaries_by_parent[pid]

        # 5. 重排序（2026-10-01：开关 + 分数回填 + RERANK_K 截断 + 低置信度保护）
        t2 = time.time()
        rerank_disabled = not conf.RERANK_ENABLED
        if conf.RERANK_ENABLED and len(parent_docs) >= 2:
            pairs = [[query, doc.page_content] for doc in parent_docs]
            scores = self.reranker.predict(pairs)
            # 分数回填：每个父块 metadata 带 rerank_score（评估闭环/阈值标定输入）
            for doc, s in zip(parent_docs, scores):
                doc.metadata["rerank_score"] = float(s)
            # 稳定排序：只按 score 排序；score 相同时保持 MySQL 返回顺序，
            # 避免 sorted 回退比较 Document 对象报错（同分父块是真实存在的）
            ranked = sorted(parent_docs, key=lambda d: d.metadata["rerank_score"], reverse=True)
        else:
            # rerank 关闭（RERANK_ENABLED=false）或候选不足 2 个：不调 CrossEncoder，
            # 保持召回顺序；分数不回填（评估字段保持空）
            ranked = parent_docs
        # 低置信度保护：仅当显式配置 threshold(>0) 且重排已执行、top1 分数低于阈值时，
        # 判定整批候选不可信 → 返回空（留给 [10]在线评估/[11]自愈触发入口），
        # 绝不在内部砍第 N 名（实测第5名与第6名分数几乎无断档，硬砍会误杀）
        if (conf.RERANK_ENABLED and conf.RERANK_THRESHOLD > 0 and ranked
                and ranked[0].metadata.get("rerank_score", 0.0) < conf.RERANK_THRESHOLD):
            logger.warning(f"rerank 低置信度保护触发: top1="
                           f"{ranked[0].metadata.get('rerank_score'):.4f} < "
                           f"threshold={conf.RERANK_THRESHOLD} (query: '{query}')")
            self.last_timings["rerank_low_confidence"] = True
            ranked = []
        else:
            self.last_timings["rerank_low_confidence"] = False
        # 精排后保留数量：rerank_k（默认 conf.RERANK_K=3；None 时用默认值；
        # [11] 自愈传更大值扩大 topk，正常路径不传行为不变）
        ranked = ranked[:rerank_k if rerank_k else conf.RERANK_K]
        self.last_timings["rerank"] = time.time() - t2
        # rerank 阶段统计（评估闭环输入，供阈值标定；rerank 关闭/候选不足时为空）
        score_list = [d.metadata["rerank_score"] for d in parent_docs
                      if "rerank_score" in d.metadata]
        self.last_timings["rerank_score_min"] = min(score_list) if score_list else 0.0
        self.last_timings["rerank_score_max"] = max(score_list) if score_list else 0.0
        self.last_timings["rerank_score_mean"] = (sum(score_list) / len(score_list)) if score_list else 0.0
        self.last_timings["rerank_score_median"] = _median(score_list)
        self.last_timings["rerank_candidate_count"] = len(parent_docs)
        self.last_timings["rerank_disabled"] = rerank_disabled
        # 表格命中统计：最终返回的父块中，携带 [表格] 摘要的父块数（即表格来源命中数）
        self.last_timings["table_hit_count"] = len(
            [d for d in ranked if d.metadata.get("table_summaries")])
        self.last_timings["total"] = time.time() - search_start
        return ranked

    def _get_parent_contents(self, parent_ids: list) -> list:
        """从 MySQL 批量查父块内容"""
        if not parent_ids:
            return []

        # 用 IN 批量查
        placeholders = ','.join(['%s'] * len(parent_ids))
        sql = f"SELECT * FROM parent_docs WHERE id IN ({placeholders})"
        self.mysql.cursor.execute(sql, parent_ids)
        rows = self.mysql.cursor.fetchall()

        # 转成 Document 对象，用 parent_content 字段作为内容
        docs = []
        for row in rows:
            content = row.get("parent_content", "") or row.get("file_name", "")
            docs.append(Document(
                page_content=content,
                metadata={
                    "parent_id": str(row["id"]),
                    "file_name": row.get("file_name"),
                    "source": row.get("pdf_type"),
                    "code": row.get("code"),
                    "name": row.get("name"),
                    # 公告/文档时间（pdf_time），供 source_metas["time"] 展示"公告日期"；
                    # 注意与 Milvus 子块的 timestamp（入库处理时间 processed_at）语义不同
                    "timestamp": row.get("pdf_time") or "",
                }
            ))
        return docs


if __name__ == "__main__":
    vector_store = VectorStore()
    print("VectorStore 初始化成功")




