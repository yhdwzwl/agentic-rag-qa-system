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
                })

            # 更新文档状态
            self.mysql.update_doc_status(doc_id, "indexed")

        # 3. 批量写 Milvus
        if milvus_data:
            self.client.upsert(collection_name=self.collection_name, data=milvus_data)
            logger.info(f"Milvus 已插入 {len(milvus_data)} 条子块")

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

    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        """
        混合检索 + 重排序
        流程：
            1. Milvus 混合检索召回子块
            2. 提取 parent_id 列表（去重）
            3. MySQL 批量查父块内容
            4. 重排序返回
        """
        # 1. 生成查询向量
        query_embeddings = self.embedding_function([query])
        dense_query_vector = query_embeddings["dense"][0]
        sparse_query_vector = self._to_sparse_dict(query_embeddings["sparse"][[0]])

        # 2. Milvus 混合检索
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
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

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=WeightedRanker(1.0, 0.7),
            limit=k,
            output_fields=["text", "parent_id", "source", "timestamp"]
        )[0]

        if not results:
            return []

        # 3. 提取子块 + parent_id 列表
        sub_chunks = []
        parent_ids = set()
        for hit in results:
            entity = hit["entity"]
            pid = entity.get("parent_id")
            if pid:
                parent_ids.add(int(pid))
            sub_chunks.append(Document(
                page_content=entity.get("text", ""),
                metadata={
                    "parent_id": pid,
                    "source": entity.get("source"),
                    "timestamp": entity.get("timestamp")
                }
            ))

        # 4. MySQL 批量查父块内容
        parent_docs = self._get_parent_contents(list(parent_ids))

        # 5. 重排序
        if len(parent_docs) < 2:
            return parent_docs[:conf.CANDIDATE_M]

        pairs = [[query, doc.page_content] for doc in parent_docs]
        scores = self.reranker.predict(pairs)
        ranked = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        return ranked[:conf.CANDIDATE_M]

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
                }
            ))
        return docs


if __name__ == "__main__":
    vector_store = VectorStore()
    print("VectorStore 初始化成功")




