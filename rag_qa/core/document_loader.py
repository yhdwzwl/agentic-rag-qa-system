# core/document_loader.py
# 薄包装：调用 document_processor.py 的新逻辑
import os
from datetime import datetime
from typing import List

from langchain_core.documents import Document

from base import logger
from rag_qa.core.document_processor import (
    DocumentComplexityScorer,
    DocumentParser,
    StructureAwareSplitter,
    warmup_mineru,
)


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """从目录加载文件（保留旧接口，内部走新流程）"""
    documents = []
    source = os.path.basename(directory_path).replace("_data", "")

    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in (".pdf", ".txt", ".md"):
                logger.warning(f"不支持的文件类型: {file_path}")
                continue
            try:
                # 新流程：打分 → 解析 → 分块
                scorer = DocumentComplexityScorer(file_path)
                score, features = scorer.score()
                parser = DocumentParser(file_path, score, features)
                md_text = parser.parse()

                splitter = StructureAwareSplitter()
                metadata = {
                    "source": source,
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "parse_method": parser.actual_method,  # 用实际的解析方式
                    "complexity_score": score,
                    "timestamp": datetime.now().isoformat()
                }
                chunks = splitter.split(md_text, metadata)
                documents.extend(chunks)
                logger.info(f"成功处理: {file_path} → {len(chunks)} 块")
            except Exception as e:
                logger.error(f"处理文件 {file_path} 失败: {e}")

    return documents


def process_documents(directory_path: str, parent_chunk_size=None,
                      child_chunk_size=None, chunk_overlap=None, **kwargs) -> List[Document]:
    """主入口：处理目录下所有文档，返回子块列表
    
    注：parent_chunk_size / child_chunk_size / chunk_overlap 参数已废弃，
    新的分块策略是结构驱动（按 Markdown 标题层级切分），不再使用固定字符数。
    保留这些参数仅为兼容旧调用，实际被忽略。
    """
    # 自动预热 MinerU 模型，调用者不用管
    warmup_mineru()

    logger.info(f"处理目录: {directory_path}")
    chunks = load_documents_from_directory(directory_path)
    logger.info(f"总子块数量: {len(chunks)}")
    return chunks


if __name__ == "__main__":
    chunks = process_documents(
        r'F:\small_project\aiagent\ailearn\rag\Itcast_qa_system\rag_qa\data\ai_data'
    )
    for i, chunk in enumerate(chunks):
        print(chunk)
        print("=" * 50)
        if i == 5:
            break
