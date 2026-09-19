# 金融公告RAG问答系统

基于东方财富股票公告数据构建的金融领域RAG问答系统，采用「BERT意图路由 + 智能文档解析 + 双存储架构 + 混合检索重排」全链路设计。

## 核心特性

- **智能文档解析**：6分制复杂度打分器，自动选择MinerU（版面分析）或markitdown（轻量解析）
- **双存储架构**：MySQL存父块元数据+文件去重，Milvus存子块稠密/稀疏向量
- **BERT意图路由**：二分类模型自动判断query是财经相关还是通用问题，通用问题不进检索
- **4种检索策略**：简单问题/直接问题/多跳问题/假设问题，LLM自动选择最优改写策略
- **混合检索**：BGE-M3稠密+稀疏双路向量，Milvus粗排 → 父文档去重 → Reranker精排

## 技术栈

| 模块 | 选型 |
|------|------|
| 向量数据库 | Milvus 2.4 |
| Embedding | BGE-M3（稠密+稀疏） |
| 重排模型 | BGE-Reranker |
| 意图分类 | BERT二分类 |
| 结构化存储 | MySQL 8.0 |
| 文档解析 | MinerU + markitdown |
| 爬虫 | 东方财富公告并发下载 |

## 项目结构

```
rag_qa/core/
├── document_processor.py   # 智能文档解析（打分器+双路径+分块）
├── document_loader.py      # 文档加载入口
├── vector_store.py         # 双存储向量库
├── prompts.py              # Prompt模板库
├── query_classifier.py     # BERT意图分类
├── train_bert.py           # BERT训练脚本
├── generate_test_queries.py # 测试query自动生成
└── new_df_crawl_pdf.py     # 东方财富PDF爬虫
```

## 数据流程

```
PDF下载 → 复杂度打分 → 解析分块 → 父块入MySQL + 子块向量入Milvus
                                                                    ↓
用户Query → BERT路由 → Query改写 → 混合检索 → Reranker精排 → LLM生成答案
```
