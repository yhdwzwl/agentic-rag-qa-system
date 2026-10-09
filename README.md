# 金融公告 RAG 问答系统

基于东方财富上市公司公告做的一个金融领域 RAG 问答系统。数据侧从公告 PDF 开始，走完解析、分块、入库；问答侧在常规的检索生成之外，加了在线质量评估和自愈，尽量让"答错"这件事变得可发现、可修复。

项目还在持续迭代中，检索参数、评估阈值这些起步值都靠线上落库的数据慢慢标定。

## 系统是怎么工作的

**数据侧**：

```
PDF 下载 → 复杂度打分（6 分制，决定用 MinerU 还是 markitdown）
         → 解析分块 → 父块存 MySQL（元数据 + 去重），子块向量存 Milvus（BGE-M3 稠密 + 稀疏双路）
```

**问答侧**：

1. **意图分类**：BERT 二分类判断是财经专业问题还是通用问题，通用问题不进检索链路。
2. **检索策略选择**：直接检索 / 假设问题检索 / 子查询检索，能规则判定的直接走，不用调 LLM 省一次调用；拿不准的再让 LLM 选。
3. **二级缓存**：Redis 向量缓存，命中的直接复用上下文，不重复走检索（只缓存上下文，不缓存答案）。
4. **混合检索**：Milvus 粗排 → 父文档去重 → BGE-Reranker 精排。表格类问题单独处理，有低置信度保护：精排分数整体过低时整批返回空，留给评估环节去触发自愈。
5. **上下文组装**：父块优先，表格类 query 分"概览型 / 精确型"两条路，超出预算时分层截断（问题完整 > 历史近轮优先 > 上下文按块），答案带 [N] 编号可溯源。
6. **生成**：流式输出，多轮会话带三层记忆（窗口记忆 / 工作记忆 / 长期记忆）。

## 质量评估与自愈

传统 RAG 答完就结束，好坏没人管。这里补了两块：

- **在线质量评估**：每次回答后用独立的判分模型（LLM-as-Judge）对忠实度打分，检查答案里的结论在上下文里有没有依据——数字、金额、日期对不对。分数低于阈值（目前 0.6）、检测到拒答、或者 rerank 置信度过低，都会记入 quality_issues。判分模型和主模型解耦，用便宜的模型轮换压成本；判分失败就跳过，不影响主流程。

- **ReAct 自愈 Agent**：有 quality_issues 就进入自愈循环。LLM 当 Thought 分析现状，从四个动作里选一个——换检索策略 / 重写 query / 扩大精排 / 放弃，执行后重新检索、重新生成、重新评估，最多 3 轮。首版是流式输出，已经发给用户没法撤回，所以修好的答案以追加段的形式输出（"已重新检索，修正回答"）。防死循环做了三重保障：轮次上限、禁止连续两轮用同一个动作、检索结果没变化就提前终止。修不好也不静默，会给出明确的兜底提示。

评估分数、每轮动作、耗时、完整 prompt 都会落 MongoDB，方便复盘和标定阈值。

## 快速开始

```bash
pip install -r requirements.txt
```

需要先起好 Milvus、MySQL、Redis、MongoDB，`docker-compose.yml` 里有编排参考。然后：

- 爬公告：`rag_qa/core/new_df_crawl_pdf.py`
- 入库：`ingest_all.py` 或 `batch_ingest.py`
- 启动服务：`uvicorn app:app`（FastAPI，流式响应），或 `python new_main.py`（命令行）；`streamlit_app.py` 是个简单的前端

## 配置

`config.ini` 按段配置：各数据库连接、LLM 模型（主生成 qwen3.8-27b，判分用 qwen3.8-flash 这类便宜模型）、检索参数（rerank_k、candidate_m）、在线评估（开关、阈值、截断长度）、自愈（开关、最大轮数、扩 topk 档位）、三层记忆。`config.ini.example` 有样例；`config.ini` 本身含密钥，已 gitignore，不提交。

## 目录结构（核心）

```
rag_qa/core/
├── session_qa.py          # 会话编排主入口（FastAPI / CLI 都走这里）
├── new_rag_system.py      # 核心问答链路 + 在线评估 + ReAct 自愈
├── prompts.py             # Prompt 模板（生成 / 判分 / 反思 / 自愈决策）
├── vector_store.py        # Milvus + MySQL 双存储检索与重排
├── redis_vector_cache.py  # 二级向量缓存
├── strategy_selector.py   # 检索策略选择
├── query_classifier.py    # BERT 意图分类
├── document_processor.py  # 文档解析（打分器 + 双路径 + 分块）
├── memory_manager.py      # 三层记忆
├── table_enhancer.py      # 表格语义化增强
├── quality_gate.py        # 质量门（评估信号汇总）
├── rag_eval_model.py      # 在线忠实度评估
├── ragas_evaluate.py      # 离线 RAGAS 评估
└── df_crawl_pdf.py        # 东方财富公告爬虫
```

## 说明

- 数据来自东方财富公开公告，项目用于学习 / 演示，回答不构成投资建议。
- 评估与自愈默认开启，生产使用时建议先跑一段落库数据，按实际分布标定阈值再决定开关。
