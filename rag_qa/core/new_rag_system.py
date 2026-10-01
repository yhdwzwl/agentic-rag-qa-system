from .prompts import RAGPrompts
#   导入 time 模块，用于计算时间
import time
#   导入 re 模块，用于解析答案中的来源编号 [N]
import re
from datetime import datetime
from openai import OpenAI  # [10] 在线忠实度判分 LLM（非流式，temperature=0，独立于主 LLM）
from base import logger, Config
from langchain_core.documents import Document
from .query_classifier import QueryClassifier  # 导入查询分类器
from .strategy_selector import StrategySelector  # 导入策略选择器

conf = Config()

# 精确数字型 query 触发词：命中则表格块上下文用原文（保证精确数字可查），未命中用 [表格] 摘要
PRECISE_QUERY_WORDS = ["多少", "几", "分别", "具体", "明细", "逐项",
                       "列出", "到底", "数值", "金额是", "具体数字", "具体金额"]

# —— [6] 表格类 query 触发词：命中则检索时叠加 is_table == 1 过滤（优先召回表格块）——
# 保守原则：只收"明确问表格内容"的高置信词，宁可少判（漏判只是退回普通混合检索，
# 语义召回仍可用），不可误判（误判会把候选限制在表格块上，文本块被排除）
TABLE_QUERY_WORDS = [
    "明细", "清单", "一览", "列表", "名单", "名册",
    "前十大", "十大股东", "股东名单", "持股情况", "持股明细",
    "质押明细", "减持明细", "增持明细", "回购明细",
    "分红方案", "派息方案", "分红派息", "利润分配方案",
    "募集资金使用", "募投项目", "资金使用情况",
    "资产负债表", "利润表", "现金流量表", "财务指标",
    "使用情况表", "表格", "什么表",
]

# —— [5] 检索策略选择：直接检索的规则判定（用户拍板：直接检索不需要策略、不需要改写）——
# 否定信号：命中任一 → 判定为"非直接检索"（走 LLM 选策略 + 改写）
DIRECT_QUERY_NEGATIVE_WORDS = [
    # 抽象/分析类（需要假设答案或深度改写）
    "为什么", "原因", "怎么看", "如何看", "分析", "解读", "风险", "趋势", "展望",
    "预测", "影响", "意味着", "说明什么", "会怎样", "后市", "怎么走", "值得",
    # 比较/选择类（多为多实体多维度，需拆子查询）
    "比较", "对比", "vs", "哪个", "谁更", "孰优", "差异", "区别", "不同",
    # 指代类（依赖上下文，需回溯改写）
    "它", "这个公司", "这家公司", "该公司", "上述", "前者", "后者", "那边",
    # 汇总/时间跨度类（需跨多条公告汇总，直接检索单块易漏）
    "今年以来", "最近几年", "历年", "累计", "总共", "合计", "汇总", "整体情况", "所有年份",
]
# 肯定信号：无公司实体命中时，再要求命中强事实词才算"直接检索"
DIRECT_QUERY_FACT_WORDS = [
    "多少", "几", "哪天", "何时", "什么时间", "日期", "金额", "数量", "比例",
    "价格", "股价", "市值", "持股", "减持", "增持", "融资", "回购", "分红",
    "质押", "解禁", "收购", "中标", "业绩", "营收", "净利润", "毛利率", "负债",
    "现金流", "评级", "公告", "披露", "发布", "派息", "定增", "重组", "股数",
]

# —— [9] 拒答检测短语（起步词表，供 [10]评估/[11]自愈触发）——
# 保守原则：只收"明确拒答/无法从资料作答"的高置信短语，宁可漏判（不触发自愈）
# 不可误判（误触发自愈会多消耗一次 LLM 调用）；后续靠 rag_eval_results 的
# refusal_detected 分布标定增删
REFUSAL_PHRASES = [
    "无法回答", "暂时无法回答", "无法回答这个问题", "无法回答您的问题",
    "根据现有资料，暂时无法回答", "根据现有资料无法回答", "现有资料无法回答",
    "没有找到", "未找到", "找不到", "无法找到", "未能找到", "没有查找到",
    "没有检索到", "未检索到", "检索不到", "没有搜索到", "未搜索到",
    "没有相关信息", "没有相关资料", "没有找到相关资料", "未查到", "查不到",
    "无法提供", "无法给出", "无法确认", "无法判断", "无法从资料中", "无法根据资料",
    "资料中未提及", "公告中未提及", "未提及该", "未提及相关",
    "无法基于现有资料", "无法根据现有资料", "不能回答", "不知道",
]


#   定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    #   初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #   设置大语言模型调用函数
        self.llm = llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        #   初始化查询分类器
        self.query_classifier = QueryClassifier(
            conf.bert_intent_cls
            # model_path='/Users/chan/projects/Itcast_qa_system/rag_qa/core/bert_query_classifier',
        )
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()
        # 初始化Redis向量缓存（cache_enabled 总开关；Redis 不可用自动降级跳过）
        self.vector_cache = None
        if conf.CACHE_ENABLED:
            try:
                from .redis_vector_cache import RedisVectorCache
                self.vector_cache = RedisVectorCache()
                logger.info("Redis向量缓存初始化成功")
            except Exception as e:
                logger.warning(f"Redis向量缓存初始化失败，将跳过缓存: {e}")
                self.vector_cache = None
        else:
            logger.info("Redis向量缓存已禁用 (cache_enabled=false)")
        # 最大8192字符
        self.max_prompt_length = 16384

        # 记录每次查询的中间产物
        self.last_query_result = {
            "query_text": "",
            "generated_answer": "",
            "full_prompt": "",
            "contexts": [],
            "retrieved_doc_ids": [],
            "retrieved_scores": [],
            "source_metas": [],
            "retrieval_strategy": "",
            "strategy_select_time": 0.0,
            "rewritten_query": "",
            "classify_time": 0.0,
            "rewrite_time": 0.0,
            "cache_time": 0.0,
            "retrieval_time": 0.0,
            "milvus_search_time": 0.0,
            "mysql_fetch_time": 0.0,
            "rerank_time": 0.0,
            "rerank_disabled": False,
            "generation_time": 0.0,
            "total_time": 0.0,
            "cache_hit": False,
            "cache_level": "",
            "fallback": "",
            "matched_companies": [],
            # [8] 上下文组装评估字段（空上下文标记 + 分层截断信息，供 [10]评估/[11]自愈观测）
            "context_empty": False,
            "truncated_info": {},
            # [9] 来源标注与拒答检测（LLM 成功路径按真实值填充，供前端溯源渲染与 [11] 自愈触发）
            "citations": [],
            "refusal_detected": False,
            "refusal_reason": "",
            # [10] 在线质量自动评估（LLM-as-Judge 忠实度；fail-open 时 score=None、eval_error 记录原因）
            "online_faithfulness": None,
            "online_eval_time": 0.0,
            "eval_error": "",
            # 触发信号汇总（refusal / faithfulness 低 / rerank_low_confidence，供 [11] 自愈触发）
            "quality_issues": [],
            # [7] rerank 低置信度保护标记（vector_store 触发时整批返回空 → context_empty → 拒答）
            "rerank_low_confidence": False,
            # [11] React 自愈（质量不达标时执行；方案A：首版流式 + 修正段追加 yield）
            "heal_triggered": False,
            "heal_reason": [],
            "heal_rounds": 0,
            "heal_actions": [],
            "heal_time": 0.0,
            "heal_final_quality_issues": [],
            "healed_answer": "",
            # 自愈触发但最终仍有问题时的兜底提示文案（不静默返回坏答案）
            "heal_fallback_text": "",
        }

    #   定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query, source_filter=None, table_only=False, rerank_k=None):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        #   获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()  # 使用 template 后缀区分
        #   调用大语言模型生成假设答案
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(query=query))
            hypo_answer = "".join(hypo_answer)
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            #   使用假设答案进行检索，并返回检索结果
            #   注意：HyDE 通常只用于生成检索向量，不一定需要 rerank 这一步，但这里复用了
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer, k=conf.RETRIEVAL_K, source_filter=source_filter,  # 使用 K 而非 M
                table_only=table_only, rerank_k=rerank_k
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    #   定义私有方法，使用子查询进行检索
    def _retrieve_with_subqueries(self, query, source_filter=None, table_only=False, rerank_k=None):
        """
        1.获取提示词
        2、调用大模型得到子问题
        3、循环处理子问题
        4、合并父文档 返回topk
        :param query:
        :return:
        """
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        #   获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成子查询列表
            subqueries_text = self.llm(subquery_prompt_template.format(query=query))
            subqueries_text = "".join(subqueries_text)
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            #   初始化空列表，用于存储所有子查询的检索结果
            all_docs = []
            #   遍历每个子查询
            for sub_q in subqueries:
                #   使用子查询进行检索，并将结果添加到列表中
                #   这里对每个子查询都执行了 hybrid search + rerank，开销可能较大
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.RETRIEVAL_K, source_filter=source_filter,  # 使用 K
                    table_only=table_only, rerank_k=rerank_k
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            #   对所有检索结果进行去重 (基于对象内存地址，如果 Document 内容相同但对象不同则无法去重)
            #   更可靠的去重方式是基于文档内容或 ID
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}  # 基于内容去重
            unique_docs = list(unique_docs_dict.values())

            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            #   返回去重后的文档，限制数量 (是否需要在此处限制? retrieve_and_merge 末尾会限制)
            # return unique_docs[: Config.CANDIDATE_M]
            return unique_docs  # 返回所有唯一文档，让 retrieve_and_merge 处理数量

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    #   定义私有方法，结合对话历史改写查询（所有进入 RAG 的 query 统一先改写）
    def _rewrite_query(self, query, history_context):
        """
        结合对话历史，把口语化/带指代的 query 改写成更适合公告库检索的完整问题。
        - 无论有无历史都调用 LLM 改写：
          · 有历史：消解指代（"它/这个"→具体公司）、补全实体
          · 无历史（首轮提问）：跳过指代补全，直接精炼、明确化、补金融专业关键词；
            多跳复杂问题同样受益（模板要求展开成可独立检索的表述，不丢子问题）
        - LLM 失败或返回空：回退原 query，不影响主流程
        :param query: 原始用户问题
        :param history_context: 格式化的对话历史（Q:/A: 多行），可为 ""
        :return: (rewritten_query, elapsed_time)
        """
        rewrite_start = time.time()
        rewritten = query.strip()
        if not query.strip():
            logger.warning(f"query为空，跳过改写")
            return rewritten, time.time() - rewrite_start
        try:
            rewrite_template = RAGPrompts.rewrite_prompt()
            # 无历史时给显式占位，模板据此跳过指代补全
            history_placeholder = history_context if history_context else "（无）"
            prompt = rewrite_template.format(query=query, history=history_placeholder)
            # 兼容两类 llm 接口：生成器函数（yield str）与直接返回字符串
            raw = self.llm(prompt)
            if isinstance(raw, str):
                rewritten = raw.strip()
            else:
                parts = []
                for token in raw:
                    if isinstance(token, tuple):
                        # 防御：某些流式封装 yield (role, content)，只取内容部分
                        token = token[1] if len(token) > 1 else ""
                    if token:
                        parts.append(str(token))
                rewritten = "".join(parts).strip()
            if not rewritten:
                logger.warning(f"query改写返回空，回退原 query (查询: '{query}')")
                rewritten = query.strip()
            else:
                logger.info(f"query改写完成: '{query}' → '{rewritten}'")
        except Exception as e:
            logger.error(f"query改写失败，回退原 query: {e}")
            rewritten = query.strip()
        return rewritten, time.time() - rewrite_start

    #   定义私有方法，使用回溯问题进行检索
    def _retrieve_with_backtracking(self, query, source_filter=None, table_only=False, rerank_k=None):
        """
        回溯问题策略：query 已在 generate_answer 入口按需改写（_rewrite_query，仅非直接检索类），
        此处直接用（已改写的）query 检索，不再二次改写（避免改变检索意图、重复消耗 LLM）。
        :param query: 已改写（或原样）的查询
        :param source_filter: 公告类型过滤，透传给 Milvus
        :param table_only: 表格类 query 过滤，透传给 Milvus
        :param rerank_k: 精排后保留数（None 用 conf.RERANK_K），自愈扩 topk 用
        :return:
        """
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        try:
            # 直接用传入的（已改写的）query 执行混合检索
            return self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter,  # 使用 K
                table_only=table_only, rerank_k=rerank_k
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    #   定义方法，检索并合并相关文档
    def retrieve_and_merge(self, query, source_filter=None, strategy=None, table_only=False, rerank_k=None, candidate_m=None):  # 新增 strategy 参数
        """
        检索并合并相关文档。
        :param rerank_k: 精排后保留数（None 用 conf.RERANK_K），[11] 自愈扩大 topk 用
        :param candidate_m: 最终保留数（None 用 conf.CANDIDATE_M）；
                            自愈扩 rerank_k 时必须同步传（否则会被 CANDIDATE_M 截断砍回）
        """
        # 如果未指定检索策略，则使用策略选择器选择
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        #   根据检索策略选择不同的检索方式
        ranked_sub_chunks = []  # 初始化
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query, source_filter=source_filter, table_only=table_only, rerank_k=rerank_k)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query, source_filter=source_filter, table_only=table_only, rerank_k=rerank_k)  # 返回的是唯一文档列表
            # 注意：子查询返回的是已 rerank 过的父文档或子块列表，后续合并逻辑可能需要调整
            # 当前实现中，子查询返回的是初步检索（可能已rerank）的块，再进行合并
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query, source_filter=source_filter, table_only=table_only, rerank_k=rerank_k)
        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter,  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档
                table_only=table_only, rerank_k=rerank_k
            )

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_sub_chunks)} 个候选文档 (可能已是父文档)")
        final_context_docs = ranked_sub_chunks[:candidate_m if candidate_m else conf.CANDIDATE_M]

        return final_context_docs

    def _is_precise_query(self, query: str) -> bool:
        """判断是否为精确数字型 query（命中触发词则表格块上下文用原文而非摘要）"""
        if not query:
            return False
        return any(w in query for w in PRECISE_QUERY_WORDS)

    def _is_table_query(self, query: str) -> bool:
        """规则判定 query 是否"表格类"（问表格内容的问题）。

        命中触发词 → 检索时叠加 is_table == 1 过滤（优先召回表格块，空结果自动回退）。
        保守原则：宁可少判（漏判只是退回普通混合检索），不可误判。
        表格类 ≠ 精确数字型：精确型控制"上下文用原文还是 [表格] 摘要"，表格类控制"召回过滤"。
        """
        if not query:
            return False
        q = query.strip().lower()
        return any(w in q for w in TABLE_QUERY_WORDS)

    def _is_direct_query(self, query: str, matched_companies: list) -> bool:
        """
        规则判定 query 是否属于"直接检索"（意图明确的单实体事实查询）。
        判定为"直接"需同时满足：
        1. 不含否定信号（抽象/比较/指代/汇总跨度）
        2. 不是多实体复合问题（matched_companies >= 2，交给子查询策略）
        3. 有明确公司实体，或命中强事实触发词
        保守原则：宁可少判直接（多走一次改写，改写失败会回退原 query），
        不可误判直接（复杂问题漏改写会丢检索信号）。
        直接检索类：不调 LLM 选策略、不改写，用原 query 直接检索。
        """
        if not query or not query.strip():
            return False
        q_lower = query.strip().lower()
        # 1) 否定信号
        if any(w in q_lower for w in DIRECT_QUERY_NEGATIVE_WORDS):
            return False
        # 2) 多实体复合问题 → 非直接（子查询）
        if matched_companies and len(matched_companies) >= 2:
            return False
        # 3) 有公司实体（强信号）或命中强事实词 → 直接检索
        if matched_companies:
            return True
        return any(w in q_lower for w in DIRECT_QUERY_FACT_WORDS)

    # —— [8] 上下文组装（父块优先）：三个独立方法，可单测、可复用（[11] 自愈可重新组装）——

    def _build_context_blocks(self, context_docs, query, rewritten_query):
        """检索结果（父块 Document 列表）→ 内容块列表 + 结构化来源列表（同序）。

        表格分流：概览型 query 用 [表格] 语义摘要（LLM 易读）；精确数字型（原 query OR
        改写 query 任一命中触发词，防改写精简掉触发词）用父块原文（保证精确数字可查）。
        :return: (blocks, source_metas, is_precise)
        """
        is_precise = self._is_precise_query(query) or self._is_precise_query(rewritten_query)
        blocks = []
        source_metas = []
        for doc in context_docs:
            summaries = doc.metadata.get("table_summaries")
            if summaries and not is_precise:
                blocks.append("\n".join(summaries))
            else:
                blocks.append(doc.page_content)
            source_metas.append({
                "title": doc.metadata.get("file_name") or doc.metadata.get("name", ""),
                "code": doc.metadata.get("code", ""),
                "time": doc.metadata.get("timestamp", ""),
                "source": doc.metadata.get("source", ""),
                "name": doc.metadata.get("name", ""),
            })
        return blocks, source_metas, is_precise

    def _format_context_blocks(self, blocks, source_metas):
        """内容块 + 来源 → 编号引用上下文（用户拍板格式：块前 [1][2]… + 尾部来源列表）。

        编号与来源一一对应（blocks[i] ↔ source_metas[i]），LLM 回答时可引用编号，
        为 [9] 答案逐句标注来源提供可映射依据（数据层铺垫）。
        来源列表放上下文尾部，随块数对齐（截断后只列保留块）。
        """
        if not blocks:
            return ""
        parts = [f"[{i}] {blk}" for i, blk in enumerate(blocks, 1)]
        metas = source_metas or []
        refs = []
        for i, sm in enumerate(metas, 1):
            title = (sm.get("title") or sm.get("name") or "").strip()
            t = (sm.get("time") or "").strip()
            refs.append(f"[{i}] {title}（{t}）" if t else f"[{i}] {title}")
        if refs:
            parts.append("来源列表：\n" + "\n".join(refs))
        return "\n\n".join(parts)

    def _truncate_layered(self, blocks, source_metas, history, question):
        """分层截断（[8] 遗留风险①修复，基础版，用户拍板）。

        优先级：question 完整保留 > history 近轮优先（从最新往前）> context 按块顺序
        （上游已按 rerank 分数降序，按序追加即分数优先）> 来源列表（随保留块滚动预留）。
        编号前缀（5 字符/块）与来源列表开销滚动计入预算，保证最终组装后不超 max_prompt_length。
        :return: (kept_blocks, kept_metas, kept_history, truncated_info)
        """
        base_len = len(self.rag_prompt.format(context="", history="", question=""))
        avail = self.max_prompt_length - base_len - len(question)
        # 1) history 从最新（列表末尾）往最旧保留；轮间分隔符 1 字符（"\n"，与 history_context 构造一致）
        kept_history = []
        for h in reversed(history or []):
            block = f"Q: {h['question']}\nA: {h['answer']}"
            cost = len(block) + (1 if kept_history else 0)
            if cost > avail:
                break  # 预算不足：丢弃更旧轮（最近一轮都放不下则全部丢弃，宁缺毋滥）
            kept_history.append(h)
            avail -= cost
        kept_history = list(reversed(kept_history))  # 恢复 最旧→最新 顺序
        # 2) context 按块追加：编号前缀 5 字符/块 + 块间分隔 2 字符恒定预留
        #    （"\n\n" 同时承接编号块之间、以及末块与尾部来源列表之间；最后一块的 2 由 refs 或模板前缀承接）
        kept_blocks, kept_metas = [], []
        refs_reserve = 0
        metas = source_metas or []
        for idx, blk in enumerate(blocks or []):
            meta = metas[idx] if idx < len(metas) else {}
            ref_line = len(f"[{idx + 1}] {(meta.get('title') or meta.get('name') or '')}（{meta.get('time') or ''}）")
            new_refs_reserve = refs_reserve + ref_line + (len("来源列表：\n") if refs_reserve == 0 else 0)
            cost = len(blk) + 5 + 2
            if cost > avail - new_refs_reserve:
                # 预算不足：首块都放不下且仍有预算时按剩余预算硬截保前部（扣编号5+块间2+refs）；
                # 已有保留块则丢弃后续（保持块完整）
                if not kept_blocks and (avail - new_refs_reserve) > 7:
                    kept_blocks.append(blk[:max(0, avail - new_refs_reserve - 7)])
                    kept_metas.append(meta)
                break
            kept_blocks.append(blk)
            kept_metas.append(meta)
            avail -= cost
            refs_reserve = new_refs_reserve
        truncated_info = {
            "truncated": len(kept_blocks) < len(blocks or []) or len(kept_history) < len(history or []),
            "history_total_rounds": len(history or []),
            "history_kept_rounds": len(kept_history),
            "context_blocks_before": len(blocks or []),
            "context_blocks_after": len(kept_blocks),
        }
        return kept_blocks, kept_metas, kept_history, truncated_info

    # —— [9] 答案生成（LLM流式）收尾：来源标注 + 拒答检测（数据层，可单测、可复用）——

    def _extract_citations(self, answer, source_metas):
        """从生成答案中解析 [N] 来源编号 → 结构化引用列表（[9] 来源标注数据层）。

        编号与上下文块编号一一对应（_format_context_blocks 的 [1][2]…，source_metas 同序）。
        - 仅接受 1..len(source_metas) 范围内的编号，越界忽略（防御模型幻觉编号）
        - 同一编号多次出现只保留首次（按答案中出现顺序）
        - 无编号/无来源返回空列表，绝不抛异常
        :return: [{"index": N, "title": ..., "time": ..., "source": ..., "code": ..., "name": ...}, ...]
        """
        if not answer or not source_metas:
            return []
        seen = set()
        citations = []
        n = len(source_metas)
        for m in re.finditer(r"\[(\d{1,3})\]", answer):
            idx = int(m.group(1))
            if idx < 1 or idx > n or idx in seen:
                continue
            seen.add(idx)
            sm = source_metas[idx - 1]
            citations.append({
                "index": idx,
                "title": sm.get("title") or sm.get("name") or "",
                "time": sm.get("time") or "",
                "source": sm.get("source") or "",
                "code": sm.get("code") or "",
                "name": sm.get("name") or "",
            })
        return citations

    def _detect_refusal(self, answer, context_empty, is_general=False):
        """拒答检测（[9]）：模型明确表示"找不到/无法回答"时置位，作为 [11] 自愈的触发入口。

        信号1：上下文为空（专业咨询无检索块 → 无可依据资料，直接视为拒答场景）
        信号2：答案文本命中拒答短语（REFUSAL_PHRASES 保守词表，起步值待标定）
        - 通用知识分支不依赖检索，空上下文是常态，不判拒答（is_general=True 跳过信号1）
        :return: (refusal_detected, reason)
        """
        if is_general:
            # 通用知识不依赖检索，空上下文是常态，不判拒答
            return False, ""
        if context_empty:
            return True, "context_empty"
        text = (answer or "").strip()
        for phrase in REFUSAL_PHRASES:
            if phrase in text:
                return True, f"phrase:{phrase}"
        return False, ""

    # —— [10] 在线质量自动评估：LLM-as-Judge 忠实度判分（独立可测、fail-open 不阻断主流程）——

    @staticmethod
    def _parse_faithfulness_response(raw):
        """从判分 LLM 输出中解析忠实度 JSON。

        LLM 可能输出：纯 JSON / JSON 前后带文字 / 空 / 乱码。
        - 优先整串解析，失败则提取首个 {...} 子串（括号计数支持嵌套）再解析
        - score 必须是 0~1 的数值，否则视为解析失败（返回 None，由调用方 fail-open）
        :return: (score, unsupported_claims) 或 None
        """
        import json as _json
        if not raw or not raw.strip():
            return None
        text = raw.strip()
        candidates = [text]
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:i + 1])
                        break
        for cand in candidates:
            try:
                obj = _json.loads(cand)
                if not isinstance(obj, dict):
                    continue
                score = obj.get("score")
                # 排除 bool（Python 中 bool 是 int 子类，score=true 会误通过）
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    continue
                score = float(score)
                if score < 0.0 or score > 1.0:
                    continue
                claims = obj.get("unsupported_claims") or []
                if not isinstance(claims, list):
                    claims = []
                return score, [str(c) for c in claims]
            except Exception:
                continue
        return None

    def _build_quality_issues(self, online_faithfulness, refusal_detected=None,
                              refusal_reason=None, rerank_low_confidence=None):
        """汇总在线质量触发信号（[10] 目标2：分数低/拒答/低置信度 → 供 [11] 自愈触发）。

        注意：context_empty 在专业咨询时已被 _detect_refusal 捕获为 refusal:context_empty，
        不在此重复计入。faithfulness=None（评估未跑/fail-open）不触发。
        :param online_faithfulness: 忠实度分数（None 表示未评估/评估失败，不触发）
        :param refusal_detected / refusal_reason / rerank_low_confidence: 显式传入时
               用传入值（[11] 自愈轮按轮内状态评估）；不传时读 last_query_result（首轮流式路径）
        """
        if refusal_detected is None:
            refusal_detected = self.last_query_result.get("refusal_detected", False)
        if refusal_reason is None:
            refusal_reason = self.last_query_result.get("refusal_reason", "")
        if rerank_low_confidence is None:
            rerank_low_confidence = self.last_query_result.get("rerank_low_confidence", False)
        issues = []
        if refusal_detected:
            issues.append(f"refusal:{refusal_reason}")
        if (online_faithfulness is not None
                and online_faithfulness < conf.QUALITY_FAITHFULNESS_THRESHOLD):
            issues.append(f"faithfulness:{online_faithfulness:.2f}")
        if rerank_low_confidence:
            issues.append("rerank_low_confidence")
        return issues

    def _evaluate_faithfulness(self, answer, question, blocks):
        """在线忠实度评估（LLM-as-Judge，一次非流式判分调用，temperature=0）。

        把截断后实际进 prompt 的上下文块编号（[1][2]...，与 _format_context_blocks
        编号规则一致），连同问题与答案发给判分 LLM，解析 JSON 分数。
        - 判分模型用 eval_llm_models 轮换（便宜模型压成本；额度耗尽自动切下一个；
          未配置时回退主模型），与主 LLM 解耦
        - 上下文/答案超长先截断（eval_max_*_chars），防判分 prompt 超限
        - 任何异常/解析失败 → 返回 (None, [], elapsed) fail-open，不阻断主流程
        :return: (score, unsupported_claims, elapsed_seconds)
        """
        eval_start = time.time()
        try:
            if not answer or not answer.strip() or not blocks:
                return None, [], time.time() - eval_start
            # 编号上下文（编号规则与 _format_context_blocks 一致：blocks[i] ↔ [i+1]）
            ctx_parts = [f"[{i}] {blk}" for i, blk in enumerate(blocks, 1)]
            ctx_text = "\n\n".join(ctx_parts)
            if len(ctx_text) > conf.QUALITY_EVAL_MAX_CONTEXT_CHARS:
                ctx_text = ctx_text[:conf.QUALITY_EVAL_MAX_CONTEXT_CHARS]
            ans_text = answer.strip()
            if len(ans_text) > conf.QUALITY_EVAL_MAX_ANSWER_CHARS:
                ans_text = ans_text[:conf.QUALITY_EVAL_MAX_ANSWER_CHARS]
            prompt = RAGPrompts.faithfulness_prompt().format(
                question=question, context=ctx_text, answer=ans_text)
            raw = self._call_eval_llm(prompt)
            text = (raw or "").strip()
            parsed = self._parse_faithfulness_response(text)
            if parsed is None:
                logger.warning(
                    f"忠实度判分解析失败，fail-open (question: '{str(question)[:30]}')")
                return None, [], time.time() - eval_start
            score, claims = parsed
            logger.info(
                f"在线忠实度评估完成: score={score:.3f} claims={len(claims)} "
                f"(question: '{str(question)[:30]}')")
            return score, claims, time.time() - eval_start
        except Exception as e:
            logger.warning(f"在线忠实度评估异常，fail-open: {e}")
            return None, [], time.time() - eval_start

    def _call_eval_llm(self, prompt):
        """调用判分 LLM（非流式，temperature=0 保证分数稳定）。

        模型列表轮换：conf.QUALITY_EVAL_LLM_MODELS 未配置时用主模型 conf.LLM_MODEL；
        单模型失败自动尝试下一个；全部失败抛出 RuntimeError（由调用方 fail-open）。
        :return: 判分模型返回的字符串
        """
        models = conf.QUALITY_EVAL_LLM_MODELS or [conf.LLM_MODEL]
        last_err = None
        for model in models:
            try:
                client = OpenAI(api_key=conf.DASHSCOPE_API_KEY,
                                base_url=conf.DASHSCOPE_BASE_URL)
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一个严格的金融RAG忠实度评估器。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    timeout=30,
                )
                content = completion.choices[0].message.content if completion.choices else ""
                if content and content.strip():
                    return content
                last_err = f"model={model} empty response"
            except Exception as e:
                last_err = f"model={model} failed: {e}"
                logger.warning(f"判分模型调用失败，尝试下一个: {last_err}")
        raise RuntimeError(f"所有判分模型均失败: {last_err}")

    # —— [11] React 自愈 Agent：规则反思 + 换策略/重写/扩 topk → 重新检索生成（运行时闭环）——

    def _collect_llm(self, prompt):
        """收集 LLM 完整输出（兼容 langchain stream / 生成器函数 / 直接返回 str）。

        [11] 自愈轮不流式（修正答案最后一次性追加），复用主 LLM 生成。
        """
        if hasattr(self.llm, "stream"):
            response = self.llm.stream(prompt)
            return "".join(
                chunk.content if isinstance(chunk.content, str) else str(chunk)
                for chunk in response)
        raw = self.llm(prompt)
        if isinstance(raw, str):
            return raw
        parts = []
        for token in raw:
            if isinstance(token, tuple):
                # 防御：某些流式封装 yield (role, content)，只取内容部分
                token = token[1] if len(token) > 1 else ""
            if token:
                parts.append(str(token))
        return "".join(parts)

    def _next_rerank_k(self, current):
        """取 heal_rerank_k_list 中比 current（None=conf.RERANK_K）大的最小档位；无更大档返回 None。"""
        base = current if current else conf.RERANK_K
        for k in sorted(conf.SELF_HEAL_RERANK_K_LIST):
            if k > base:
                return k
        return None

    def _pick_heal_action(self, issues, strategy):
        """规则选自愈动作（可解释、不依赖 LLM 稳定输出；LLM 反思仅记录 heal_log）。

        按触发信号优先级返回一个动作：
        - context_empty（无召回）→ 换策略（直接→假设问题→子查询；已是子查询则强制重写）
        - refusal（有召回但模型拒答）→ 强制重写 query
        - rerank_low_confidence → 扩大 topk（rerank_k 升一档）
        - faithfulness 低 → 换策略（直接→假设问题）+ 扩大 topk
        :return: dict {"name": "strategy"|"rewrite"|"expand_topk"|"none",
                       "strategy_to": str|None, "rerank_k": int|None, "detail": str}
        """
        strategy_rotation = ["直接检索", "假设问题检索", "子查询检索"]
        # 1) 无召回：换策略最可能有效（HyDE 补 query 与文档措辞鸿沟）
        if any(i.startswith("refusal:context_empty") for i in issues):
            if strategy in strategy_rotation and strategy != strategy_rotation[-1]:
                nxt = strategy_rotation[strategy_rotation.index(strategy) + 1]
                return {"name": "strategy", "strategy_to": nxt,
                        "rerank_k": None, "detail": f"{strategy}→{nxt}"}
            return {"name": "rewrite", "strategy_to": None, "rerank_k": None,
                    "detail": "context_empty 且策略已轮换到末尾，强制重写 query"}
        # 2) 拒答（有召回）→ 强制重写
        if any(i.startswith("refusal:") for i in issues):
            return {"name": "rewrite", "strategy_to": None, "rerank_k": None,
                    "detail": "拒答（有召回），强制重写 query 后再检索"}
        # 3) rerank 低置信度 → 扩大 topk
        if "rerank_low_confidence" in issues:
            return {"name": "expand_topk", "strategy_to": None,
                    "rerank_k": self._next_rerank_k(None),
                    "detail": "rerank 低置信度，扩大精排保留数"}
        # 4) 忠实度低 → 换策略 + 扩大 topk
        if any(i.startswith("faithfulness:") for i in issues):
            nxt = "假设问题检索" if strategy == "直接检索" else strategy
            return {"name": "strategy", "strategy_to": nxt,
                    "rerank_k": self._next_rerank_k(None),
                    "detail": f"忠实度低，换策略({nxt}) + 扩大 topk"}
        return {"name": "none", "strategy_to": None, "rerank_k": None, "detail": ""}

    @staticmethod
    def _parse_heal_plan_response(resp):
        """宽容解析自愈规划 JSON（ReAct Thought 输出），校验动作枚举。

        优先整串解析，失败则提取首个 {...} 再解析；action 必须 ∈ {rewrite, expand_topk,
        strategy, none}，strategy_to ∈ {直接检索, 假设问题检索, 子查询检索}，
        rerank_k ∈ {5,10} 否则置 None。非法/失败返回 None，由调用方降级规则查表。
        :return: dict {"action","strategy_to","rerank_k","reason"} 或 None
        """
        import json as _json
        if not resp or not isinstance(resp, str):
            return None
        text = resp.strip()
        if not text:
            return None
        obj = None
        try:
            obj = _json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = _json.loads(text[start:end + 1])
                except Exception:
                    obj = None
        if not isinstance(obj, dict):
            return None
        action = obj.get("action")
        if action not in ("rewrite", "expand_topk", "strategy", "none"):
            return None
        strategy_to = obj.get("strategy_to")
        if strategy_to not in (None, "直接检索", "假设问题检索", "子查询检索"):
            return None
        rerank_k = obj.get("rerank_k")
        if rerank_k is not None and rerank_k not in (5, 10):
            rerank_k = None
        return {
            "action": action,
            "strategy_to": strategy_to,
            "rerank_k": rerank_k,
            "reason": str(obj.get("reason", ""))[:200],
        }

    def _plan_heal_action(self, issues, strategy, used_actions, query, rewritten, rerank_k=None):
        """ReAct Thought：LLM 从 4 个动作中选 1 个（heal_plan_prompt）。

        - LLM 结果非法/异常 → 降级 _pick_heal_action 规则查表（detail 前缀"规则兜底"）
        - 防死循环保障②：与最近一轮动作组合重复时，强制按规则轮换推进；无新动作 → none
        :return: dict {"name","strategy_to","rerank_k","detail"}
        """
        act = None
        try:
            pp = RAGPrompts.heal_plan_prompt().format(
                query=query,
                rewritten_query=rewritten,
                strategy=strategy,
                retrieved_count=len(self.last_query_result.get("retrieved_scores", [])),
                rerank_scores=str(self.last_query_result.get("retrieved_scores", [])[:5]),
                refusal=str(self.last_query_result.get("refusal_detected", False)),
                faithfulness=str(self.last_query_result.get("online_faithfulness")),
                quality_issues=",".join(issues),
                used_actions=str(used_actions[-3:] or []),
            )
            resp = self._call_eval_llm(pp)
            parsed = self._parse_heal_plan_response(resp)
            if parsed:
                reason = parsed["reason"] or ""
                if parsed["action"] == "strategy":
                    if parsed["strategy_to"] and parsed["strategy_to"] != strategy:
                        act = {"name": "strategy", "strategy_to": parsed["strategy_to"],
                               "rerank_k": parsed["rerank_k"], "detail": f"LLM决策: {reason or '换策略'}"}
                elif parsed["action"] == "rewrite":
                    act = {"name": "rewrite", "strategy_to": None, "rerank_k": parsed["rerank_k"],
                           "detail": f"LLM决策: {reason or '重写 query'}"}
                elif parsed["action"] == "expand_topk":
                    act = {"name": "expand_topk", "strategy_to": None,
                           "rerank_k": parsed["rerank_k"] or self._next_rerank_k(rerank_k),
                           "detail": f"LLM决策: {reason or '扩大 topk'}"}
                elif parsed["action"] == "none":
                    act = {"name": "none", "strategy_to": None, "rerank_k": None,
                           "detail": "LLM决策: 无有效改进，提前结束"}
        except Exception as e:
            logger.warning(f"自愈规划失败（降级规则查表）: {e}")
            act = None
        if act is None:
            act = dict(self._pick_heal_action(issues, strategy))
            act["detail"] = "规则兜底: " + act["detail"]
        # 防死循环保障②：与最近一轮动作组合完全相同时，强制推进规则轮换；无新动作则 none
        if used_actions:
            last = used_actions[-1]
            if (act.get("name") == last.get("action")
                    and act.get("strategy_to") == last.get("strategy_to")
                    and act.get("rerank_k") == last.get("rerank_k")):
                rule_act = self._pick_heal_action(issues, strategy)
                if not (rule_act["name"] == last.get("action")
                        and rule_act.get("strategy_to") == last.get("strategy_to")
                        and rule_act.get("rerank_k") == last.get("rerank_k")):
                    act = dict(rule_act)
                    act["detail"] = "防重推进: " + rule_act["detail"]
                else:
                    act = {"name": "none", "strategy_to": None, "rerank_k": None,
                           "detail": "无新动作可选，提前终止自愈"}
        return act

    def _self_heal(self, query, source_filter, history_context, history, memory_context=""):
        """[11] React 自愈：质量不达标时，规则反思 + 换策略/重写/扩 topk → 重新检索生成。

        方案 A 配套：首版已流式输出，本方法产出"修正版答案"，由调用方追加 yield。
        - 跳过 Redis 缓存（避免坏上下文反复命中）
        - 动作由规则决定（_pick_heal_action）；LLM 反思仅记录 heal_log（失败降级规则继续）
        - 每轮：反思 → 动作 → 重新检索（新 strategy/rewritten/rerank_k/candidate_m）
                 → 重新组装 → 重新生成 → 重新评估
        - 单轮异常即停；任何失败返回原答案（不阻断、不追加）
        - [12] 三层记忆：memory_context 与首轮一致拼到自愈轮 prompt 的 history 段最前
        :return: (healed_answer, heal_info)
        """
        heal_start = time.time()
        first_answer = self.last_query_result.get("generated_answer", "")
        current_issues = list(self.last_query_result.get("quality_issues", []) or [])
        heal_info = {
            "triggered": False, "reason": list(current_issues), "rounds": 0,
            "actions": [], "time": 0.0, "final_issues": list(current_issues),
        }
        if not current_issues:
            return first_answer, heal_info

        # 自愈参数初始值 = 首轮实际值（None 表示沿用 conf 默认）
        strategy = self.last_query_result.get("retrieval_strategy", "") or "直接检索"
        rewritten = self.last_query_result.get("rewritten_query", "") or query
        table_only = bool(self.last_query_result.get("table_only", False))
        rerank_k = None
        candidate_m = None
        best_answer = first_answer
        max_rounds = max(1, conf.SELF_HEAL_MAX_ROUNDS)
        rounds = 0
        try:
            # 已用动作历史（防死循环保障②：不允许连续两轮选同一动作组合）
            used_actions = []
            prev_context_key = None  # 上一轮检索父块 id 集合（无效循环检测，保障③）
            while current_issues and rounds < max_rounds:
                rounds += 1
                # 1) 反思（LLM 诊断，失败降级；只记录不决定动作）
                reflection = ""
                try:
                    rp = RAGPrompts.self_reflect_prompt().format(
                        query=query,
                        rewritten_query=rewritten,
                        strategy=strategy,
                        retrieved_count=len(self.last_query_result.get("retrieved_scores", [])),
                        rerank_scores=str(self.last_query_result.get("retrieved_scores", [])[:5]),
                        refusal=str(self.last_query_result.get("refusal_detected", False)),
                        faithfulness=str(self.last_query_result.get("online_faithfulness")),
                        quality_issues=",".join(current_issues),
                    )
                    reflection = (self._call_eval_llm(rp) or "").strip()
                except Exception as e:
                    logger.warning(f"自愈反思失败（规则继续）: {e}")
                    reflection = ""
                # 2) Thought：LLM 选动作（ReAct）；异常/非法降级规则查表；两者都防"连续同动作"
                act = self._plan_heal_action(
                    current_issues, strategy, used_actions, query, rewritten, rerank_k=rerank_k)
                action_detail = act["detail"]  # 动作说明（写入 heal_log）
                # 3) 执行动作（更新自愈参数）
                if act["name"] == "strategy" and act.get("strategy_to"):
                    strategy = act["strategy_to"]
                elif act["name"] == "rewrite":
                    try:
                        rewritten, _ = self._rewrite_query(query, history_context)
                    except Exception as e:
                        logger.warning(f"自愈重写失败，沿用原 query: {e}")
                        rewritten = query
                if act.get("rerank_k"):
                    rerank_k = act["rerank_k"]
                    candidate_m = rerank_k  # 同步放大最终保留数，避免被 CANDIDATE_M 截断砍回
                # 4) 重新检索（跳过缓存）
                try:
                    context_docs = self.retrieve_and_merge(
                        rewritten, source_filter=source_filter, strategy=strategy,
                        table_only=table_only, rerank_k=rerank_k, candidate_m=candidate_m)
                except Exception as e:
                    logger.warning(f"自愈检索失败，停止自愈: {e}")
                    break
                # 4.1) 无效循环检测（保障③）：检索父块集合未变化 → 提前终止
                context_key = tuple(sorted(
                    {d.metadata.get("parent_id") for d in context_docs if d.metadata}))
                if prev_context_key is not None and context_key == prev_context_key:
                    logger.warning(f"自愈无效循环：检索结果未变化，提前终止 (rounds={rounds})")
                    heal_info["actions"].append({
                        "round": rounds,
                        "action": "none",
                        "detail": "无效循环：检索结果未变化，提前终止",
                        "strategy_used": strategy,
                        "rerank_k_used": rerank_k or conf.RERANK_K,
                        "rewritten_used": rewritten,
                        "retrieved_count": len(context_docs),
                        "reflection": (reflection or "")[:300],
                        "faithfulness_after": None,
                        "refusal_after": False,
                    })
                    break
                prev_context_key = context_key
                # 5) 重新组装（复用 [8] 三方法，格式与首版一致）
                try:
                    blocks, metas, _ = self._build_context_blocks(context_docs, query, rewritten)
                    kept_blocks, kept_metas, kept_history, _tr = self._truncate_layered(
                        blocks, metas, history or [], query)
                    context = self._format_context_blocks(kept_blocks, kept_metas)
                    hctx = "\n".join(
                        [f"Q: {h['question']}\nA: {h['answer']}" for h in kept_history])
                    # [12] 三层记忆：自愈轮与首轮一致拼到 history 段最前
                    if memory_context:
                        hctx = f"{memory_context}\n{hctx}"
                    prompt = self.rag_prompt.format(context=context, history=hctx, question=query)
                    if len(prompt) > self.max_prompt_length:
                        prompt = prompt[:self.max_prompt_length]
                except Exception as e:
                    logger.warning(f"自愈组装失败，停止自愈: {e}")
                    break
                # 6) 重新生成（自愈轮不流式，收集完整后一次性追加）
                try:
                    new_answer = self._collect_llm(prompt)
                except Exception as e:
                    logger.warning(f"自愈生成失败，停止自愈: {e}")
                    break
                if new_answer and new_answer.strip():
                    best_answer = new_answer
                # 7) 重新评估（轮内状态；显式传参避免污染 last_query_result 首版值）
                context_empty = not kept_blocks
                ref_det, ref_reason = self._detect_refusal(new_answer, context_empty, is_general=False)
                tim = getattr(self.vector_store, "last_timings", None) or {}
                rerank_low = tim.get("rerank_low_confidence", False)
                score, _claims, _el = self._evaluate_faithfulness(new_answer, query, kept_blocks)
                new_issues = self._build_quality_issues(
                    score, refusal_detected=ref_det, refusal_reason=ref_reason,
                    rerank_low_confidence=rerank_low)
                # 记录本轮动作（heal_log，供后续分析高频问题）
                heal_info["actions"].append({
                    "round": rounds,
                    "action": act["name"],
                    "detail": action_detail,
                    "strategy_used": strategy,
                    "rerank_k_used": rerank_k or conf.RERANK_K,
                    "rewritten_used": rewritten,
                    "retrieved_count": len(context_docs),
                    "reflection": (reflection or "")[:300],
                    "faithfulness_after": score,
                    "refusal_after": ref_det,
                })
                used_actions.append({
                    "action": act["name"],
                    "strategy_to": act.get("strategy_to"),
                    "rerank_k": act.get("rerank_k"),
                })
                current_issues = new_issues
            heal_info["triggered"] = True
            heal_info["rounds"] = rounds
            heal_info["final_issues"] = list(current_issues)
            heal_info["time"] = time.time() - heal_start
            if current_issues:
                logger.warning(
                    f"自愈完成但仍有问题: {current_issues} (rounds={rounds})，返回最后生成的答案")
            else:
                logger.info(f"自愈成功: rounds={rounds}，问题已消除")
            return best_answer, heal_info
        except Exception as e:
            logger.warning(f"自愈异常，返回原答案（不阻断）: {e}")
            heal_info["time"] = time.time() - heal_start
            return first_answer, heal_info

    # #   定义方法，生成答案
    # def generate_answer(self, query, source_filter=None):
    #     #   记录查询开始时间
    #     start_time = time.time()
    #     logger.info(f"开始处理查询: '{query}', 公告类型过滤: {source_filter}")
    #
    #     #   判断查询类型
    #     query_category = self.query_classifier.predict_category(query)
    #     logger.info(f"查询分类结果：{query_category} (查询: '{query}')")
    #
    #     #   如果查询属于“通用知识”类别，则直接使用 LLM 回答
    #     if query_category == "通用知识":
    #         logger.info("查询为通用知识，直接调用 LLM")
    #         prompt_input = self.rag_prompt.format(
    #             context="", question=query
    #         )  #   不使用上下文
    #         try:
    #             answer = self.llm(prompt_input)
    #         except Exception as e:
    #             logger.error(f"直接调用 LLM 失败: {e}")
    #             answer = f"抱歉，处理您的通用知识问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
    #         processing_time = time.time() - start_time
    #         logger.info(
    #             f"通用知识查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
    #         )
    #         return answer
    #
    #     #   否则，进行 RAG 检索并生成答案
    #     logger.info("查询为专业咨询，执行 RAG 流程")
    #     #   选择检索策略
    #     strategy = self.strategy_selector.select_strategy(query)
    #
    #     #   检索相关文档
    #     context_docs = self.retrieve_and_merge(
    #         query, source_filter=source_filter, strategy=strategy
    #     )  #   传递 strategy
    #
    #     #   准备上下文
    #     if context_docs:
    #         context = "\n\n".join([doc.page_content for doc in context_docs]) # 使用换行符分隔文档
    #         logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
    #         # logger.debug(f"上下文内容:\n{context[:500]}...") # Debug 日志可以打印部分上下文
    #     else:
    #         context = ""
    #         logger.info("未检索到相关文档，上下文为空")
    #
    #     #   构造 Prompt，调用大语言模型生成答案
    #     prompt_input = self.rag_prompt.format(
    #         context=context, question=query
    #     )
    #     # logger.debug(f"最终生成的 Prompt:\n{prompt_input}") # Debug 日志
    #
    #     try:
    #         answer = self.llm(prompt_input)
    #     except Exception as e:
    #         logger.error(f"调用 LLM 生成最终答案失败: {e}")
    #         answer = f"抱歉，处理您的专业咨询问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
    #
    #
    #     #   记录查询处理完成的日志
    #     processing_time = time.time() - start_time
    #     logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
    #     return answer

    # 添加完历史消息后，定义方法，生成答案
    # 修改 rag_system.py 中的 generate_answer 方法
    def generate_answer(self, query, source_filter=None, history=None, memory_context=""):
        """生成答案，支持对话历史，限制提示长度，并支持流式输出

        [12] 三层记忆：memory_context 为记忆注入文本（长期偏好/本会话关注），
        默认空串 → 现有调用（streamlit/app 旧调用）行为完全不变；
        非空时拼到最终 prompt 的 history 段最前（在分层截断之后，配合限长与兜底硬截）。
        """
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 公告类型过滤: {source_filter}")

        # 验证历史格式
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            history = history[-conf.WINDOW_ROUNDS:]  # [12] 窗口记忆轮数（config 化，默认3，原硬编码5）
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}，忽略历史")
                    history = []
                    break

        # 构造历史上下文
        history_context = ""
        if history:
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")

        # 判断查询类型，通用知识还是专业知识
        classify_start = time.time()
        category_result = self.query_classifier.predict_category(query)
        if isinstance(category_result, tuple):
            query_category, matched_companies = category_result
        else:
            query_category, matched_companies = category_result, []
        self.last_query_result["classify_time"] = time.time() - classify_start
        self.last_query_result["query_text"] = query
        self.last_query_result["matched_companies"] = matched_companies
        # 每次调用先复位缓存/耗时评估字段（避免多轮对话时残留上次命中状态污染评估记录）
        self.last_query_result["cache_hit"] = False
        self.last_query_result["cache_level"] = ""
        self.last_query_result["cache_time"] = 0.0
        self.last_query_result["milvus_search_time"] = 0.0
        self.last_query_result["mysql_fetch_time"] = 0.0
        self.last_query_result["rerank_time"] = 0.0
        self.last_query_result["rerank_disabled"] = False
        # 表格类判定与回退标记（每次复位；未命中检索时按真实值填充，缓存命中保持默认）
        self.last_query_result["table_only"] = False
        self.last_query_result["table_fallback"] = False
        self.last_query_result["table_hit_count"] = 0
        self.last_query_result["fallback_to_mysql"] = False
        # [8] 上下文组装评估字段（每次复位；检索/缓存路径按真实值填充）
        self.last_query_result["context_empty"] = False
        self.last_query_result["truncated_info"] = {
            "truncated": False, "history_total_rounds": 0, "history_kept_rounds": 0,
            "context_blocks_before": 0, "context_blocks_after": 0
        }
        # [9] 来源标注与拒答检测（每次复位；LLM 成功路径按真实值填充，失败/异常保持默认）
        self.last_query_result["citations"] = []
        self.last_query_result["refusal_detected"] = False
        self.last_query_result["refusal_reason"] = ""
        # [10] 在线质量评估字段（每次复位；通用知识分支不评估保持默认）
        self.last_query_result["online_faithfulness"] = None
        self.last_query_result["online_eval_time"] = 0.0
        self.last_query_result["eval_error"] = ""
        self.last_query_result["quality_issues"] = []
        self.last_query_result["rerank_low_confidence"] = False
        # [11] 自愈字段（每次复位；不触发时保持默认）
        self.last_query_result["heal_triggered"] = False
        self.last_query_result["heal_reason"] = []
        self.last_query_result["heal_rounds"] = 0
        self.last_query_result["heal_actions"] = []
        self.last_query_result["heal_time"] = 0.0
        self.last_query_result["heal_final_quality_issues"] = []
        self.last_query_result["healed_answer"] = ""
        self.last_query_result["heal_fallback_text"] = ""
        logger.info(f"查询分类结果: {query_category} (查询: '{query}', 命中公司: {matched_companies})")

        # 准备上下文
        try:
            retrieval_strategy = ""
            # 第二层缓存相关中间变量（通用知识分支保持默认：不查缓存、不回填）
            cache_hit = False
            q_vec = None
            context_parts = []
            parent_ids = []
            source_metas = []
            rewritten_query = ""
            if query_category in ("通用知识", "通用无关"):
                logger.info("查询为通用知识，直接调用 LLM")
                # 不进入检索链路，无需改写；评估字段置为原 query、耗时 0
                self.last_query_result["rewritten_query"] = query
                self.last_query_result["rewrite_time"] = 0.0
                context_docs = []
            else:
                logger.info("查询为专业咨询，执行 RAG 流程")
                # —— [5] 检索策略选择（前置判定：先判策略，再决定是否改写）——
                # 直接检索类（规则判定，不调 LLM、不改写）：意图明确的单实体事实查询，用原 query 直接检索
                # 非直接类（抽象/比较/指代/多实体/汇总）：先 LLM 选策略（用原 query，保留指代/多跳/抽象等
                #   原始信号，避免改写后信号丢失），再统一改写（改写结果用于缓存与检索）
                select_start = time.time()
                if self._is_direct_query(query, matched_companies):
                    retrieval_strategy = "直接检索"
                    rewritten_query = query.strip()   # 不改写
                    rewrite_elapsed = 0.0
                    logger.info(f"规则判定为直接检索，跳过策略选择与改写 (查询: '{query}')")
                else:
                    # select_strategy 内部完成归一化（无法识别自动回退"直接检索"）
                    retrieval_strategy = self.strategy_selector.select_strategy(query)
                    rewritten_query, rewrite_elapsed = self._rewrite_query(query, history_context)
                self.last_query_result["strategy_select_time"] = time.time() - select_start
                self.last_query_result["rewritten_query"] = rewritten_query
                self.last_query_result["rewrite_time"] = rewrite_elapsed

                # —— [4] 第二层缓存：Redis 向量缓存（query 来源：直接类=原query；非直接类=改写后query）——
                # 命中：跳过 混合检索 + 重排（全链路最耗时环节），复用缓存上下文，
                #       仍走下方共用 LLM 生成（只缓存上下文，不缓存答案）
                # 未命中：走原链路；生成成功后回填上下文到缓存
                cache_start = time.time()
                cache_hit_data = None
                if self.vector_cache is not None:
                    try:
                        q_vec = self.vector_store.get_query_vector(rewritten_query)
                        cache_hit_data = self.vector_cache.search_similar(q_vec, source_filter=source_filter)
                    except Exception as e:
                        logger.warning(f"向量缓存查询异常，跳过缓存: {e}")  # 缓存绝不阻断主流程
                self.last_query_result["cache_time"] = time.time() - cache_start

                if cache_hit_data:
                    # —— 命中：复用缓存上下文 ——
                    cache_hit = True
                    context_parts = cache_hit_data["chunks"]
                    # 缓存按序对齐加来源头（用户拍板）：chunks[i] ↔ source_metas[i]
                    # （save_cache 时两列表同序保存；老缓存缺字段用 .get 防御）
                    source_metas = cache_hit_data.get("source_metas", [])
                    context_docs = []
                    parent_ids = cache_hit_data["parent_ids"]
                    self.last_query_result["retrieved_doc_ids"] = [str(p) for p in parent_ids]
                    retrieval_strategy = "cache"
                    logger.info(f"向量缓存命中 | 级别={'formal' if cache_hit_data['is_formal'] else 'temporary'} | 复用上下文 {len(context_parts)} 块")
                else:
                    # —— 未命中：按已选策略检索 → 上下文组装 ——
                    # 表格类 query 判定：原 query OR 改写后 query 任一命中即叠加 is_table==1 过滤
                    # （与精确型同样的保守原则：改写可能精简掉触发词，双判防漏）
                    table_only = self._is_table_query(query) or self._is_table_query(rewritten_query)
                    self.last_query_result["table_only"] = table_only
                    retrieval_start = time.time()
                    context_docs = self.retrieve_and_merge(
                        rewritten_query, source_filter=source_filter, strategy=retrieval_strategy,
                        table_only=table_only
                    )
                    self.last_query_result["retrieval_time"] = time.time() - retrieval_start
                    # 分段耗时（vector_store.last_timings 由 hybrid_search_with_rerank 填充；
                    # 子查询/多路检索时取最后一次调用的分段值）
                    tim = getattr(self.vector_store, "last_timings", None) or {}
                    self.last_query_result["milvus_search_time"] = tim.get("milvus_search", 0.0)
                    self.last_query_result["mysql_fetch_time"] = tim.get("mysql_fetch", 0.0)
                    self.last_query_result["rerank_time"] = tim.get("rerank", 0.0)
                    # rerank 关闭标记（配置关闭或候选不足未执行 CrossEncoder）
                    self.last_query_result["rerank_disabled"] = tim.get("rerank_disabled", False)
                    # 表格过滤回退标记 + 表格命中数（供 rag_eval_results 标定 TABLE_QUERY_WORDS）
                    self.last_query_result["table_fallback"] = tim.get("table_filter_fallback", False)
                    self.last_query_result["table_hit_count"] = tim.get("table_hit_count", 0)
                    # Milvus 异常时降级 MySQL 检索的标记（可观测 Milvus 故障频率）
                    self.last_query_result["fallback_to_mysql"] = tim.get("fallback_to_mysql", False)
                    # [7] rerank 低置信度保护标记（top1 分数低于 rerank_threshold 时整批返回空，
                    #     由 [10] 评估/[11] 自愈触发入口消费；缓存命中路径不设置保持默认 False）
                    self.last_query_result["rerank_low_confidence"] = tim.get("rerank_low_confidence", False)
                    # —— [8] 上下文组装（父块优先）——
                    # 表格分流：概览型 query 用 [表格] 语义摘要（LLM 易读）；精确数字型（原 query OR
                    # 改写 query 任一命中触发词，防改写精简掉触发词）用父块原文（保证精确数字可查）。
                    # 编号引用 + 来源列表在下方统一组装段生成（与缓存命中路径格式一致）
                    context_parts, source_metas, is_precise = self._build_context_blocks(
                        context_docs, query, rewritten_query)
                    logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块 (精确型={is_precise})")

                    # 记录中间产物：上下文块、来源、doc_ids
                    parent_ids = []
                    for doc in context_docs:
                        pid = doc.metadata.get("parent_id")
                        if pid:
                            try:
                                parent_ids.append(int(pid))
                            except (TypeError, ValueError):
                                parent_ids.append(pid)  # 非数字 id 原样保留
                    self.last_query_result["retrieved_doc_ids"] = [str(p) for p in parent_ids]
                    # rerank 分数（评估闭环：分数分布是阈值标定的输入；缓存命中保持空）
                    self.last_query_result["retrieved_scores"] = [
                        doc.metadata.get("rerank_score") for doc in context_docs
                    ]

                # 统一写缓存评估字段（命中/未命中都写，避免多轮对话残留上次命中状态）
                self.last_query_result["cache_hit"] = cache_hit
                self.last_query_result["cache_level"] = (
                    "formal" if cache_hit_data["is_formal"] else "temporary"
                ) if cache_hit else ""

            self.last_query_result["retrieval_strategy"] = retrieval_strategy

            # —— [8] 上下文组装：分层截断（question 完整 > history 近轮优先 > context 按块 > 来源列表）
            # 所有分支统一走这里（通用知识/缓存命中/检索未命中），编号引用格式一致
            context_parts_final, source_metas_final, history_final, truncated_info = self._truncate_layered(
                context_parts, source_metas, history or [], query)
            context = self._format_context_blocks(context_parts_final, source_metas_final)
            hctx = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history_final])
            # [12] 三层记忆注入：拼到 history 段最前（截断之后拼接，不改 _truncate_layered 预算；
            # 注入文本由 MemoryManager 限长（memory_max_chars）+ 下方兜底硬截双保险，保证不超 prompt 预算）
            history_context = f"{memory_context}\n{hctx}" if memory_context else hctx
            # 评估字段同步为实际进入 prompt 的内容（一致性：截断后只记保留块）
            self.last_query_result["context_empty"] = not context_parts_final
            self.last_query_result["truncated_info"] = truncated_info
            self.last_query_result["contexts"] = context_parts_final
            self.last_query_result["source_metas"] = source_metas_final
            self.last_query_result["retrieved_doc_ids"] = (
                self.last_query_result.get("retrieved_doc_ids") or [])[:len(context_parts_final)]
            self.last_query_result["retrieved_scores"] = (
                self.last_query_result.get("retrieved_scores") or [])[:len(context_parts_final)]

            # 构造提示
            prompt_input = self.rag_prompt.format(
                context=context,
                history=history_context,
                question=query
            )
            self.last_query_result["full_prompt"] = prompt_input

            # 兜底截断：正常路径已由分层截断保证不超长；仅当 question 本身异常超长时触发硬截
            if len(prompt_input) > self.max_prompt_length:
                logger.warning(f"提示长度 {len(prompt_input)} 超过 {self.max_prompt_length}，兜底硬截")
                prompt_input = prompt_input[:self.max_prompt_length]
                logger.info(f"截断后提示长度: {len(prompt_input)}")

            # 调用 LLM（流式输出）
            gen_start = time.time()
            try:
                # 兼容两种 LLM 接口：
                # 1) langchain 类（ChatOpenAI 等）：必须用 stream() 拿真正的流式 chunk，
                #    invoke(stream=True) 只返回单个完整 AIMessage，对它迭代会得到 (字段, 值) 元组
                # 2) call_dashscope 等生成器函数：直接调用返回迭代器，yield 字符串
                answer_parts = []
                if hasattr(self.llm, "stream"):
                    # langchain 流式接口，chunk 为 AIMessageChunk，.content 是正文文本
                    response = self.llm.stream(prompt_input)
                    for chunk in response:
                        token = chunk.content if isinstance(chunk.content, str) else str(chunk)
                        answer_parts.append(token)
                        yield token
                else:
                    # 生成器函数接口（如 new_main.py 的 call_dashscope）
                    response = self.llm(prompt_input)
                    for token in response:
                        if isinstance(token, tuple):
                            # 防御：某些流式封装 yield (role, content)，只取内容部分
                            token = token[1] if len(token) > 1 else ""
                        answer_parts.append(token)
                        yield token

                self.last_query_result["generated_answer"] = "".join(answer_parts)
                self.last_query_result["generation_time"] = time.time() - gen_start
                self.last_query_result["total_time"] = time.time() - start_time

                # —— [9] 来源标注 + 拒答检测（数据层；前端可点击溯源 + [11] 自愈触发入口）——
                # 来源标注：解析答案中的 [N] 编号 → 结构化引用（编号↔标题/时间/来源/代码），
                # 供前端后续渲染可点击溯源；模型未引用编号时为空列表，不报错不阻断
                self.last_query_result["citations"] = self._extract_citations(
                    self.last_query_result["generated_answer"],
                    self.last_query_result.get("source_metas", []))
                # 拒答检测：上下文为空 或 答案命中拒答短语 → 标记，供 [10]评估/[11]自愈触发
                # （通用知识分支不判拒答——不依赖检索，空上下文是常态）
                refusal_detected, refusal_reason = self._detect_refusal(
                    self.last_query_result["generated_answer"],
                    self.last_query_result.get("context_empty", False),
                    is_general=query_category in ("通用知识", "通用无关"))
                self.last_query_result["refusal_detected"] = refusal_detected
                self.last_query_result["refusal_reason"] = refusal_reason
                if refusal_detected:
                    logger.warning(
                        f"拒答检测触发: reason={refusal_reason} (query: '{query[:30]}', "
                        f"context_empty={self.last_query_result.get('context_empty', False)})")

                # —— 缓存回填（仅未命中且生成成功；只缓存上下文，不缓存答案；失败不影响主流程）——
                # 存截断后的块（context_parts_final），避免缓存超长块下次命中仍要重复截断
                if not cache_hit and q_vec is not None and self.vector_cache is not None and context_parts_final:
                    try:
                        self.vector_cache.save_cache(
                            query_text=query,
                            rewritten_query=rewritten_query,
                            query_vector=q_vec,
                            chunks=context_parts_final,
                            parent_ids=parent_ids,
                            source_metas=self.last_query_result.get("source_metas", []),
                            source_filter=source_filter,
                        )
                        logger.info(f"缓存回填完成 (query: '{query[:30]}')")
                    except Exception as e:
                        logger.warning(f"缓存回填失败（不影响主流程）: {e}")

                logger.info(
                    f"LLM查询处理完成 (生成耗时: {self.last_query_result['generation_time']:.2f}s, "
                    f"总耗时: {self.last_query_result['total_time']:.2f}s, 查询: {query}"
                )

                # —— [10] 在线质量自动评估（LLM-as-Judge 忠实度；fail-open 不阻断主流程）——
                # 通用知识分支不评估（不依赖检索）；专业咨询 + 有答案才判分
                eval_start = time.time()
                eval_error = ""
                online_faithfulness = None
                if conf.QUALITY_EVAL_ENABLED and query_category not in ("通用知识", "通用无关"):
                    try:
                        online_faithfulness, _unsupported_claims, _ = self._evaluate_faithfulness(
                            self.last_query_result["generated_answer"],
                            query,
                            self.last_query_result.get("contexts", []),
                        )
                        if online_faithfulness is None:
                            eval_error = "eval_parse_failed_or_empty"
                    except Exception as e:
                        eval_error = f"{type(e).__name__}: {str(e)[:200]}"
                        online_faithfulness = None
                self.last_query_result["online_faithfulness"] = online_faithfulness
                self.last_query_result["online_eval_time"] = time.time() - eval_start
                self.last_query_result["eval_error"] = eval_error

                # 触发信号汇总（供 [11] 自愈触发；Phase 1 先落库观测，不触发动作）
                quality_issues = self._build_quality_issues(online_faithfulness)
                self.last_query_result["quality_issues"] = quality_issues
                if quality_issues:
                    logger.warning(
                        f"在线质量触发信号: {quality_issues} (query: '{query[:30]}', "
                        f"faithfulness={online_faithfulness})")

                # —— [11] React 自愈（质量不达标时：规则反思 + 换策略/重写/扩 topk → 重检索生成）——
                # 方案 A：首版已流式输出（无法撤回），自愈修正答案作为追加段 yield；
                # 任何异常 fail-open，不改变已输出内容、不阻断主流程
                healed_answer = ""
                fallback_text = ""
                heal_info = {"triggered": False, "reason": [], "rounds": 0,
                             "actions": [], "time": 0.0, "final_issues": []}
                if (conf.SELF_HEAL_ENABLED and quality_issues
                        and query_category not in ("通用知识", "通用无关")
                        and self.last_query_result.get("generated_answer")):
                    try:
                        healed_answer, heal_info = self._self_heal(
                            query, source_filter, history_context, history or [], memory_context=memory_context)
                        # 仅当自愈确实产出新答案才追加（模型原样复读时不重复输出）
                        if healed_answer and healed_answer != self.last_query_result["generated_answer"]:
                            yield "\n\n【已重新检索，修正回答】\n" + healed_answer
                        # 自愈触发过但最终仍有问题 → 兜底提示（文档原则：不静默返回坏答案）
                        if heal_info["final_issues"]:
                            if any(i.startswith("refusal") for i in heal_info["final_issues"]):
                                fallback_text = ("\n\n【未能检索到足够依据】当前资料库暂未找到可支撑该问题的内容，"
                                                 "建议补充更多背景信息或换一种问法。")
                            else:
                                fallback_text = ("\n\n【提示】本次回答可能未完全覆盖您的问题，内容仅供参考，"
                                                 "建议核对原始公告/年报原文。")
                            yield fallback_text
                    except Exception as e:
                        logger.warning(f"自愈执行异常（不影响主流程）: {e}")
                        healed_answer = ""
                        # 异常也保留触发信号，便于兜底提示（不静默）
                        heal_info = {"triggered": False, "reason": list(quality_issues), "rounds": 0,
                                     "actions": [], "time": 0.0, "final_issues": list(quality_issues)}
                self.last_query_result["heal_triggered"] = heal_info["triggered"]
                self.last_query_result["heal_reason"] = heal_info["reason"]
                self.last_query_result["heal_rounds"] = heal_info["rounds"]
                self.last_query_result["heal_actions"] = heal_info["actions"]
                self.last_query_result["heal_time"] = heal_info["time"]
                self.last_query_result["heal_final_quality_issues"] = heal_info["final_issues"]
                self.last_query_result["healed_answer"] = healed_answer
                self.last_query_result["heal_fallback_text"] = fallback_text

                # 保存本次查询的中间产物到 MongoDB（失败不影响主流程）
                self._save_result_to_mongo()
            except Exception as e:
                logger.error(f"调用LLM失败: {e}")
                yield f"抱歉，处理您的问题时出错，请重试"

        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            yield f"抱歉，处理您的问题时出错，请重试"

    def _save_result_to_mongo(self):
        """将本次查询的中间产物写入 MongoDB rag_eval_results（失败不影响主流程）"""
        try:
            from pymongo import MongoClient
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
            col = client["rag_doc_data"]["rag_eval_results"]
            doc = {
                "source": "main_flow",
                "query_text": self.last_query_result.get("query_text", ""),
                "generated_answer": self.last_query_result.get("generated_answer", ""),
                "full_prompt": self.last_query_result.get("full_prompt", ""),
                "contexts": self.last_query_result.get("contexts", []),
                "retrieved_doc_ids": self.last_query_result.get("retrieved_doc_ids", []),
                "source_metas": self.last_query_result.get("source_metas", []),
                "retrieval_strategy": self.last_query_result.get("retrieval_strategy", ""),
                "strategy_select_time": self.last_query_result.get("strategy_select_time", 0.0),
                "rewritten_query": self.last_query_result.get("rewritten_query", ""),
                "rewrite_time": self.last_query_result.get("rewrite_time", 0.0),
                "classify_time": self.last_query_result.get("classify_time", 0.0),
                "retrieval_time": self.last_query_result.get("retrieval_time", 0.0),
                "milvus_search_time": self.last_query_result.get("milvus_search_time", 0.0),
                "mysql_fetch_time": self.last_query_result.get("mysql_fetch_time", 0.0),
                "rerank_time": self.last_query_result.get("rerank_time", 0.0),
                "rerank_disabled": self.last_query_result.get("rerank_disabled", False),
                "retrieved_scores": self.last_query_result.get("retrieved_scores", []),
                "table_only": self.last_query_result.get("table_only", False),
                "table_fallback": self.last_query_result.get("table_fallback", False),
                "table_hit_count": self.last_query_result.get("table_hit_count", 0),
                "fallback_to_mysql": self.last_query_result.get("fallback_to_mysql", False),
                "generation_time": self.last_query_result.get("generation_time", 0.0),
                "total_time": self.last_query_result.get("total_time", 0.0),
                "cache_hit": self.last_query_result.get("cache_hit", False),
                "cache_level": self.last_query_result.get("cache_level", ""),
                "cache_time": self.last_query_result.get("cache_time", 0.0),
                "matched_companies": self.last_query_result.get("matched_companies", []),
                # [8] 上下文组装评估字段（空上下文标记 + 分层截断信息，供 [10]评估/[11]自愈观测）
                "context_empty": self.last_query_result.get("context_empty", False),
                "truncated_info": self.last_query_result.get("truncated_info", {}),
                # [9] 来源标注与拒答检测（结构化引用列表 + 拒答标记，供前端溯源与 [11] 自愈触发）
                "citations": self.last_query_result.get("citations", []),
                "refusal_detected": self.last_query_result.get("refusal_detected", False),
                "refusal_reason": self.last_query_result.get("refusal_reason", ""),
                # [10] 在线质量自动评估（LLM-as-Judge 忠实度；字段名与离线 RAGAS 的
                #      faithfulness 区分，避免两个来源互相覆盖；fail-open 时 score=None + eval_error）
                "online_faithfulness": self.last_query_result.get("online_faithfulness"),
                "online_eval_time": self.last_query_result.get("online_eval_time", 0.0),
                "eval_error": self.last_query_result.get("eval_error", ""),
                "quality_issues": self.last_query_result.get("quality_issues", []),
                "rerank_low_confidence": self.last_query_result.get("rerank_low_confidence", False),
                # [11] React 自愈（触发信号/动作日志/耗时/修正答案，供分析高频问题）
                "heal_triggered": self.last_query_result.get("heal_triggered", False),
                "heal_reason": self.last_query_result.get("heal_reason", []),
                "heal_rounds": self.last_query_result.get("heal_rounds", 0),
                "heal_actions": self.last_query_result.get("heal_actions", []),
                "heal_time": self.last_query_result.get("heal_time", 0.0),
                "heal_final_quality_issues": self.last_query_result.get("heal_final_quality_issues", []),
                "healed_answer": self.last_query_result.get("healed_answer", ""),
                "heal_fallback_text": self.last_query_result.get("heal_fallback_text", ""),
                "eval_time": datetime.now(),
            }
            col.insert_one(doc)
            client.close()
            logger.info(
                f"中间产物已写入 MongoDB rag_eval_results (query: {self.last_query_result.get('query_text', '')[:30]})"
            )
        except Exception as e:
            logger.warning(f"写入 MongoDB 中间产物失败（不影响主流程）: {e}")
# query 意图识别 使用bert 分类
# 改进：
# 1. 用LLM做意图识别，考虑最近几轮历史问答
# 2. 最近几轮的query 拼接([CLS]current_query[SEP] history_query[SEP] ) 使用多轮对话数据训练Bert








