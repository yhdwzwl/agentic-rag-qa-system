#   导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain.prompts import PromptTemplate


#   定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    #   定义 RAG 提示模板
    # # 优化后再添加：如果添加了历史消息
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""
        你是一个智能助手，负责帮助用户回答问题。请按照以下步骤处理：

        1. **分析问题和上下文**：
           - 基于提供的上下文（如果有）和你的知识回答问题。
           - 当答案引用了上下文中的内容（尤其是数字、结论）时，必须在对应句子末尾标注来源编号，格式如：2025年营业收入为100亿元[1]。多个来源并列写作 [1][2]。
           - 如果答案来源于检索到的文档，必须在回答末尾附上来源，格式：
             "---
             信息来源：请根据你实际引用的上下文编号（如 [1]、[2]）逐一列出所有来源"

        2. **评估对话历史**：
           - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）。
           - 如果对话历史与问题相关，请结合历史信息生成更准确的回答。
           - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答。

        3. **生成回答**：
           - 提供清晰、准确的回答，避免无关信息。
           - 如果上下文和历史消息均不足以回答问题，请明确说明"根据现有资料，暂时无法回答这个问题"。
        **问题**: {question}
        **上下文**: {context}
        **对话历史**:
        {history}
        
        **回答**:
        """,
            input_variables=["context", "history", "question"],
        )

    #   定义假设问题生成的 Prompt 模板
    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: {query}  
            假设答案：  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            查询: {query}  
            子查询：  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
            你是一个金融搜索专家。请根据对话历史，把用户当前的问题改写成更适合在股票公告库中检索的形式。

            对话历史：
            {history}

            用户当前问题：{query}

            要求：
            1. 结合历史，把"它"、"这个"、"该公司"这种指代替换成具体的公司名
            2. 保留原意，不要改变核心意图
            3. 增加金融领域的专业关键词
            4. 去掉口语化的废话，让query更精炼

            改写后的搜索query：
            """,
            #   定义输入变量
            input_variables=["query", "history"],
        )

    #   定义金融场景query改写 Prompt 模板
    @staticmethod
    def rewrite_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
            你是一个金融搜索专家。请根据对话历史，把用户的问题改写成更适合在股票公告库中检索的形式。

            对话历史：
            {history}
            （若上方对话历史为空或标注"（无）"，说明这是会话的第一轮提问，无需做指代补全，直接精炼当前问题即可）

            用户问题：{query}

            要求：
            1. 若存在对话历史：结合历史，把"它"、"这个公司"具体指什么补全
            2. 若没有对话历史：跳过指代补全，直接对当前问题做改写
            3. 保留原意，不要改变问题的核心意图
            4. 增加金融领域的专业关键词（如：年报、公告、融资、减持、股价、营收、毛利率、公告类型等），提升检索匹配度
            5. 去掉口语化的废话，让query更精炼
            6. 如果问题太模糊，请明确化
            7. 如果问题涉及多个实体或多个方面（多跳问题），展开成完整、可独立检索的表述，不要丢失任何一个子问题

            改写后的搜索query：
            """,
            #   定义输入变量
            input_variables=["query", "history"],
        )

    #   定义在线忠实度评估（LLM-as-Judge）的 Prompt 模板（[10] 在线质量自动评估）
    #   输入：问题 + 编号上下文 + 答案；输出：JSON {score, unsupported_claims}
    #   注意：模板中的字面大括号必须双写（{{ }}）转义，避免被 PromptTemplate 当作变量
    @staticmethod
    def faithfulness_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
            你是一个金融 RAG 系统的忠实度评估器。请判断"回答"中的每个事实性结论是否都能从"上下文"中找到依据。

            规则：
            1. 忠实 = 回答中的每个事实性结论（数字、金额、日期、比例、事件、公司行为等）都能在上下文中直接找到，或由上下文明确推理得出
            2. 不忠实 = 回答中存在上下文里没有依据的结论（编造、张冠李戴、数字错误、时间错误、公司行为错误等）
            3. 上下文按 [1][2][3]... 编号；回答中可能引用编号，以引用为准，但未引用编号的结论也要检查
            4. 只依据给定上下文判断，不要使用你自己的知识补充
            5. 若回答明确表示"无法回答/未找到资料"，score 应为 1.0（拒答本身是诚实的，不算不忠实）

            只输出 JSON（不要输出任何其他文字）：
            {{
              "score": 0.0 ~ 1.0,
              "unsupported_claims": ["无依据的结论原文1", "无依据的结论原文2"]
            }}
            score 表示有依据的结论占全部结论的比例；没有无依据结论时 unsupported_claims 为空数组。

            问题：{question}
            上下文：
            {context}
            回答：
            {answer}
            """,
            #   定义输入变量
            input_variables=["question", "context", "answer"],
        )

    #   定义自愈反思（[11] React 自愈 Agent）的 Prompt 模板
    #   输入：首轮失败的链路信息；输出：JSON {diagnosis, suggested_action}
    #   注意：动作执行以代码内规则为准（可解释、不依赖 LLM 稳定输出），反思仅记录 heal_log
    @staticmethod
    def self_reflect_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
            你是一个金融 RAG 系统的自省诊断器。系统第一轮回答质量不达标，请分析最可能的原因并给出修复建议。

            输入：
            问题：{query}
            改写后查询：{rewritten_query}
            检索策略：{strategy}
            召回父块数：{retrieved_count}
            rerank 分数分布（前若干）：{rerank_scores}
            拒答检测：{refusal}
            在线忠实度分数：{faithfulness}
            触发信号：{quality_issues}

            常见失败原因：
            - query 改写不当（改写后丢失关键实体/数字/时间）
            - 检索策略不当（直接检索召不回抽象/比较/多实体内容）
            - 召回不足（topk 太小，关键文档未进入候选）
            - 上下文组装问题（关键块被截断丢失）

            只输出 JSON（不要输出其他文字）：
            {{
              "diagnosis": "一句话诊断",
              "suggested_action": "rewrite | change_strategy | expand_topk | none"
            }}
            """,
            #   定义输入变量
            input_variables=["query", "rewritten_query", "strategy", "retrieved_count",
                             "rerank_scores", "refusal", "faithfulness", "quality_issues"],
        )



    def heal_plan_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
            你是一个金融 RAG 系统的自愈规划器（ReAct Agent 的 Thought 决策）。第一轮回答质量不达标，请从可用动作中选**一个**最可能修复问题的动作，并给出理由。

            当前状态：
            问题：{query}
            改写后查询：{rewritten_query}
            检索策略：{strategy}
            召回父块数：{retrieved_count}
            rerank 分数分布（前若干）：{rerank_scores}
            拒答检测：{refusal}
            在线忠实度分数：{faithfulness}
            触发信号：{quality_issues}
            已执行过的动作（不得重复）：{used_actions}

            可用动作（每次只选 1 个）：
            - rewrite：重写 query。适用：query 表述不清/丢失关键实体数字时间；拒答但已有召回
            - expand_topk：扩大精排保留数（rerank_k 升一档 5→10）。适用：召回相关但精排砍掉关键文档
            - strategy：换检索策略，strategy_to 必须填写（直接检索→假设问题检索→子查询检索 单向推进）。适用：当前策略不适合该问题
            - none：已无有效改进，提前结束自愈

            决策规则：
            1. 已执行过的动作组合（used_actions）禁止重复
            2. strategy 动作的 strategy_to 必须不同于当前策略 {strategy}
            3. 无把握或无新动作时选 none
            4. 只输出 JSON（不要输出其他文字）：
            {{
              "action": "rewrite | expand_topk | strategy | none",
              "strategy_to": "直接检索 | 假设问题检索 | 子查询检索 | null",
              "rerank_k": 5,
              "reason": "一句话理由"
            }}
            """,
            #   定义输入变量
            input_variables=["query", "rewritten_query", "strategy", "retrieved_count",
                             "rerank_scores", "refusal", "faithfulness", "quality_issues",
                             "used_actions"],
        )
