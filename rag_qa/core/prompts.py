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


