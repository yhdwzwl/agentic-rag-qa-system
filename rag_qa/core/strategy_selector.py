# 导入 LangChain 提示模板
from langchain.prompts import PromptTemplate
# 导入日志和配置
from base import logger, Config
# 导入 OpenAI
from openai import OpenAI
# [统一模型调用] 策略选择 LLM 轮换：模型列表来自 config.ini [llm] table_llm_models，
# 额度耗尽自动切换下一个模型（2026-10-03 接入 llm_router）
from rag_qa.core import llm_router
# 导入 re，用于策略名清洗（去标点/空白后做归一化匹配）
import re

# 四种规范策略名
STRATEGY_DIRECT = "直接检索"
STRATEGY_HYDE = "假设问题检索"
STRATEGY_SUBQUERY = "子查询检索"
STRATEGY_BACKTRACKING = "回溯问题检索"
STRATEGY_NAMES = (STRATEGY_DIRECT, STRATEGY_HYDE, STRATEGY_SUBQUERY, STRATEGY_BACKTRACKING)

# LLM 策略输出归一化映射：键为清洗后（小写、去空白/标点）的别名，值为规范策略名。
# 覆盖常见变体：带括号/标点/英文名/简写，保证"实际执行策略"与"评估记录策略"口径一致。
STRATEGY_ALIASES = {
    # 直接检索
    "直接检索": STRATEGY_DIRECT,
    "直接搜索": STRATEGY_DIRECT,
    "直接查询": STRATEGY_DIRECT,
    "直接": STRATEGY_DIRECT,
    "direct": STRATEGY_DIRECT,
    "directretrieval": STRATEGY_DIRECT,
    "直接检索策略": STRATEGY_DIRECT,
    # 假设问题检索（HyDE）
    "假设问题检索": STRATEGY_HYDE,
    "假设问题检索hyde": STRATEGY_HYDE,
    "假设问题": STRATEGY_HYDE,
    "假设答案检索": STRATEGY_HYDE,
    "假设答案": STRATEGY_HYDE,
    "hyde": STRATEGY_HYDE,
    "hyde检索": STRATEGY_HYDE,
    "hypotheticaldocumentembeddings": STRATEGY_HYDE,
    # 子查询检索
    "子查询检索": STRATEGY_SUBQUERY,
    "子查询": STRATEGY_SUBQUERY,
    "子问题检索": STRATEGY_SUBQUERY,
    "子问题": STRATEGY_SUBQUERY,
    "subquery": STRATEGY_SUBQUERY,
    "subqueries": STRATEGY_SUBQUERY,
    "subquery检索": STRATEGY_SUBQUERY,
    # 回溯问题检索
    "回溯问题检索": STRATEGY_BACKTRACKING,
    "回溯问题": STRATEGY_BACKTRACKING,
    "回溯检索": STRATEGY_BACKTRACKING,
    "回溯": STRATEGY_BACKTRACKING,
    "backtracking": STRATEGY_BACKTRACKING,
    "backtracking检索": STRATEGY_BACKTRACKING,
}


class StrategySelector:
    def __init__(self):
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=Config().DASHSCOPE_API_KEY,
                             base_url=Config().DASHSCOPE_BASE_URL)
        # 获取策略选择提示模板
        self.strategy_prompt_template = self._get_strategy_prompt()

    def call_dashscope(self, prompt):
        # 调用 DashScope API（[统一模型调用] table_llm_models 轮换，额度耗尽自动切下一个；
        # temperature=0.1 保证策略输出稳定；全部失败回退"直接检索"，不阻断主流程）
        try:
            return "".join(llm_router.stream_generate(
                Config(), prompt, temperature=0.1))
        except Exception as e:
            # 记录 API 调用失败
            logger.error(f"DashScope API 调用失败: {e}")
            # 默认返回直接检索
            return "直接检索"

    def _get_strategy_prompt(self):
        #   定义私有方法，获取策略选择 Prompt 模板
        return PromptTemplate(
            template="""
            你是一个金融领域智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从公告库中检索**特定事实信息**的问题，例如：
                    * 示例：
                        * 查询：某公司的融资公告是哪天发的？
                        * 策略：直接检索
                    * 查询：某某公司这次减持了多少股份？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：先基于查询生成一段假设答案，再用假设答案做检索，弥补 query 与文档的措辞鸿沟。
                * 适用场景：适用于查询较为抽象、直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：ST股票一般有什么风险？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个公司或多个方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：比较宁德时代和比亚迪的融资规模和减持情况。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：查询带有指代或上下文依赖（如"它"、"该公司"），需要结合对话历史改写成更基础、可独立检索的问题后再检索。
                * 适用场景：适用于查询较为复杂，有指代或上下文依赖，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：它上次减持之后，股价现在怎么样了？
                        * 策略：回溯问题检索

            根据用户查询 {query}，只输出以下四个策略名称之一："直接检索"、"假设问题检索"、"子查询检索"、"回溯问题检索"。
            不要输出标点、括号、解释、多余文字或换行，只输出策略名称本身。
            """
            ,
            input_variables=["query"],
        )

    def normalize(self, strategy):
        """
        把 LLM 返回的策略文本归一化为四种规范策略名之一。
        - 清洗：小写、去空白/标点/括号，先精确匹配，再按别名长度降序做包含匹配
          （长别名优先，避免"子查询"先匹配到"子问题"这类短串误判）
        - 无法识别（空/乱码/无关输出）→ 回退"直接检索"并记录 warning，保证主流程不崩
        :param strategy: LLM 原始输出
        :return: 规范策略名（STRATEGY_NAMES 之一）
        """
        if not strategy:
            logger.warning("策略选择返回空，回退直接检索")
            return STRATEGY_DIRECT
        cleaned = re.sub(r'[\s\-_（）()\[\]【】。.!！,，、;；:：\'"“”‘’]', '', str(strategy).strip().lower())
        if cleaned in STRATEGY_ALIASES:
            return STRATEGY_ALIASES[cleaned]
        # 包含匹配：按别名长度降序，优先长别名
        for alias in sorted(STRATEGY_ALIASES, key=len, reverse=True):
            if alias and (alias in cleaned or cleaned in alias):
                return STRATEGY_ALIASES[alias]
        logger.warning(f"策略名无法识别: {strategy!r}，回退直接检索")
        return STRATEGY_DIRECT

    #   定义方法，选择检索策略
    def select_strategy(self, query):
        #   调用 LLM 获取检索策略
        raw_strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        #   归一化为规范策略名（无法识别自动回退"直接检索"）
        strategy = self.normalize(raw_strategy)
        logger.info(f"为查询 '{query}' 选择的检索策略：{raw_strategy!r} → {strategy}")
        return strategy


if __name__ == '__main__':
    ss = StrategySelector()
    print(ss.select_strategy('AI大模型的课程大纲和JAVA的课程大纲，以及AI的课程学费？'))
