# -*- coding: utf-8 -*-
"""
统一 LLM 模型轮换调用器（额度耗尽自动切换下一个模型）。

背景：主链路（答案生成 / 改写 / HyDE / 子查询 / 自愈）此前只使用单模型
conf.LLM_MODEL，任一模型额度耗尽即整条链路失败（2026-10-03 Bug1 实测：
glm-5.3 403 insufficient_quota 直接中断所有生成）。

本模块提供与 DashScope compatible-mode 兼容的 OpenAI 客户端封装：
- 模型列表统一从 [llm] table_llm_models 读取（未配置时回退 [llm] model）；
- 请求抛出额度耗尽类错误（HTTP 403/402/429 或 insufficient_quota/额度等
  关键词）→ 自动切换下一个模型重试；
- 非额度类调用失败（网络/超时/空响应）同样切换下一个模型；
- 已向调用方输出部分 token 后失败 → 中断并抛错，不重试（避免流式内容
  重复/接龙错乱，前端已展示的内容不可撤回）；
- 所有模型均失败 → raise RuntimeError，由调用方 fail-open 降级
  （generate_answer 兜底 try 会向用户返回错误提示，不静默、不崩链路）。

调用方无需关心额度状态；本模块不持有跨调用状态（无环形游标/失败黑名单，
每次调用独立重试全部模型——在线链路每次请求都应尽力而为，与
table_enhancer 批处理场景的断点语义不同）。
"""
import logging
from openai import OpenAI

logger = logging.getLogger("EduRAG")

# 额度耗尽类错误关键词（错误消息常见于 DashScope compatible-mode）
_QUOTA_KEYWORDS = (
    "insufficient_quota", "quota", "exhausted", "limit exceeded",
    "无额度", "额度不足", "额度用尽", "欠费", "余额不足",
)


def is_quota_error(exc) -> bool:
    """判定异常是否属于"额度耗尽"（HTTP 403/402/429 或消息含额度关键词）。

    与 table_enhancer.TableLLMEnhancer._is_quota_error 判定口径一致，
    供轮换与日志分类复用。
    """
    msg = str(exc).lower()
    status = None
    for attr in ("status_code", "http_status", "code"):
        v = getattr(exc, attr, None)
        if v is not None:
            status = v
            break
    if isinstance(status, int) and status in (402, 403, 429):
        return True
    return any(k in msg for k in _QUOTA_KEYWORDS)


def _model_list(conf):
    """模型列表：优先 [llm] table_llm_models；未配置回退 [llm] model。"""
    models = list(getattr(conf, "TABLE_LLM_MODELS", None) or [])
    if not models and getattr(conf, "LLM_MODEL", None):
        models = [conf.LLM_MODEL]
    return models


def stream_generate(conf, prompt, system_prompt=None, timeout=30, temperature=None):
    """按 table_llm_models 轮换流式生成；额度耗尽自动切下一个模型。

    生成器函数：yield 模型输出的文本 chunk；正常完成即结束。
    所有模型失败抛 RuntimeError（调用方 fail-open）；已输出部分内容后
    失败同样抛 RuntimeError（不重复输出）。
    :param conf: base.config.Config 实例（需含 TABLE_LLM_MODELS /
                 LLM_MODEL / DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL）
    :param prompt: 用户侧 prompt 文本
    :param system_prompt: system 角色内容（默认"你是一个有用的助手。"，
                与 new_main.call_dashscope 原行为一致）
    :param timeout: 单次请求超时（秒）
    :param temperature: 采样温度（None=使用服务端默认；策略选择等需稳定
                输出的调用方可传 0.1）
    """
    models = _model_list(conf)
    if not models:
        raise RuntimeError("未配置任何 LLM 模型（[llm] model / table_llm_models 均为空）")
    sys_msg = system_prompt if system_prompt else "你是一个有用的助手。"
    last_err = None
    for model in models:
        sent_any = False
        try:
            client = OpenAI(api_key=conf.DASHSCOPE_API_KEY,
                            base_url=conf.DASHSCOPE_BASE_URL)
            kwargs = {"timeout": timeout, "stream": True}
            if temperature is not None:
                kwargs["temperature"] = temperature
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt},
                ],
                **kwargs,
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    sent_any = True
                    yield chunk.choices[0].delta.content
            return  # 本模型完整输出，成功结束
        except Exception as e:
            if sent_any:
                # 已输出部分内容：中断而非重试，避免流式内容重复/接龙错乱
                raise RuntimeError(f"模型 {model} 流式中断（已输出部分内容，不重试以免重复）: {e}")
            if is_quota_error(e):
                logger.warning(f"模型 {model} 额度耗尽，自动切换下一个模型: {e}")
            else:
                logger.warning(f"模型 {model} 调用失败，尝试下一个模型: {e}")
            last_err = f"model={model} failed: {e}"
    raise RuntimeError(f"所有 LLM 模型均失败: {last_err}")


def make_streaming_callable(conf):
    """返回 callable(prompt) -> generator，供 RAGSystem(self.llm) 使用。

    与 new_main.call_dashscope 等价（yield 字符串流）；new_main /
    streamlit_app 均可复用，避免两处重复实现轮换逻辑。
    """
    def _call(prompt):
        return stream_generate(conf, prompt)
    return _call
