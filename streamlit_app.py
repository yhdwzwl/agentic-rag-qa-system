# -*- coding: utf-8 -*-
"""
金融RAG系统前端 - Streamlit版
直接运行：streamlit run streamlit_app.py
"""
import sys
import os
import uuid

ROOT = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system"
sys.path.insert(0, ROOT)

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="金融公告RAG问答系统",
    page_icon="📊",
    layout="wide",
)

# 会话 id 持久化（浏览器会话级；user_id 不传（None），由 memory_manager 按 session_id 稳定映射到写死列表）
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# 初始化会话编排层（只初始化一次；@st.cache_resource 单例，多标签共享实例，会话按 session_id 隔离）
@st.cache_resource
def init_qa():
    from rag_qa.core.session_qa import SessionQASystem

    return SessionQASystem()

# 页面标题
st.title("📊 金融公告RAG问答系统")
st.caption("基于东方财富公告数据的检索增强生成系统")

# 侧边栏：显示中间产物
with st.sidebar:
    st.header("🔍 检索过程")
    st.markdown("---")
    if "last_contexts" not in st.session_state:
        st.session_state["last_contexts"] = []
    if "last_sources" not in st.session_state:
        st.session_state["last_sources"] = []
    if "last_strategy" not in st.session_state:
        st.session_state["last_strategy"] = ""
    if "last_time" not in st.session_state:
        st.session_state["last_time"] = {}

    st.subheader("当前查询策略")
    st.write(st.session_state["last_strategy"] or "暂无")

    st.subheader("耗时统计")
    if st.session_state["last_time"]:
        for k, v in st.session_state["last_time"].items():
            if isinstance(v, (int, float)):
                st.write(f"{k}: {v:.2f}s")
            else:
                st.write(f"{k}: {v}")
    else:
        st.write("暂无")

    st.subheader("信息来源")
    if st.session_state["last_sources"]:
        for i, src in enumerate(st.session_state["last_sources"], 1):
            with st.expander(f"来源 {i}: {src.get('title', '未知公告')}"):
                st.write(f"股票代码: {src.get('code', '-')}")
                st.write(f"公告日期: {src.get('time', '-')}")
                st.write(f"公告类型: {src.get('type', '-')}")
    else:
        st.write("暂无")

    st.subheader("检索到的上下文")
    if st.session_state["last_contexts"]:
        for i, ctx in enumerate(st.session_state["last_contexts"], 1):
            with st.expander(f"上下文 {i}"):
                st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
    else:
        st.write("暂无")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是金融公告问答助手，你可以问我关于上市公司公告的问题。", "sources": []}
    ]

# 显示对话历史
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 信息来源（{len(msg['sources'])}个）"):
                for i, src in enumerate(msg["sources"], 1):
                    st.write(f"{i}. {src.get('title', '未知公告')} | {src.get('code','')} | {src.get('time','')}")

# 输入框
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户输入
    st.session_state["messages"].append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.write(prompt)

    # 调用RAG系统
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []

        try:
            qa = init_qa()
            # 流式生成（完整会话链路：BM25 快路径 → RAG 主链路 → 窗口/工作/长期三层记忆）
            # BM25 精确命中判定：query() 第一次迭代即返回 (非空完整答案, True) 且非"未找到答案"分支
            # （RAG 流式首轮必为 (token, False)，末轮为 ("", True)，两者均不会误判为精确命中）
            exact_hit = False
            first_yield = True
            for token, is_complete in qa.query(
                    prompt,
                    source_filter=None,
                    session_id=st.session_state["session_id"],
                    user_id=None):
                if token:
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
                if first_yield and is_complete and token and token != "未找到答案":
                    exact_hit = True
                first_yield = False
                if is_complete:
                    break
            message_placeholder.markdown(full_response)

            # 更新侧边栏的中间产物
            if exact_hit:
                # BM25 精确命中：不经过 RAGSystem，last_query_result 是上次残留，侧边栏直接标记并清空
                st.session_state["last_strategy"] = "✅ 精确问答命中（MySQL 标准问答库直返，未走 RAG）"
                st.session_state["last_contexts"] = []
                st.session_state["last_sources"] = []
                st.session_state["last_time"] = {"模式": "MySQL 标准问答库直接命中", "总耗时": "—"}
            else:
                result = qa.rag_system.last_query_result
                st.session_state["last_strategy"] = result["retrieval_strategy"]
                st.session_state["last_contexts"] = result["contexts"]
                # 提取来源信息
                sources = []
                for doc_meta in result.get("source_metas", []):
                    sources.append({
                        "title": doc_meta.get("title", "未知公告"),
                        "code": doc_meta.get("code", "-"),
                        "time": doc_meta.get("time", "-"),
                        "type": doc_meta.get("source", "-")
                    })
                st.session_state["last_sources"] = sources
                st.session_state["last_time"] = {
                    "分类": result["classify_time"],
                    "检索": result["retrieval_time"],
                    "生成": result["generation_time"],
                    "总耗时": result["total_time"],
                }

        except Exception as e:
            full_response = f"出错了: {e}"
            message_placeholder.markdown(full_response)

    # 保存对话历史（包括来源）
    st.session_state["messages"].append({"role": "assistant", "content": full_response, "sources": sources})

    # 立即重跑页面，让侧边栏展示本次检索过程（策略/耗时/来源/上下文）
    st.rerun()

# 页面底部
st.markdown("---")
st.caption("💡 小提示：右侧边栏可以看到每次检索的上下文、耗时统计和信息来源")
