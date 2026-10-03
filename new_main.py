# -*- coding: utf-8 -*-
"""
兼容层：new_main 现为薄壳。

IntegratedQASystem 继承 SessionQASystem（rag_qa/core/session_qa.py，会话编排单一权威），
类名与接口保持不变，app.py 与 CLI（main()）零改动。实际逻辑（初始化 / 会话方法 /
完整编排 / LLM 入口）已全部剥离至 SessionQASystem，本文件只保留 main()。
"""
# 导入会话编排类（单一权威）
from rag_qa.core.session_qa import SessionQASystem
# 导入配置和日志
from base import logger
# 导入UUID生成唯一会话ID
import uuid


class IntegratedQASystem(SessionQASystem):
    """兼容层：保留类名与接口，app.py / CLI 零改动。实际逻辑在 SessionQASystem。"""

    pass


def main():
    qa_system = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    print("\n欢迎使用集成问答系统！")
    print(f"会话ID: {session_id}")
    print(f"支持的学科类别：{qa_system.config.VALID_SOURCES}")
    print("输入查询进行问答，输入 'exit' 退出。")

    try:
        while True:
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                logger.info("退出系统")
                print("再见！")
                break
            source_filter = input(
                f"请输入学科类别 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                logger.warning(f"无效的学科类别 '{source_filter}'，将不过滤")
                source_filter = None

            print("\n答案: ", end="", flush=True)  # 开始打印答案
            answer = ""  # 用于累积完整答案
            # 迭代 query 方法的生成器
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:  # 仅当 token 非空时打印
                    print(token, end="", flush=True)  # 流式打印
                    answer += token  # 累积答案
                if is_complete:  # 如果是完整答案或流结束，换行并退出循环
                    print()  # 换行
                    break

            # 打印对话历史
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")
    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"发生错误: {e}")
    finally:
        qa_system.mysql_client.close()


if __name__ == "__main__":
    main()
