# 导入 MySQL 客户端
from db.mysql_client import MySQLClient
# 导入 BM25 搜索
from retrieval.bm25_search import BM25Search
# 导入日志
from base import logger
# 导入时间库
import time


class MySQLQASystem:
    def __init__(self, bm25_threshold:float=0.9):
        # 初始化日志
        self.logger = logger
        # 初始化 MySQL 客户端
        self.mysql_client = MySQLClient()
        # 初始化 BM25 搜索（纯内存，无Redis）
        self.bm25_search = BM25Search(self.mysql_client)
        self.bm25_threshold = bm25_threshold    # 阈值配置化

    def query(self, query: str) -> str:
        start_time = time.time()
        self.logger.info(f"处理查询: '{query}'")

        answer, status = self.bm25_search.search(query, threshold=self.bm25_threshold)

        # 根据状态码记录结构化日志
        processing_time = time.time() - start_time
        self.logger.info(f"查询完成 | 耗时: {processing_time:.3f}s | 状态: {status}")

        if status == "bm25_hit":
            return answer

        # fallback / invalid / error 均走 RAG 兜底
        self.logger.info(f"BM25未解决 (status={status}), 准备调用RAG系统")
        # TODO: 这里接入你的向量检索 + LLM 生成逻辑
        return "正在为您深度检索..."

    def close(self):
        """统一资源释放入口"""
        try:
            self.mysql_client.close()
            self.logger.info("数据库连接已安全关闭")
        except Exception as e:
            self.logger.error(f"关闭连接时发生错误: {e}")

def main():
    # 初始化 MySQL 问答系统
    system = MySQLQASystem()
    try:
        # 打印欢迎信息
        print("\n欢迎使用 MySQL 标准问答缓存系统！")
        print("输入查询进行问答，输入 'exit' 退出。")
        while True:
            # 获取用户输入
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                # 记录退出日志
                logger.info("退出系统")
                # 打印退出信息
                print("再见！")
                break
            # 执行查询
            answer = system.query(query)
            # 打印答案
            print(f"\n答案: {answer}")
    except Exception as e:
        # 记录系统错误
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 关闭连接
        system.close()


if __name__ == "__main__":
    # 运行主程序
    main()
