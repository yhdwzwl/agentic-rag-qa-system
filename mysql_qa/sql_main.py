# 导入 MongoDB 客户端
from db.mongo_client import MongoDBClient
# 导入 Redis 客户端
from cache.redis_client import RedisClient
# 导入 BM25 搜索
from retrieval.bm25_search import BM25Search
# 导入日志
from base import logger
# 导入时间库
import time


class MongoDBQASystem:
    def __init__(self, bm25_threshold:float=0.85):
        # 初始化日志
        self.logger = logger
        # 初始化 MySQL 客户端
        self.mongo_client = MongoDBClient()
        # 初始化 Redis 客户端
        self.redis_client = RedisClient()  # 确保Redis也被纳入生命周期管理
        # 初始化 BM25 搜索
        self.bm25_search = BM25Search(self.redis_client, self.mongo_client) # 初始化BM25搜索 缓存 + 持久化
        self.bm25_threshold = bm25_threshold    # 阈值配置化

    def query(self, query: str) -> str:
        start_time = time.time()
        self.logger.info(f"处理查询: '{query}'")

        answer, status = self.bm25_search.search(query, threshold=self.bm25_threshold)

        # 根据状态码记录结构化日志
        processing_time = time.time() - start_time
        self.logger.info(f"查询完成 | 耗时: {processing_time:.3f}s | 状态: {status}")

        if status in ("cache_hit", "bm25_hit"):
            return answer

        # fallback / invalid / error 均走 RAG 兜底
        self.logger.info(f"BM25未解决 (status={status}), 准备调用RAG系统")
        # TODO: 这里接入你的向量检索 + LLM 生成逻辑
        return "正在为您深度检索..."

    def close(self):
        """统一资源释放入口"""
        try:
            self.mongo_client.close()
            self.redis_client.close()  # 补全Redis连接关闭
            self.logger.info("数据库与缓存连接已安全关闭")
        except Exception as e:
            self.logger.error(f"关闭连接时发生错误: {e}")

def main():
    # 初始化 MongoDB 系统
    Mongo_system = MongoDBQASystem()
    try:
        # 打印欢迎信息
        print("\n欢迎使用 MongoDB 问答系统！")
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
            answer = Mongo_system.query(query)
            # 打印答案
            print(f"\n答案: {answer}")
    except Exception as e:
        # 记录系统错误
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 关闭  连接
        Mongo_system.close()


if __name__ == "__main__":
    # 运行主程序
    main()
