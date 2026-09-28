# 导入 MongoDB 驱动
from pymongo import MongoClient, ASCENDING
# 导入 pandas (保留用于读取 CSV)
import pandas as pd
# 导入配置和日志
from base import Config, logger


class MongoDBClient:
    def __init__(self):
        self.logger = logger
        try:
            # 1. 获取配置 (假设 Config 类里有 MONGO_URI 等配置)
            # 如果 Config 里没有，你可以直接用下面的字符串替换
            config = Config()
            mongo_uri = getattr(config, 'MONGO_URL', 'mongodb://localhost:27017/')
            db_name = getattr(config, 'MONGO_DB', 'rag_database')
            collection_name = getattr(config, 'MONGO_COLLECTION', 'jpkb')

            # 2. 建立连接
            self.client = MongoClient(mongo_uri)
            # 选择数据库和集合 (相当于 MySQL 的 Database 和 Table)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]

            # 3. 创建索引 (可选，但推荐，能加速根据问题查找)
            # 在 MongoDB 中，我们通常给 'question' 字段加索引
            self.collection.create_index([("question", ASCENDING)], unique=True)

            self.logger.info(f"MongoDB 连接成功 -> {db_name}.{collection_name}")

        except Exception as e:
            self.logger.error(f"MongoDB 连接失败: {e}")
            raise

    def insert_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path, encoding='utf-8')
            column_mapping = {
                "学科名称": "subject_name",
                "问题": "question",
                "答案": "answer"
            }
            data.rename(columns=column_mapping, inplace=True)
            if "subject_name" in data.columns:
                # apply 函数会对每一行执行 lambda 操作
                data["subject_name"] = data["subject_name"].apply(
                    lambda x: str(x).replace("学科", "") if pd.notna(x) else x
                )
            records = data.to_dict('records')
            if not records:
                self.logger.warning("CSV 文件为空，未插入数据")
                return

            result = self.collection.insert_many(records)
            self.logger.info(f"数据插入成功，共插入 {len(result.inserted_ids)} 条记录")

        except Exception as e:
            self.logger.error(f"数据插入失败: {e}")
            raise

    def fetch_questions(self):
        # 获取所有问题
        try:
            # projection={"question": 1, "_id": 0} 表示只取 question 字段，不取 _id
            cursor = self.collection.find({}, {"question": 1, "_id": 0})

            # 将结果转为列表
            results = [doc['question'] for doc in cursor]

            self.logger.info(f"成功获取 {len(results)} 个问题")
            return results

        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            return []

    def fetch_answer(self, question):
        # 获取指定问题的答案
        try:
            # find_one 类似于 SQL 的 SELECT ... LIMIT 1
            result = self.collection.find_one({"question": question}, {"answer": 1, "_id": 0})

            # 如果找到了，返回 answer 字段的值；否则返回 None
            return result['answer'] if result else None

        except Exception as e:
            self.logger.error(f"答案获取失败: {e}")
            return None

    def close(self):
        # 关闭数据库连接
        try:
            self.client.close()
            self.logger.info("MongoDB 连接已关闭")
        except Exception as e:
            self.logger.error(f"关闭连接失败: {e}")


