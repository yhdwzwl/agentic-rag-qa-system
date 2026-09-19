# 导入配置解析库
import configparser
# 导入路径操作库
import os


class Config:
    # 初始化配置，加载 config.ini 文件
    def __init__(self, config_file=r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system\config.ini"):
        # 创建配置解析器
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read(config_file, encoding='utf-8')  # windows包gbk错误，增加encoding=utf-8

        # MySQL 配置
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        self.MYSQL_PORT = self.config.getint('mysql', 'port', fallback=3306)
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='')
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='')

        # MongoDB 配置
        self.MONGO_URL = self.config.get('mongodb', 'mongo_url', fallback='localhost')
        self.MONGO_USER = self.config.get('mongodb', 'user', fallback='root')
        self.MONGO_DB = self.config.get('mongodb', 'mongo_db_name', fallback='subjects_kg')
        self.MONGO_COLLECTION = self.config.get('mongodb', 'mongo_collection_name', fallback='subjects_kg')
        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        # Redis 端口
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)
        # Redis 密码
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='')
        # Redis 数据库编号
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)

        # Redis 向量缓存配置
        self.REDIS_VECTOR_CACHE_INDEX = self.config.get('redis_vector_cache', 'index_name', fallback='rag_query_vector_cache')
        self.REDIS_VECTOR_CACHE_DIM = self.config.getint('redis_vector_cache', 'vector_dim', fallback=768)
        self.REDIS_VECTOR_CACHE_SIMILARITY_THRESHOLD = self.config.getfloat('redis_vector_cache', 'similarity_threshold', fallback=0.95)
        self.REDIS_VECTOR_CACHE_TEMPORARY_TTL = self.config.getint('redis_vector_cache', 'temporary_ttl', fallback=86400)
        self.REDIS_VECTOR_CACHE_FORMAL_TTL = self.config.getint('redis_vector_cache', 'formal_ttl', fallback=604800)
        self.REDIS_VECTOR_CACHE_HOT_THRESHOLD = self.config.getint('redis_vector_cache', 'hot_threshold', fallback=3)
        self.REDIS_VECTOR_CACHE_MAX_SIZE = self.config.getint('redis_vector_cache', 'max_cache_size', fallback=100000)

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        # Milvus 端口
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback='19530')
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='itcast')
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='edurag_final')

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen3.7-plus')
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key')
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        # 有效来源列表
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')
        # 日志文件路径
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')

        # model path
        self.nlp_bert_doc_seg = r"F:\small_project\aiagent\ailearn\rag/Itcast_qa_system/rag_qa/models/nlp_bert_document-segmentation_chinese-base"
        self.bert_intent_cls = r"F:\small_project\aiagent\ailearn\rag/Itcast_qa_system/rag_qa/core/bert_query_classifier"


if __name__ == '__main__':
    conf = Config()
    print(conf.MYSQL_HOST)
