# 瀵煎叆閰嶇疆瑙ｆ瀽搴?
import configparser
# 瀵煎叆璺緞鎿嶄綔搴?
import os


class Config:
    # 鍒濆鍖栭厤缃紝鍔犺浇 config.ini 鏂囦欢
    def __init__(self, config_file=r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system\config.ini"):
        # 鍒涘缓閰嶇疆瑙ｆ瀽鍣?
        self.config = configparser.ConfigParser()
        # 璇诲彇閰嶇疆鏂囦欢
        self.config.read(config_file, encoding='utf-8')  # windows鍖単bk閿欒锛屽鍔爀ncoding=utf-8

        # MySQL 閰嶇疆
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        self.MYSQL_PORT = self.config.getint('mysql', 'port', fallback=3306)
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='')
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='')

        # MongoDB 閰嶇疆
        self.MONGO_URL = self.config.get('mongodb', 'mongo_url', fallback='localhost')
        self.MONGO_USER = self.config.get('mongodb', 'user', fallback='root')
        self.MONGO_DB = self.config.get('mongodb', 'mongo_db_name', fallback='subjects_kg')
        self.MONGO_COLLECTION = self.config.get('mongodb', 'mongo_collection_name', fallback='subjects_kg')
        # Redis 閰嶇疆
        # Redis 涓绘満鍦板潃
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        # Redis 绔彛
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)
        # Redis 瀵嗙爜
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='')
        # Redis 鏁版嵁搴撶紪鍙?
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)

        # Redis 鍚戦噺缂撳瓨閰嶇疆
        self.REDIS_VECTOR_CACHE_INDEX = self.config.get('redis_vector_cache', 'index_name', fallback='rag_query_vector_cache')
        self.REDIS_VECTOR_CACHE_DIM = self.config.getint('redis_vector_cache', 'vector_dim', fallback=768)
        self.REDIS_VECTOR_CACHE_SIMILARITY_THRESHOLD = self.config.getfloat('redis_vector_cache', 'similarity_threshold', fallback=0.95)
        self.REDIS_VECTOR_CACHE_TEMPORARY_TTL = self.config.getint('redis_vector_cache', 'temporary_ttl', fallback=86400)
        self.REDIS_VECTOR_CACHE_FORMAL_TTL = self.config.getint('redis_vector_cache', 'formal_ttl', fallback=604800)
        self.REDIS_VECTOR_CACHE_HOT_THRESHOLD = self.config.getint('redis_vector_cache', 'hot_threshold', fallback=3)
        self.REDIS_VECTOR_CACHE_MAX_SIZE = self.config.getint('redis_vector_cache', 'max_cache_size', fallback=100000)

        # Milvus 閰嶇疆
        # Milvus 涓绘満鍦板潃
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        # Milvus 绔彛
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback='19530')
        # Milvus 鏁版嵁搴撳悕
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='itcast')
        # Milvus 闆嗗悎鍚?
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='edurag_final')

        # LLM 閰嶇疆
        # LLM 妯″瀷鍚?
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen3.7-plus')
        # DashScope API 瀵嗛挜
        self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key')
        # DashScope API 鍦板潃
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 妫€绱㈠弬鏁?
        # 鐖跺潡澶у皬
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 瀛愬潡澶у皬
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 鍧楅噸鍙犲ぇ灏?
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 妫€绱㈣繑鍥炴暟閲?
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 鏈€缁堝€欓€夋暟閲?
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 璁惧閰嶇疆锛歝pu / dml(DirectML GPU) / cuda
        self.DEVICE = self.config.get('device', 'device', fallback='cpu')

        # 搴旂敤閰嶇疆
        # 鏈夋晥鏉ユ簮鍒楄〃
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]'))
        # 瀹㈡湇鐢佃瘽
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')
        # 鏃ュ織鏂囦欢璺緞
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')

        # model path
        self.nlp_bert_doc_seg = r"F:\small_project\aiagent\ailearn\rag/Itcast_qa_system/rag_qa/models/nlp_bert_document-segmentation_chinese-base"
        self.bert_intent_cls = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system/rag_qa/models/bert_classification_output"


if __name__ == '__main__':
    conf = Config()
    print(conf.MYSQL_HOST)

