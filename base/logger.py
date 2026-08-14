
import os
import logging
from logging.handlers import TimedRotatingFileHandler

from .config import Config


def setup_logging(log_file=Config().LOG_FILE):
    """
    配置日志系统，支持按天自动轮转日志文件。
    日志文件命名格式: {原文件名}.YYYY-MM-DD.log
    """
    # 创建日志目录
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("EduRAG")
    logger.setLevel(logging.INFO)

    # 避免重复添加处理器（防止多次调用时日志重复输出）
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(name)s - %(levelname)s - %(message)s'
        )

        # ✅ 核心改动：使用 TimedRotatingFileHandler 替代 FileHandler
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',       # 每天午夜轮转
            interval=1,            # 每1天
            backupCount=30,        # 保留最近30天的日志，按需调整
            encoding='utf-8',
            utc=False              # 使用本地时间而非UTC
        )
        file_handler.suffix = "%Y-%m-%d.log"  # 轮转后文件后缀格式
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # 控制台处理器保持不变
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# 初始化日志器
logger = setup_logging()
