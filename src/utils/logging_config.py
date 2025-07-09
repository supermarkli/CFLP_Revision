import logging
import os
from datetime import datetime
from pathlib import Path

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging(create_file=False):
    """设置日志配置，包括控制台和文件输出
    
    配置两个处理器：
    1. 控制台处理器：只显示 INFO 级别以上的简要信息
    2. 文件处理器：记录 DEBUG 级别以上的详细信息
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 检查是否已添加控制台 handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 检查是否已添加文件 handler
    if create_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        log_dir = Path(PROJECT_ROOT) / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志文件已创建: {log_file}")

    return logger

def get_logger(create_file=False):
    """获取logger实例"""
    logger = logging.getLogger()
    logger = setup_logging(create_file)
    return logger


