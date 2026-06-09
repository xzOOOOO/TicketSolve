"""
日志配置模块

提供统一的日志记录功能，支持控制台和文件输出。
设计要点：
1. 同时输出到控制台和文件（开发时看控制台，排查问题时看文件）
2. 文件自动轮转（防止单文件过大占用磁盘）
3. 错误日志单独存放（快速定位问题）
4. 控制台输出简洁，文件输出详细（含文件名、行号、函数名）
"""
# logging：Python 标准日志模块
import logging
# sys：用于获取标准输出流
import sys
# Path：用于构建日志文件路径
from pathlib import Path
# RotatingFileHandler：自动轮转的文件日志处理器
from logging.handlers import RotatingFileHandler


def setup_logger(name: str = "TicketSolve", log_level: int = logging.DEBUG) -> logging.Logger:
    """
    配置并返回日志记录器

    参数：
        name: 日志记录器名称，用于区分不同模块的日志
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）

    返回：
        配置好的 Logger 实例

    功能特性：
        1. 同时输出到控制台和文件
        2. 文件自动轮转（防止单文件过大）
        3. 错误日志单独存放
    """
    # logger：获取或创建指定名称的日志记录器
    logger = logging.getLogger(name)
    # 设置日志级别，低于此级别的日志会被忽略
    logger.setLevel(log_level)

    # 如果已有处理器，说明已经初始化过，直接返回（避免重复添加处理器）
    if logger.handlers:
        return logger

    # log_dir：日志文件存放目录（项目根目录下的 logs 文件夹）
    log_dir = Path(__file__).parent.parent / "logs"
    # exist_ok=True：目录已存在时不报错
    log_dir.mkdir(exist_ok=True)

    # detailed_formatter：详细格式，包含时间、级别、名称、文件名、行号、函数名、消息
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # simple_formatter：简洁格式，只包含时间、级别、消息
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # console_handler：控制台处理器，输出到标准输出
    console_handler = logging.StreamHandler(sys.stdout)
    # 控制台只显示 INFO 及以上级别（避免 DEBUG 信息刷屏）
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # file_handler：应用日志文件处理器，记录 DEBUG 及以上级别
    # RotatingFileHandler：当文件达到 maxBytes 时自动轮转，保留 backupCount 个备份
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,               # 保留 5 个备份
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # error_handler：错误日志文件处理器，只记录 ERROR 及以上级别
    error_handler = RotatingFileHandler(
        log_dir / "error.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,               # 保留 5 个备份
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # 将三个处理器添加到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


def get_logger(name: str = "mygongdan") -> logging.Logger:
    """
    获取日志记录器实例

    参数：
        name: 日志记录器名称

    返回：
        Logger 实例
    """
    return logging.getLogger(name)


# logger：项目默认日志记录器实例，通过 from logger import logger 使用
logger = setup_logger()
