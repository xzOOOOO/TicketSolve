"""
日志配置模块
提供统一的日志记录功能，支持控制台和文件输出
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from functools import wraps
from typing import Callable, Any
import traceback


class ContextFilter(logging.Filter):
    """
    上下文过滤器
    为日志记录添加额外的上下文信息
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.ticket_id = getattr(record, 'ticket_id', '-')
        return True


def setup_logger(name: str = "TicketSolve", log_level: int = logging.DEBUG) -> logging.Logger:
    """
    配置并返回日志记录器
    
    Args:
        name: 日志记录器名称
        log_level: 日志级别
    
    Returns:
        配置好的 Logger 实例
    
    Features:
        1. 同时输出到控制台和文件
        2. 文件自动轮转（防止单文件过大）
        3. 错误日志单独存放
        4. 支持上下文信息
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    if logger.handlers:
        return logger
    
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    console_handler.addFilter(ContextFilter())
    
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    error_handler = RotatingFileHandler(
        log_dir / "error.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str = "mygongdan") -> logging.Logger:
    """
    获取日志记录器实例
    
    Args:
        name: 日志记录器名称
    
    Returns:
        Logger 实例
    """
    return logging.getLogger(name)


class LogContext:
    """
    日志上下文管理器
    用于在特定代码块中添加额外的日志上下文
    """
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)
        if exc_type is not None:
            self.logger.error(f"异常发生: {exc_type.__name__}: {exc_val}")
            self.logger.debug(f"异常堆栈:\n{''.join(traceback.format_exception(exc_type, exc_val, exc_tb))}")
        return False


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    函数调用日志装饰器
    
    Args:
        logger: 日志记录器
        level: 日志级别
    
    Usage:
        @log_function_call(logger)
        def my_function(a, b):
            return a + b
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = func.__name__
            logger.log(level, f"调用函数: {func_name}, args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"函数 {func_name} 执行成功, 返回类型: {type(result).__name__}")
                return result
            except Exception as e:
                logger.error(f"函数 {func_name} 执行失败: {type(e).__name__}: {str(e)}")
                raise
        return wrapper
    return decorator


def log_exceptions(logger: logging.Logger, reraise: bool = True, default_return: Any = None):
    """
    异常捕获日志装饰器
    
    Args:
        logger: 日志记录器
        reraise: 是否重新抛出异常
        default_return: 发生异常时的默认返回值
    
    Usage:
        @log_exceptions(logger, reraise=False, default_return=None)
        def risky_function():
            # 可能抛出异常的代码
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"函数 {func.__name__} 发生异常: {type(e).__name__}: {str(e)}")
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


logger = setup_logger()
