import logging
import os
import datetime
import sys
import inspect
from pathlib import Path
import traceback

def setup_symbol_logger(name=None, log_file=None, level=logging.INFO, console_output=False):
    """
    设置并返回一个配置好的logger实例，保存在check_symbol文件夹下
    
    Args:
        name: 日志记录器名称，默认使用调用模块名
        log_file: 日志文件名，默认自动生成时间戳文件名
        level: 日志级别，默认INFO
        console_output: 是否同时输出到控制台
    
    Returns:
        配置好的logger对象
    """
    # 如果未提供名称，使用调用者的模块名
    if name is None:
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        name = mod.__name__ if mod else 'forsee_symbol'
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 移除现有handlers（避免重复）
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建check_symbol文件夹（如果不存在）
    # 获取当前脚本所在目录的父目录（Forsee目录）
    current_dir = Path(__file__).resolve().parent.parent.parent
    log_dir = current_dir / "check_symbol"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # 确定日志文件路径
    if log_file is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d')
        log_file = log_dir / f"forsee_symbol_{timestamp}.log"
    else:
        if not os.path.isabs(log_file):
            log_file = log_dir / log_file
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    
    # 如果需要控制台输出
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"日志配置完成，将保存在: {log_file}")
    return logger

class SymbolLogger:
    """Forsee符号特征的日志记录工具类"""
    
    def __init__(self, name=None, log_file=None, level=logging.INFO):
        """初始化日志记录器"""
        self.logger = setup_symbol_logger(name, log_file, level)
    
    def info(self, msg, *args, **kwargs):
        """记录信息级别日志"""
        self.logger.info(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        """记录调试级别日志"""
        self.logger.debug(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """记录警告级别日志"""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """记录错误级别日志"""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """记录严重错误级别日志"""
        self.logger.critical(msg, *args, **kwargs)
    
    def tensor_info(self, tensor, name="tensor"):
        """记录张量信息"""
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                info = (f"张量[{name}]: shape={tensor.shape}, dtype={tensor.dtype}, "
                       f"device={tensor.device}, 均值={tensor.mean().item():.4f}")
                self.info(info)
            else:
                self.info(f"变量[{name}]: 类型={type(tensor)}, 不是张量")
        except Exception as e:
            self.error(f"检查张量[{name}]时出错: {str(e)}")
    
    def var_info(self, var, name="var"):
        """记录变量详细信息"""
        try:
            var_type = type(var).__name__
            
            if hasattr(var, "shape"):  # numpy数组或PyTorch张量
                shape_info = str(var.shape)
                self.info(f"变量[{name}]: 类型={var_type}, 形状={shape_info}")
            
            elif isinstance(var, (list, tuple)):
                self.info(f"变量[{name}]: 类型={var_type}, 长度={len(var)}")
                if len(var) > 0:
                    self.info(f"  - 第一个元素: 类型={type(var[0]).__name__}")
            
            elif isinstance(var, dict):
                self.info(f"变量[{name}]: 类型={var_type}, 键数量={len(var)}")
                if var:
                    self.info(f"  - 键列表: {list(var.keys())[:5]}")
            
            else:
                value = str(var)
                if len(value) > 100:
                    value = value[:100] + "..."
                self.info(f"变量[{name}]: 类型={var_type}, 值={value}")
                
        except Exception as e:
            self.error(f"记录变量[{name}]信息时出错: {str(e)}")
    
    def exception(self, e, context=""):
        """记录异常信息和堆栈跟踪"""
        self.error(f"异常: {type(e).__name__}, 信息: {str(e)}")
        if context:
            self.error(f"上下文: {context}")
        self.error(f"堆栈跟踪:\n{traceback.format_exc()}")
    
    def section(self, title):
        """记录分隔区块"""
        self.info(f"{'='*30} {title} {'='*30}")

def get_symbol_logger(name=None, log_file=None):
    """快速获取SymbolLogger的便捷函数"""
    return SymbolLogger(name, log_file)

# 示例用法
if __name__ == "__main__":
    # 创建日志记录器
    logger = get_symbol_logger("example")
    
    # 记录一般信息
    logger.section("开始测试")
    logger.info("这是一条测试信息")
    
    # 记录变量信息
    test_list = [1, 2, 3, 4, 5]
    logger.var_info(test_list, "test_list")
    
    # 模拟记录张量信息
    try:
        import torch
        test_tensor = torch.randn(64, 23)
        logger.tensor_info(test_tensor, "random_tensor")
    except ImportError:
        logger.warning("PyTorch未安装，跳过张量测试")
    
    # 记录异常
    try:
        result = 1 / 0
    except Exception as e:
        logger.exception(e, "除零测试")
    
    logger.section("测试结束")