import torch
import numpy as np
'''
输入文件路径，得到向量列表
code--json data_get.py中analyze_code_by_lines方法
json--vec get_vector_non_vul.py
'''
from .data_get import LexicalAanalysis
def calculate_ratios(data):
    result = {}
    
    #logger.info("开始计算特征比例")
    
    # 计算命名风格比例
    total_names = sum(data['naming'].values())
    result['total_names'] = total_names
    if total_names > 0:
        for style in ['camel_case', 'snake_case', 'constant_case', 'lowercase', 'other']:
            result[f'{style}_ratio'] = data['naming'][style] / total_names
    else:
        #logger.warning("未找到命名风格数据")
        for style in ['camel_case', 'snake_case', 'constant_case', 'lowercase', 'other']:
            result[f'{style}_ratio'] = 0
            
    # 计算分号使用比例
    total_semicolons = data['symbols']['with_semicolons'] + data['symbols']['without_semicolons']
    result['semicolon_ratio'] = data['symbols']['with_semicolons'] / total_semicolons if total_semicolons > 0 else 0
    
    # 计算引号使用比例
    total_quotes = sum(data['symbols']['quotes'].values())
    result['total_quotes'] = total_quotes
    if total_quotes > 0:
        for quote_type in ['single', 'double', 'backtick']:
            result[f'{quote_type}_quote_ratio'] = data['symbols']['quotes'][quote_type] / total_quotes
    else:
        #logger.warning("未找到引号使用数据")
        for quote_type in ['single', 'double', 'backtick']:
            result[f'{quote_type}_quote_ratio'] = 0
            
    # 计算函数风格比例
    total_functions = sum(data['symbols']['function_style'].values())
    result['total_functions'] = total_functions
    if total_functions > 0:
        result['arrow_func_ratio'] = data['symbols']['function_style']['arrow'] / total_functions
        result['regular_func_ratio'] = data['symbols']['function_style']['regular'] / total_functions
    else:
        #logger.warning("未找到函数风格数据")
        result['arrow_func_ratio'] = 0
        result['regular_func_ratio'] = 0
        
    # 计算运算符使用比例
    total_equality = sum(data['operators']['equality'].values())
    result['total_equality_ops'] = total_equality
    if total_equality > 0:
        result['strict_equality_ratio'] = data['operators']['equality']['strict'] / total_equality
        result['non_strict_equality_ratio'] = data['operators']['equality']['non_strict'] / total_equality
    else:
        #logger.warning("未找到相等运算符数据")
        result['strict_equality_ratio'] = 0
        result['non_strict_equality_ratio'] = 0
        
    # 计算空格使用比例
    total_spacing = sum(data['operators']['spacing'].values())
    result['total_spacing'] = total_spacing
    if total_spacing > 0:
        result['spacing_with_space_ratio'] = data['operators']['spacing']['with_space'] / total_spacing
    else:
        #logger.warning("未找到运算符间距数据")
        result['spacing_with_space_ratio'] = 0
        
    # 计算一致性指标
    result['naming_consistency'] = max(result['camel_case_ratio'], 
                                     result['snake_case_ratio'],
                                     result['constant_case_ratio'], 
                                     result['lowercase_ratio'])
    result['quote_consistency'] = max(result['single_quote_ratio'],
                                    result['double_quote_ratio'],
                                    result['backtick_quote_ratio'])
    result['spacing_consistency'] = max(result['spacing_with_space_ratio'],
                                      1 - result['spacing_with_space_ratio'])
    
    # 计算总体符号使用
    result['total_symbols'] = total_quotes + total_functions + total_equality
    return result
def get_symbol(file_path):
    code_analyzer = LexicalAanalysis()
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    result = code_analyzer.analyze_code_by_lines(lines,is_list=True)
    result_2 = calculate_ratios(result)
    # 将结果转换为列表,目前有23个属性
    result_list = [
        result_2["total_names"],
        result_2['camel_case_ratio'],
        result_2['snake_case_ratio'],
        result_2['constant_case_ratio'],
        result_2['lowercase_ratio'],
        result_2['other_ratio'],
        result_2['semicolon_ratio'],
        result_2['total_quotes'],
        result_2['single_quote_ratio'],
        result_2['double_quote_ratio'],
        result_2['backtick_quote_ratio'],
        result_2['total_functions'],
        result_2['arrow_func_ratio'],
        result_2['regular_func_ratio'],
        result_2['total_equality_ops'],
        result_2['strict_equality_ratio'],
        result_2['non_strict_equality_ratio'],
        result_2['total_spacing'],
        result_2['spacing_with_space_ratio'],
        result_2['naming_consistency'],
        result_2['quote_consistency'],
        result_2['spacing_consistency'],
        result_2['total_symbols']
    ]
    #将list转换为ndarray 
    return np.array(result_list)
    
if __name__ == "__main__":
    import os
    import logging
    def setup_logger():
    # 清空之前的日志文件（如果存在）
        log_file = "get_vec_js-test.log"
        if os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write('')  # 清空文件内容
        
        # 创建logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.handlers = []  # 清除之前的handler
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    logger = setup_logger()
    # 测试代码
    #file_path = "Forsee/dataset/js-test/non-vul/3n3m1%40ukr.net__coderaiser%2Fcloudcmd__client.js__6c263c8c02ccf6a7fd9c51b1d8c303594f9ee6ac.js"
    #对js-test中所有的js文件进行处理，查看有没有异常情况
    root_dir = "Forsee/dataset/js-test"

    for author_name in os.listdir(root_dir):
        author_path = os.path.join(root_dir, author_name)
        # 确保是目录且不是隐藏文件
        if os.path.isdir(author_path) and not author_name.startswith('.'):
            # 遍历作者目录下的所有文件
            for file_name in os.listdir(author_path):
                # 过滤出JS文件
                if file_name.endswith('.js'):
                    file_path = os.path.join(author_path, file_name)
                    
                    # 获取并打印向量
                    try:
                        vector = get_symbol(file_path)
                        logger.info(f"Author: {author_name}, File: {file_name}")
                        logger.info(vector)
                    except Exception as e:
                        logger.error(f"Error processing file {file_name}: {e}")
                        continue
