import numpy as np

class Testconfig:
    def __init__(self,symbol_max_value=100,symbol_vector_dim = 256):
        self.symbol_max_value = symbol_max_value
        self.symbol_vector_dim = symbol_vector_dim

def symbol_feature(processed_layouts):
    """
    处理符号特征，将多维特征向量转换为固定长度的向量
    
    Args:
        group_symbols: 符号特征列表，每个元素是形状为 (n_vectors, 14) 的特征向量列表
    
    Returns:
        处理后的符号特征列表
    """
    # 方法一 直接把对应列加起来
    return np.sum(processed_layouts, axis=0)
def proc_symbol_feature(group_symbols,config: Testconfig):
    """
    处理符号特征组，使其具有统一的向量长度和值范围。

    参数:
    symbol_feature_group (list of lists): 符号特征组。
                                          每个内部列表代表一种符号在各行的分布情况。
                                          例如: [[0,1,0,...], [2,0,1,...], ...]
                                          这对应学长代码中的 group_codewaves。
    config (SymbolProcessorConfig): 包含处理参数的配置对象。

    返回:
    list of np.ndarray: 处理后的符号特征向量列表。
    """
    processed_layouts = [None] * len(group_symbols)

    for i in range(len(processed_layouts)):
        # symbol_counts_for_one_type 指的是某一种符号在所有行中的出现次数列表
        # 这就相当于学长代码中的一个 codewave
        symbol_counts_for_one_type = group_symbols[i]
        
        # 转换为 NumPy 数组
        v = np.array(symbol_counts_for_one_type)
        
        # 1. 截断计数值 (Clipping)
        # 将符号的计数值限制在 [0, config.symbol_max_count] 范围内
        # 注意：你的符号计数本身非负，所以下限可以是0或者不设。
        # 如果你的符号计数永远不会超过某个合理值，或者你不想做这种截断，可以注释掉下面这行。
        v = np.clip(v, 0, config.symbol_max_value)

        # 2. 调整向量维度 (Padding/Truncating)
        current_len = len(v)
        target_len = config.symbol_vector_dim

        if current_len > target_len:
            # 如果当前特征向量比目标维度长，则截断
            v = v[:target_len]
        elif current_len < target_len:
            # 如果当前特征向量比目标维度短，则用0填充
            # np.pad 的第二个参数是一个元组 (before, after)，表示在数组前后填充的数量
            v = np.pad(v, (0, target_len - current_len), 'constant', constant_values=0)
            
        processed_layouts[i] = v
        
    return processed_layouts

if __name__=='__main__':
    config = Testconfig(symbol_max_value=100,symbol_vector_dim=20)
    group_symbols = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
       ])
    processed_sym_feature = proc_symbol_feature(group_symbols, config)
    result = symbol_feature(processed_sym_feature)
    print(result)