import re
import os
import sys
from collections import defaultdict


def simple_count_pattern(code,pattern):
    return code.count(pattern)

def count_with_boundary(code, pattern):
    """带单词边界检查的计数"""
    count = 0
    i = 0
    pattern_len = len(pattern)
    while i <= len(code) - pattern_len:
        pos = code.find(pattern, i)
        if pos == -1:
            break
        
        # 检查单词边界
        prev_ok = (pos == 0 or not code[pos-1].isalnum() and code[pos-1] != '_' and code[pos-1] != '$')
        next_ok = (pos+pattern_len >= len(code) or not code[pos+pattern_len].isalnum() and code[pos+pattern_len] != '_' and code[pos+pattern_len] != '$')
        
        if prev_ok and next_ok:
            count += 1
        
        i = pos + 1
        
    return count  

def count_strings_safe(code, quote_char):
    """更安全的字符串计数，处理转义引号"""
    count = 0
    i = 0
    in_string = False
    escape = False
    
    while i < len(code):
        if code[i] == '\\' and not escape:
            escape = True
        elif code[i] == quote_char and not escape:
            in_string = not in_string
            if not in_string:  # 结束一个字符串
                count += 1
        else:
            escape = False
        i += 1
    
    return count
def analyze_js_lexical_style_optimized(code):
    """分析JavaScript/TypeScript代码的词法风格特征"""
    
    results = {
        # 命名风格统计
        "naming": {
            "camel_case": 0,       # 驼峰式 (camelCase)
            "snake_case": 0,       # 蛇形式 (snake_case)
            "constant_case": 0,     # 全大写常量 (MAX_SIZE)
            "lowercase": 0,         # 全小写 (res, err, a)
            "other": 0              # 不符合上述任何风格的命名
        },
        # 其他结构保持不变
        "symbols": {
            "with_semicolons": 0,    # 使用分号结束
            "without_semicolons": 0, # 不使用分号结束
            "quotes": {
                "single": 0,      # 单引号 ('string')
                "double": 0,      # 双引号 ("string")
                "backtick": 0     # 反引号 (`template`)
            },
            "function_style": {
                "arrow": 0,       # 箭头函数 (=>)
                "regular": 0      # 常规函数 (function)
            }
        },
        "operators": {
            "equality": {
                "strict": 0,      # 严格相等 (===, !==)
                "non_strict": 0   # 非严格相等 (==, !=)
            },
            "spacing": {
                "with_space": 0,  # 运算符两侧有空格
                "without_space": 0 # 运算符两侧无空格
            }
        }
    }
    # 1. 命名风格分析
    # 提取变量和函数名
    # 改进模式
    var_pattern = re.compile(r'\b(?:var|let|const)\s+([a-zA-Z_$][\w$]*)\b')
    func_pattern = re.compile(r'\bfunction\s+([a-zA-Z_$][\w$]*)\b')
    # 限制括号内容长度，防止灾难性回溯
    method_pattern = re.compile(r'(?:^|\s)([a-zA-Z_$][\w$]*)\s*\([^){]{0,500}\)\s*{')
    # 限制参数长度
    arrow_assign_pattern = re.compile(r'\b(?:var|let|const)\s+([a-zA-Z_$][\w$]*)\s*=\s*(?:\([^){]{0,500}\)|[a-zA-Z_$][\w$]*)\s*=>')
    
    # 收集所有标识符
    identifiers = []
    for pattern in [var_pattern, func_pattern, method_pattern, arrow_assign_pattern]:
        for match in pattern.finditer(code):
            identifiers.append(match.group(1))
    
    # 定义判断命名风格的函数，使用更高效的方法
    def is_camel_case(name):
        # 首字母小写，包含至少一个大写字母，没有下划线
        return name[0].islower() and any(c.isupper() for c in name) and '_' not in name
    
    def is_snake_case(name):
        # 全部小写，包含下划线
        return name.islower() and '_' in name
    
    def is_constant_case(name):
        # 全部大写，可能包含下划线
        return name.isupper() and (len(name) <= 2 or '_' in name)
    
    def is_lowercase(name):
        # 全部小写，没有下划线
        return name.islower() and '_' not in name
    
    # 分析每个标识符的命名风格
    for id in identifiers:
        if is_camel_case(id):
            results["naming"]["camel_case"] += 1
        elif is_snake_case(id):
            results["naming"]["snake_case"] += 1
        elif is_constant_case(id):
            results["naming"]["constant_case"] += 1
        elif is_lowercase(id):
            results["naming"]["lowercase"] += 1
        else:
            results["naming"]["other"] += 1
    
    # 以下代码保持不变
    # 2. 分号使用分析
    code_lines = [line.strip() for line in code.split('\n') if line.strip()]
    
    for line in code_lines:
        # 忽略注释行和块结束行
        if (line.startswith('//') or line.startswith('/*') or line.endswith('*/') or 
            line.endswith('{') or line.endswith('}')):
            continue
        
        if line.endswith(';'):
            results["symbols"]["with_semicolons"] += 1
        else:
            results["symbols"]["without_semicolons"] += 1
    
    # 3. 字符串引号使用分析 - 使用更安全的计数
    results["symbols"]["quotes"]["single"] = count_strings_safe(code, "'")
    results["symbols"]["quotes"]["double"] = count_strings_safe(code, '"')
    results["symbols"]["quotes"]["backtick"] = count_strings_safe(code, '`')
    
    # 4. 函数风格分析
    # 箭头函数 - 使用简单计数
    results["symbols"]["function_style"]["arrow"] = simple_count_pattern(code, '=>')
    
    # 常规函数 - 使用带边界检查的计数
    results["symbols"]["function_style"]["regular"] = count_with_boundary(code, 'function')
    
    # 5. 相等运算符分析
    # 严格相等 (===, !==)
    results["operators"]["equality"]["strict"] = simple_count_pattern(code, '===') + simple_count_pattern(code, '!==')
    
    # 非严格相等 (==, !=) - 需要减去严格相等的出现次数
    eq_double = simple_count_pattern(code, '==') - simple_count_pattern(code, '===')
    neq_double = simple_count_pattern(code, '!=') - simple_count_pattern(code, '!==')
    results["operators"]["equality"]["non_strict"] = max(0, eq_double) + max(0, neq_double)
    
    # 6. 操作符周围空格使用分析
    # 常见二元操作符
    binary_ops = r'[=\+\-\*\/\%\&\|\^]'
    
    # 两侧都有空格的操作符
    with_space = re.findall(r'\s' + binary_ops + r'\s', code)
    results["operators"]["spacing"]["with_space"] = len(with_space)
    
    # 至少一侧没有空格的操作符
    without_space = re.findall(r'(?:[^\s]' + binary_ops + r')|(?:' + binary_ops + r'[^\s])', code)
    # 避免重复计算
    no_space_count = len(without_space) - len(with_space)
    if no_space_count < 0:
        no_space_count = 0
    results["operators"]["spacing"]["without_space"] = no_space_count
    
    return results

def analyze_js_lexical_style_with_chunks(code, chunk_size=100*1024):
    """分块处理长代码并合并结果"""
    # 检查代码长度
    if len(code) <= chunk_size:
        # 代码不长，直接分析
        return analyze_js_lexical_style_optimized(code)
    
    # 初始化合并结果
    merged_result = {
        "naming": {
            "camel_case": 0,
            "snake_case": 0,
            "constant_case": 0,
            "lowercase": 0,
            "other": 0
        },
        "symbols": {
            "with_semicolons": 0,
            "without_semicolons": 0,
            "quotes": {
                "single": 0,
                "double": 0,
                "backtick": 0
            },
            "function_style": {
                "arrow": 0,
                "regular": 0
            }
        },
        "operators": {
            "equality": {
                "strict": 0,
                "non_strict": 0
            },
            "spacing": {
                "with_space": 0,
                "without_space": 0
            }
        }
    }
    
    # 分块处理
    chunks = []
    # 按行分块，避免切割单行代码
    lines = code.split('\n')
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        if current_size + line_size > chunk_size and current_chunk:
            # 当前块已满，添加到块列表
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            # 添加行到当前块
            current_chunk.append(line)
            current_size += line_size
    
    # 添加最后一个块
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    # 处理每个块并合并结果
    for i, chunk in enumerate(chunks):
        print(f"处理代码块 {i+1}/{len(chunks)}，大小: {len(chunk)/1024:.1f}KB")
        chunk_result = analyze_js_lexical_style_optimized(chunk)
        
        # 合并结果
        merge_analysis_results(merged_result, chunk_result)
    
    return merged_result

def merge_analysis_results(target, source):
    """合并两个分析结果"""
    if not source:
        return
    
    # 合并命名风格结果
    for style, count in source["naming"].items():
        target["naming"][style] += count
    
    # 合并符号使用结果
    target["symbols"]["with_semicolons"] += source["symbols"]["with_semicolons"]
    target["symbols"]["without_semicolons"] += source["symbols"]["without_semicolons"]
    
    for quote_type, count in source["symbols"]["quotes"].items():
        target["symbols"]["quotes"][quote_type] += count
    
    for func_style, count in source["symbols"]["function_style"].items():
        target["symbols"]["function_style"][func_style] += count
    
    # 合并运算符使用结果
    for eq_type, count in source["operators"]["equality"].items():
        target["operators"]["equality"][eq_type] += count
    
    for space_style, count in source["operators"]["spacing"].items():
        target["operators"]["spacing"][space_style] += count

def analyze_js_lexical_style(code):
    """分析JavaScript/TypeScript代码的词法风格特征"""
    # 使用优化的分块处理版本
    return analyze_js_lexical_style_with_chunks(code)

def format_percentage(count, total):
    """计算并格式化百分比"""
    if total == 0:
        return "0.0%"
    return f"{(count/total*100):.1f}%"

def print_analysis_results(results):
    """打印词法分析结果"""
    output = []
    output.append("\n===== JavaScript/TypeScript 词法风格分析 =====\n")
    
    # 1. 命名风格
    naming = results["naming"]
    total_names = sum(naming.values())
    
    output.append("----- 命名风格 (Naming Conventions) -----")
    if total_names > 0:
        output.append(f"驼峰式 (camelCase): {naming['camel_case']} ({format_percentage(naming['camel_case'], total_names)})")
        output.append(f"蛇形式 (snake_case): {naming['snake_case']} ({format_percentage(naming['snake_case'], total_names)})")
        output.append(f"烤肉串式 (kebab-case): {naming['kebab_case']} ({format_percentage(naming['kebab_case'], total_names)})")
        output.append(f"常量式 (CONSTANT_CASE): {naming['constant_case']} ({format_percentage(naming['constant_case'], total_names)})")
        output.append(f"全小写 (lowercase): {naming['lowercase']} ({format_percentage(naming['lowercase'], total_names)})")
    else:
        output.append("未找到足够的标识符进行分析")
    
    # 2. 符号使用
    symbols = results["symbols"]
    
    # 分号使用
    total_semicolons = symbols["with_semicolons"] + symbols["without_semicolons"]
    output.append("\n----- 分号使用 (Semicolon Usage) -----")
    if total_semicolons > 0:
        output.append(f"使用分号: {symbols['with_semicolons']} ({format_percentage(symbols['with_semicolons'], total_semicolons)})")
        output.append(f"不使用分号: {symbols['without_semicolons']} ({format_percentage(symbols['without_semicolons'], total_semicolons)})")
    else:
        output.append("未找到足够的语句进行分析")
    
    # 引号使用
    quotes = symbols["quotes"]
    total_quotes = quotes["single"] + quotes["double"] + quotes["backtick"]
    output.append("\n----- 引号使用 (Quote Usage) -----")
    if total_quotes > 0:
        output.append(f"单引号 ('): {quotes['single']} ({format_percentage(quotes['single'], total_quotes)})")
        output.append(f"双引号 (\"): {quotes['double']} ({format_percentage(quotes['double'], total_quotes)})")
        output.append(f"反引号 (`): {quotes['backtick']} ({format_percentage(quotes['backtick'], total_quotes)})")
    else:
        output.append("未找到字符串字面量进行分析")
    
    # 函数风格
    function_style = symbols["function_style"]
    total_functions = function_style["arrow"] + function_style["regular"]
    output.append("\n----- 函数声明风格 (Function Style) -----")
    if total_functions > 0:
        output.append(f"箭头函数 (=>): {function_style['arrow']} ({format_percentage(function_style['arrow'], total_functions)})")
        output.append(f"常规函数 (function): {function_style['regular']} ({format_percentage(function_style['regular'], total_functions)})")
    else:
        output.append("未找到函数声明进行分析")
    
    # 3. 运算符使用
    operators = results["operators"]
    
    # 相等运算符
    equality = operators["equality"]
    total_equality = equality["strict"] + equality["non_strict"]
    output.append("\n----- 相等运算符使用 (Equality Operators) -----")
    if total_equality > 0:
        output.append(f"严格相等 (===, !==): {equality['strict']} ({format_percentage(equality['strict'], total_equality)})")
        output.append(f"非严格相等 (==, !=): {equality['non_strict']} ({format_percentage(equality['non_strict'], total_equality)})")
    else:
        output.append("未找到相等运算符进行分析")
    
    # 运算符空格
    spacing = operators["spacing"]
    total_spacing = spacing["with_space"] + spacing["without_space"]
    output.append("\n----- 运算符空格使用 (Operator Spacing) -----")
    if total_spacing > 0:
        output.append(f"运算符两侧有空格: {spacing['with_space']} ({format_percentage(spacing['with_space'], total_spacing)})")
        output.append(f"运算符两侧无空格: {spacing['without_space']} ({format_percentage(spacing['without_space'], total_spacing)})")
    else:
        output.append("未找到足够的运算符进行分析")
    
    return "\n".join(output)

def analyze_code(code_snippet):
    """分析代码片段的词法风格并返回分析结果字符串"""
    results = analyze_js_lexical_style(code_snippet)
    return print_analysis_results(results)

def analyze_code_snippet(code_snippet):
    """分析代码片段的词法风格"""
    results = analyze_js_lexical_style(code_snippet)
    print("\n分析代码片段")
    analysis_output = print_analysis_results(results)
    print(analysis_output)
    return results

def analyze_file(file_path):
    """分析文件的词法风格"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        results = analyze_js_lexical_style(code)
        print(f"\n分析文件: {os.path.basename(file_path)}")
        analysis_output = print_analysis_results(results)
        print(analysis_output)
        return results
    except Exception as e:
        print(f"分析文件 {file_path} 时出错: {str(e)}")
        return None

def analyze_contributors_style(vulnerability_data, output_dir="./contributor_profiles"):
    """分析漏洞数据中各个贡献者的词法风格特征"""
    contributor_profiles = defaultdict(lambda: {
        "naming": defaultdict(int),
        "symbols": {
            "semicolons": defaultdict(int),
            "quotes": defaultdict(int),
            "functions": defaultdict(int)
        },
        "operators": {
            "equality": defaultdict(int),
            "spacing": defaultdict(int)
        },
        "code_samples": 0
    })
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    for vuln in vulnerability_data:
        if "vulnerability_details" not in vuln:
            continue
            
        for detail in vuln["vulnerability_details"]:
            if "code" not in detail or not detail["code"]:
                continue
                
            code_snippet = detail["code"]
            style_results = analyze_js_lexical_style(code_snippet)
            
            # 获取贡献者列表
            contributors = detail.get("matched_contributors", [])
            if not contributors:
                continue
                
            # 更新每个贡献者的词法风格统计
            for contributor in contributors:
                profile = contributor_profiles[contributor]
                profile["code_samples"] += 1
                
                # 更新命名风格
                for style, count in style_results["naming"].items():
                    profile["naming"][style] += count
                
                # 更新符号使用
                profile["symbols"]["semicolons"]["with"] += style_results["symbols"]["with_semicolons"]
                profile["symbols"]["semicolons"]["without"] += style_results["symbols"]["without_semicolons"]
                
                for quote_style, count in style_results["symbols"]["quotes"].items():
                    profile["symbols"]["quotes"][quote_style] += count
                    
                for func_style, count in style_results["symbols"]["function_style"].items():
                    profile["symbols"]["functions"][func_style] += count
                
                # 更新运算符使用
                for eq_style, count in style_results["operators"]["equality"].items():
                    profile["operators"]["equality"][eq_style] += count
                    
                for space_style, count in style_results["operators"]["spacing"].items():
                    profile["operators"]["spacing"][space_style] += count
    
    # 输出结果到文件
    for contributor, profile in contributor_profiles.items():
        output_file = os.path.join(output_dir, f"{contributor.replace('/', '_')}_profile.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"词法风格分析: {contributor}\n")
            f.write(f"分析代码片段数: {profile['code_samples']}\n\n")
            
            # 写入命名风格
            f.write("----- 命名风格 -----\n")
            total_names = sum(profile["naming"].values())
            if total_names > 0:
                for style, count in profile["naming"].items():
                    f.write(f"{style}: {count} ({count/total_names*100:.1f}%)\n")
            
            # 写入符号使用
            f.write("\n----- 符号使用 -----\n")
            
            # 分号
            total_semicolons = profile["symbols"]["semicolons"]["with"] + profile["symbols"]["semicolons"]["without"]
            if total_semicolons > 0:
                f.write(f"使用分号: {profile['symbols']['semicolons']['with']} " 
                       f"({profile['symbols']['semicolons']['with']/total_semicolons*100:.1f}%)\n")
                       
            # 引号类型
            total_quotes = sum(profile["symbols"]["quotes"].values())
            if total_quotes > 0:
                for quote_style, count in profile["symbols"]["quotes"].items():
                    f.write(f"{quote_style} 引号: {count} ({count/total_quotes*100:.1f}%)\n")
            
            # 函数风格
            total_funcs = sum(profile["symbols"]["functions"].values())
            if total_funcs > 0:
                for func_style, count in profile["symbols"]["functions"].items():
                    f.write(f"{func_style} 函数: {count} ({count/total_funcs*100:.1f}%)\n")
            
            # 运算符使用
            f.write("\n----- 运算符使用 -----\n")
            
            # 相等运算符
            total_eq = sum(profile["operators"]["equality"].values())
            if total_eq > 0:
                for eq_style, count in profile["operators"]["equality"].items():
                    f.write(f"{eq_style} 相等: {count} ({count/total_eq*100:.1f}%)\n")
            
            # 空格使用
            total_spacing = sum(profile["operators"]["spacing"].values())
            if total_spacing > 0:
                for space_style, count in profile["operators"]["spacing"].items():
                    f.write(f"{space_style}: {count} ({count/total_spacing*100:.1f}%)\n")
    
    return contributor_profiles

def main():
    if len(sys.argv) < 2:
        print("用法: python lexical_test.py <文件路径或代码片段>")
        return
    
    input_arg = sys.argv[1]
    
    if os.path.exists(input_arg):
        # 分析文件
        analyze_file(input_arg)
    else:
        # 分析提供的代码片段
        analyze_code_snippet(input_arg)

if __name__ == "__main__":
    main()