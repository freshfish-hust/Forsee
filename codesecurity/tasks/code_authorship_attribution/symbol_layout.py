'''
类似统计符号的布局特征：对于每一个符号，统计他们在代码中每一行的出现的次数！
'''
import re
import numpy as np
class Symbol_Layout_Extractor:
    def __init__(self):
        pass
    def count_strings_safe(self,code, quote_char):
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
    def count_with_boundary(self,code, pattern):
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
    def simple_count_pattern(self,code,pattern):
        return code.count(pattern)
    def extract_cwe_sensitive_features_per_line(self,line_of_code):
        """
        Extracts counts of CWE-sensitive keywords and API patterns from a single line of JavaScript/TypeScript code.

        Args:
            line_of_code (str): A string representing a single line of code.

        Returns:
            dict: A dictionary where keys are feature names and values are the counts
                of those features found in the line.
        """
        features = {
            "line_eval_like_keywords": 0,
            "line_dom_dangerous_methods": 0,
            "line_input_source_identifiers": 0,
            "line_fs_module_usage": 0,
            "line_child_process_usage": 0,
            "line_json_parse_usage": 0,
            "line_unescape_html_usage": 0,
            "line_regexp_constructor_usage": 0,
        }

        # 1. Eval-like keywords
        # Matches 'eval(' with word boundary to avoid matching 'evaluation('
        features["line_eval_like_keywords"] += len(re.findall(r'\beval\s*\(', line_of_code))
        # Matches 'Function(' (constructor) with word boundary
        features["line_eval_like_keywords"] += len(re.findall(r'\bFunction\s*\(', line_of_code))
        # Matches 'setTimeout("...' or setTimeout('...'
        features["line_eval_like_keywords"] += len(re.findall(r'\bsetTimeout\s*\(\s*["\']', line_of_code))
        # Matches 'setInterval("...' or setInterval('...'
        features["line_eval_like_keywords"] += len(re.findall(r'\bsetInterval\s*\(\s*["\']', line_of_code))

        # 2. DOM dangerous methods
        # Using string count for direct property access can be efficient and less prone to complex regex issues
        # if we assume these are not part of variable names commonly.
        features["line_dom_dangerous_methods"] += line_of_code.count('.innerHTML')
        features["line_dom_dangerous_methods"] += line_of_code.count('.outerHTML')
        features["line_dom_dangerous_methods"] += len(re.findall(r'\bdocument\.write\s*\(', line_of_code))
        features["line_dom_dangerous_methods"] += len(re.findall(r'\.insertAdjacentHTML\s*\(', line_of_code)) # .methodName(
        features["line_dom_dangerous_methods"] += len(re.findall(r'\.appendChild\s*\(', line_of_code)) # .methodName(
        # setAttribute with potentially dangerous attributes (event handlers, src, href, style)
        # Checking for string literals as the attribute name for common cases
        features["line_dom_dangerous_methods"] += len(re.findall(r'\.setAttribute\s*\(\s*["\'](?:on\w+|href|src|style|formaction|data|action)["\']', line_of_code, re.IGNORECASE))


        # 3. Input source identifiers
        # These are often property accesses or specific object paths
        features["line_input_source_identifiers"] += line_of_code.count('req.query') # Express.js common
        features["line_input_source_identifiers"] += line_of_code.count('req.params') # Express.js common
        features["line_input_source_identifiers"] += line_of_code.count('req.body') # Express.js common
        features["line_input_source_identifiers"] += line_of_code.count('location.search')
        features["line_input_source_identifiers"] += line_of_code.count('document.URL')
        features["line_input_source_identifiers"] += line_of_code.count('document.documentURI')
        features["line_input_source_identifiers"] += line_of_code.count('document.cookie')
        features["line_input_source_identifiers"] += line_of_code.count('window.location') # Catches .hash, .href etc.
        features["line_input_source_identifiers"] += len(re.findall(r'\blocalStorage\.getItem\s*\(', line_of_code))
        features["line_input_source_identifiers"] += len(re.findall(r'\bsessionStorage\.getItem\s*\(', line_of_code))
        features["line_input_source_identifiers"] += line_of_code.count('process.env.') # Node.js environment variables

        # 4. File System (fs) module usage (common in Node.js)
        # Matches fs.methodName( - basic catch-all for fs module methods
        features["line_fs_module_usage"] += len(re.findall(r'\bfs\.\w+\s*\(', line_of_code))
        # Specific fs methods for clarity, though the above regex covers them if 'fs.' is used.
        # features["line_fs_module_usage"] += len(re.findall(r'\bfs\.readFile\s*\(', line_of_code))
        # features["line_fs_module_usage"] += len(re.findall(r'\bfs\.writeFile\s*\(', line_of_code))
        # ... add more specific fs methods if granular counts are needed ...

        # 5. Child Process usage (common in Node.js)
        features["line_child_process_usage"] += len(re.findall(r'\bchild_process\.exec\s*\(', line_of_code))
        features["line_child_process_usage"] += len(re.findall(r'\bchild_process\.execSync\s*\(', line_of_code))
        features["line_child_process_usage"] += len(re.findall(r'\bchild_process\.spawn\s*\(', line_of_code))
        # Could also look for require('child_process') pattern if not directly using an imported 'child_process' variable.
        # For example: require\s*\(\s*['"]child_process['"]\s*\)\.\w+\s*\(

        # 6. JSON.parse usage
        features["line_json_parse_usage"] += len(re.findall(r'\bJSON\.parse\s*\(', line_of_code))

        # 7. Unescape HTML usage
        features["line_unescape_html_usage"] += len(re.findall(r'\bunescape\s*\(', line_of_code)) # Global unescape
        # For library-specific unescape, e.g., _.unescape() or someObj.unescape()
        features["line_unescape_html_usage"] += len(re.findall(r'\.\w*unescape\s*\(', line_of_code, re.IGNORECASE))


        # 8. RegExp constructor usage (potential for ReDoS if pattern is from user input)
        features["line_regexp_constructor_usage"] += len(re.findall(r'\bnew\s+RegExp\s*\(', line_of_code))

        return features
    def extract_operator_features_per_line(self,line_of_code: str) -> dict:
        """
        Counts various JavaScript/TypeScript operators on a single line of code.

        Args:
            line_of_code: A string representing a single line of code.

        Returns:
            A dictionary with counts for each defined operator category.
        """
        features = {
            "line_strict_equality": 0,
            "line_non_strict_equality": 0,
            "line_assignment_operators": 0,
            "line_logical_operators": 0,
            "line_comparison_operators": 0, # (excluding equality)
            "line_plus_operators": 0,       # (standalone '+')
            "line_minus_operators": 0,      # (standalone '-')
            "line_multiply_divide_modulo_operators": 0, # (standalone '*', '/', '%')
            "line_bitwise_operators": 0,
            "line_unary_operators": 0,      # (++ -- typeof void delete)
            "line_spread_rest_operators": 0, # (...)
            "line_instanceof_in_operators": 0, # (instanceof, in)
        }

        # 1.相等运算符分析
        # 严格相等 (===, !==)
        features["line_strict_equality"] = self.simple_count_pattern(line_of_code, '===') + self.simple_count_pattern(line_of_code, '!==')
        
        # 非严格相等 (==, !=) - 需要减去严格相等的出现次数
        eq_double = self.simple_count_pattern(line_of_code, '==') - self.simple_count_pattern(line_of_code, '===')-self.simple_count_pattern(line_of_code,'!==')
        neq_double = self.simple_count_pattern(line_of_code, '!=') - self.simple_count_pattern(line_of_code, '!==')
        features["line_non_strict_equality"] = max(0, eq_double) + max(0, neq_double)

        # --- Assignment Operators ---
        # Order can be important if some are substrings of others, or use specific regex.
        # We count specific multi-char assignment ops first.
        assign_ops_count = 0
        assign_ops_count += len(re.findall(r'\*\*\=', line_of_code)) # **=
        assign_ops_count += len(re.findall(r'\?\?\=', line_of_code)) # ??=
        assign_ops_count += len(re.findall(r'\+=', line_of_code))    # +=
        assign_ops_count += len(re.findall(r'-=', line_of_code))    # -=
        # *= (not part of **=)
        assign_ops_count += len(re.findall(r'(?<!\*)\*=(?!=)', line_of_code))
        assign_ops_count += len(re.findall(r'/=', line_of_code))    # /=
        assign_ops_count += len(re.findall(r'%=', line_of_code))    # %=
        assign_ops_count += len(re.findall(r'&=', line_of_code))    # &= (ensure not &&= if that existed)
        assign_ops_count += len(re.findall(r'\|=', line_of_code))   # |= (ensure not ||= if that existed)
        assign_ops_count += len(re.findall(r'\^=', line_of_code))   # ^=
        # Standalone = (not part of other ops like ==, ===, +=, etc.)
        assign_ops_count += len(re.findall(r'(?<![=+\-*/%&|^<>?!])=(?!=)', line_of_code))
        features["line_assignment_operators"] = assign_ops_count

        # --- Logical & Comparison (excluding equality from this category as per request) ---
        logical_ops_count = 0
        logical_ops_count += len(re.findall(r'&&', line_of_code))  # &&
        logical_ops_count += len(re.findall(r'\|\|', line_of_code)) # ||
        # ! (not part of != or !==)
        logical_ops_count += len(re.findall(r'(?<![=])!(?!=)', line_of_code))
        features["line_logical_operators"] = logical_ops_count

        comparison_ops_count = 0
        comparison_ops_count += len(re.findall(r'>=', line_of_code)) # >=
        comparison_ops_count += len(re.findall(r'<=', line_of_code)) # <=
        # > (not part of >= or >>, >>>)
        comparison_ops_count += len(re.findall(r'(?<![=<])>(?![>=])', line_of_code))
        # < (not part of <= or <<)
        comparison_ops_count += len(re.findall(r'(?<![=>])<(?![=<])', line_of_code))
        features["line_comparison_operators"] = comparison_ops_count

        # --- Arithmetic (standalone operators) ---
        # + (not part of ++ or +=)
        features["line_plus_operators"] = len(re.findall(r'(?<!\+)\+(?![+=])', line_of_code))
        # - (not part of -- or -=)
        features["line_minus_operators"] = len(re.findall(r'(?<!-)-(?![-=])', line_of_code))

        multiply_divide_modulo_count = 0
        # * (not part of ** or *=)
        multiply_divide_modulo_count += len(re.findall(r'(?<!\*)\*(?![*=])', line_of_code))
        # / (not part of //, /*, or /=)
        multiply_divide_modulo_count += len(re.findall(r'(?<!/)/(?![/*=])', line_of_code))
        # % (not part of %=)
        multiply_divide_modulo_count += len(re.findall(r'%(?!=)', line_of_code))
        features["line_multiply_divide_modulo_operators"] = multiply_divide_modulo_count

        # --- Bitwise Operators ---
        bitwise_ops_count = 0
        bitwise_ops_count += len(re.findall(r'>>>', line_of_code)) # >>> (zero-fill right shift)
        # << (left shift, not part of <<< which is not a JS op, and not confused with <=)
        bitwise_ops_count += len(re.findall(r'(?<!<)<<(?!=)', line_of_code))
        # >> (signed right shift, not part of >>>)
        bitwise_ops_count += len(re.findall(r'(?<![>])>>(?![>=])', line_of_code))
        # & (bitwise AND, not part of && or &=)
        bitwise_ops_count += len(re.findall(r'(?<!&)&(?![&=])', line_of_code))
        # | (bitwise OR, not part of || or |=)
        bitwise_ops_count += len(re.findall(r'(?<!\|)\|(?!\|\|=)', line_of_code)) # ensure not || or |=
        # ^ (bitwise XOR, not part of ^=)
        bitwise_ops_count += len(re.findall(r'\^(?!=)', line_of_code))
        bitwise_ops_count += len(re.findall(r'~', line_of_code))   # ~ (bitwise NOT)
        features["line_bitwise_operators"] = bitwise_ops_count

        # --- Unary & Other ---
        unary_ops_count = 0
        unary_ops_count += len(re.findall(r'\+\+', line_of_code)) # ++
        unary_ops_count += len(re.findall(r'--', line_of_code))  # --
        # Keyword unary operators
        unary_ops_count += len(re.findall(r'\btypeof\b', line_of_code))
        unary_ops_count += len(re.findall(r'\bvoid\b', line_of_code))
        unary_ops_count += len(re.findall(r'\bdelete\b', line_of_code))
        features["line_unary_operators"] = unary_ops_count

        features["line_spread_rest_operators"] = len(re.findall(r'\.\.\.', line_of_code)) # ...

        instanceof_in_ops_count = 0
        instanceof_in_ops_count += len(re.findall(r'\binstanceof\b', line_of_code))
        # 'in' operator (ensure it's standalone, not part of 'instanceof' or an identifier)
        instanceof_in_ops_count += len(re.findall(r'\bin\b', line_of_code))
        features["line_instanceof_in_operators"] = instanceof_in_ops_count

        return features
    def origin_feature_extractor(self, code: str) -> dict:
        '''
        其他通常特征的统计,主要包括命名风格，符号使用(分号结束，引号的使用，arrow)，
        '''
    
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
                "quotes": {
                    "single": 0,      # 单引号 ('string')
                    "double": 0,      # 双引号 ("string")
                    "backtick": 0     # 反引号 (`template`)
                },
                "function_style": {
                    "arrow": 0,       # 箭头函数 (=>)
                    "regular": 0      # 常规函数 (function)
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
        
        # 3. 字符串引号使用分析 - 使用更安全的计数
        results["symbols"]["quotes"]["single"] = self.count_strings_safe(code, "'")
        results["symbols"]["quotes"]["double"] = self.count_strings_safe(code, '"')
        results["symbols"]["quotes"]["backtick"] = self.count_strings_safe(code, '`')
        
        # 4. 函数风格分析
        # 箭头函数 - 使用简单计数
        results["symbols"]["function_style"]["arrow"] = self.simple_count_pattern(code, '=>')
        
        # 常规函数 - 使用带边界检查的计数
        results["symbols"]["function_style"]["regular"] = self.count_with_boundary(code, 'function')
        return results

class Symbol_Layout_feature:
    def __init__(self):
        self.extractor = Symbol_Layout_Extractor()
    #  @staticmethod
    def extarct_features_from_file(self,file_path):
        """
        从文件中读取JavaScript代码并提取特征布局
        
        Args:
            file_path (str): JavaScript文件路径
            
        Returns:
            dict: 包含各种特征布局的字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                js_code = f.read()
            feature_layout,total_lines =  self.get_code_features(js_code)
            feature_vec = self.get_feature_vec(feature_layout,total_lines)
            return feature_vec
        except Exception as e:
            print(f"Error reading or analyzing file {file_path}: {str(e)}")
            return None
        
    def get_code_features(self,js_code):
        '''
        返回所有特征的布局
        '''
        lines = js_code.split('\n')
        feature_layout={
            'CWE_sensitive_features':
                {
                "line_eval_like_keywords":  [0] * len(lines),
                "line_dom_dangerous_methods":  [0] * len(lines),
                "line_input_source_identifiers":  [0] * len(lines),
                "line_fs_module_usage":  [0] * len(lines),
                "line_child_process_usage":  [0] * len(lines),
                "line_json_parse_usage":  [0] * len(lines),
                "line_unescape_html_usage":  [0] * len(lines),
                "line_regexp_constructor_usage":  [0] * len(lines),
                },
                'Operator_features':
                {
                    "line_strict_equality":  [0] * len(lines),
                    "line_non_strict_equality":  [0] * len(lines),
                    "line_assignment_operators":  [0] * len(lines),
                    "line_logical_operators":  [0] * len(lines),
                    "line_comparison_operators":  [0] * len(lines), # (excluding equality)
                    "line_plus_operators":  [0] * len(lines),# (standalone '+')
                    "line_minus_operators":  [0] * len(lines),  # (standalone '-')
                    "line_multiply_divide_modulo_operators":  [0] * len(lines), # (standalone '*', '/', '%')
                    "line_bitwise_operators":  [0] * len(lines),
                    "line_unary_operators":  [0] * len(lines),      # (++ -- typeof void delete)
                    "line_spread_rest_operators": [0] * len(lines), # (...)
                    "line_instanceof_in_operators":  [0] * len(lines), # (instanceof, in)
                },
                'General_write_features':
                {
                    "naming": 
                    {
                    "camel_case": [0] * len(lines),      # 驼峰式 (camelCase)
                    "snake_case": [0] * len(lines),       # 蛇形式 (snake_case)
                    "constant_case": [0] * len(lines),    # 全大写常量 (MAX_SIZE)
                    "lowercase": [0] * len(lines),         # 全小写 (res, err, a)
                    "other": [0] * len(lines),             # 不符合上述任何风格的命名
                    },
                    "quotes": {
                        "single":[0] * len(lines),      # 单引号 ('string')
                        "double": [0] * len(lines),      # 双引号 ("string")
                        "backtick": [0] * len(lines),    # 反引号 (`template`)
                    },
                    "function_style": {
                        "arrow": [0] * len(lines),       # 箭头函数 (=>)
                        "regular": [0] * len(lines),    # 常规函数 (function)
                    }
                }
        }
        
        for i,line in enumerate(lines):
            if not line.strip():
                continue
            # 提取CWE敏感特征
            cwe_sensitive_features = self.extractor.extract_cwe_sensitive_features_per_line(line)
            for feature, count in cwe_sensitive_features.items():
                feature_layout['CWE_sensitive_features'][feature][i] = count
            # 提取操作符特征
            operator_features = self.extractor.extract_operator_features_per_line(line)
            for feature, count in operator_features.items():
                feature_layout['Operator_features'][feature][i] = count
            # 提取其他特征
            general_features = self.extractor.origin_feature_extractor(line)
            for feature, count in general_features['naming'].items():
                feature_layout['General_write_features']['naming'][feature][i] = count
            for feature, count in general_features['symbols']['quotes'].items():
                feature_layout['General_write_features']['quotes'][feature][i] = count
            for feature, count in general_features['symbols']['function_style'].items():
                feature_layout['General_write_features']['function_style'][feature][i] = count
        return feature_layout,len(lines)
    
    def get_feature_vec(self, feature_layout,total_lines):
        '''
        将统计的布局特征转化为向量
        
        Args:
            feature_layout (dict): 特征布局字典
            
        Returns:
            numpy.ndarray: 特征向量，形状为 (n_features,)
        '''
        
        feature_vec = []
        def compute_kernel_features(feature_layout):
            """计算核特征，捕捉特征间的交互作用"""
            kernel_features = []
            
            # 收集所有特征的向量表示
            all_feature_vectors = []
            feature_names = []
            
            def collect_vectors(feature_dict, prefix=""):
                for name, values in feature_dict.items():
                    if isinstance(values, list):
                        all_feature_vectors.append(np.array(values))
                        feature_names.append(f"{prefix}_{name}" if prefix else name)
                    elif isinstance(values, dict):
                        collect_vectors(values, f"{prefix}_{name}" if prefix else name)
            
            collect_vectors(feature_layout)
            
            #print("all_feature_vectors:", all_feature_vectors)
            
            return all_feature_vectors
            
        
        collect_features = compute_kernel_features(feature_layout)
        feature_vec.extend(collect_features)
        return np.array(feature_vec)
# --- Example Usage ---
if __name__ == '__main__':

#   # Example JavaScript code
    js_code = """
    function exampleFunction() {
        var x = 10;
        let y = 20;
        const MAX_SIZE = 100;
        if (x === y) {
            console.log('Equal');
        } else {
            console.log('Not Equal');
        }
        eval('console.log(x + y)');
        document.write('<h1>Hello World</h1>');
    }
    """
    
    # Create an instance of the Symbol_Layout_feature class
    extractor = Symbol_Layout_feature()
    
    # Extract features from the example code
    features,total_lines = extractor.get_code_features(js_code)
    
    kernel_vec = extractor.get_feature_vec(features,total_lines)
    print("Feature Vector:", kernel_vec)