'''
从郑炎的数据集里进行不同粒度的词法分析
commit
'''
from .lexical_imp import analyze_js_lexical_style_optimized
from .lexical_imp import analyze_js_lexical_style_with_chunks
import logging
import re
class LexicalAanalysis:
    def __init__(self):
        # 初始化日志
        self.logging = logging
        self.logging.basicConfig(level=logging.INFO, 
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def get_additions_code(self, add_code):
        """获取patch中新增的代码行列表"""
        additions_code = add_code.split('\n')
        return additions_code  # 返回行列表而不是合并字符串
    
    def analyze_code_line(self, code):
        '''
        分析该行代码，若该行代码长度过长，则每100kb分割一次进行分析
        '''
        code_length = len(code)
        if code_length > 102400:  # 超过100KB的代码
            self.logging.warning(f"代码长度为 {code_length/1024:.2f}KB，可能需要较长处理时间")
            if code_length > 1024*1024:  # 超过1MB的代码直接跳过
                self.logging.error(f"代码过长 ({code_length/1024/1024:.2f}MB)，跳过分析")
                return None
            line_result = analyze_js_lexical_style_with_chunks(code)  # 分块分析
        else:
            line_result = analyze_js_lexical_style_optimized(code)
        return line_result

    def is_minified_code(self, code):
        """检测代码是否为混淆/压缩代码"""
        # 方法1：检查平均行长度
        lines = code.split('\n')
        if lines:
            avg_line_length = sum(len(line) for line in lines) / len(lines)
            if avg_line_length > 100:  # 设置合理阈值
                return True
        
        # 方法2：检查典型混淆模式
        minified_patterns = [
            r'!function\(', 
            r'[\[{]function\([\w,]*\)',
            r'eval\(',
            r'new Function\('
        ]
        for pattern in minified_patterns:
            if re.search(pattern, code):
                return True
        
        # 方法3：单字母变量密度检测
        single_var_pattern = r'[\s{;(][a-z]\s*[=:,)]'
        matches = re.findall(single_var_pattern, code)
        if len(matches) > 10:  # 设置合理阈值
            return True
        
        return False
    def analyze_code_by_lines(self, code ,is_list = False):
        """
        对整段代码按行进行分析并合并结果,如果传入的已经是每行代码构成的列表，则直接进行分析
        :param code: 代码字符串或列表
        :param is_list: 是否已经是列表，默认为False
        """
        # 检查代码是否为混淆/压缩代码
        # if self.is_minified_code(code):
        #     self.logging.warning("检测到混淆/压缩代码，可能会影响分析结果")
        #     return None
        # 分割代码行
        if not is_list:
            code_lines = self.get_additions_code(code)
        else:
            code_lines = code
        
        # 初始化结果
        # 初始化一个空结果结构而不是None
        result = {
            "naming": {
                "camel_case": 0, "snake_case": 0,
                "constant_case": 0, "lowercase": 0, "other": 0
            },
            "symbols": {
                "with_semicolons": 0, "without_semicolons": 0,
                "quotes": {"single": 0, "double": 0, "backtick": 0},
                "function_style": {"arrow": 0, "regular": 0}
            },
            "operators": {
                "equality": {"strict": 0, "non_strict": 0},
                "spacing": {"with_space": 0, "without_space": 0}
            }
        }
        
        # 如果代码为空，直接返回初始结果
        if code_lines is None or len(code_lines) == 0:
            return result
        
        for i, line in enumerate(code_lines):
            if not line.strip():  # 跳过空行
                continue
                
            # self.logging.info(f"分析第 {i+1}/{len(code_lines)} 行代码")
            
            # 分析单行代码
            line_result = self.analyze_code_line(line)
            
            if line_result:
                if result is None:
                    # 第一个有效结果直接赋值
                    result = line_result
                else:
                    # 合并结果，复用现有的合并函数
                    self.merge_chunk_results(result, line_result)
        
        return result
    
    def merge_chunk_results(self, target, source):
        """过长代码块进行分块，现在合并分块分析结果"""
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


if __name__ == "__main__":
    #code = "var test = require('tap').test\nvar common = require('./common')\nvar serve = common.serve\n\ntest('Basic cli operation', function (t) {\n  serve([], function (req) {\n\n    req('/st.js', function (er, res, body) {\n      t.ifError(er) &&\n      t.equal(res.statusCode, 200) &&\n      t.equal(body.toString(), common.stExpect)\n    })\n\n  }, function (er, stdout, stderr) {\n    t.ifError(er)\n    t.match(stdout, /^listening at http:\\/\\/(\\[::\\]|0\\.0\\.0\\.0):[0-9]+\\n$/)\n    t.equal(stderr, '')\n    t.end()\n  })\n})\n\ntest('Listening on localhost only', function (t) {\n  serve([\"--localhost\"], function (req) {\n\n    req('/st.js', function (er, res, body) {\n      t.ifError(er) &&\n      t.equal(res.statusCode, 200) &&\n      t.equal(body.toString(), common.stExpect)\n    })\n\n  }, function (er, stdout, stderr) {\n    t.ifError(er)\n    t.match(stdout, /^listening at http:\\/\\/localhost:[0-9]+\\n$/)\n    t.equal(stderr, '')\n    t.end()\n  })\n})"
    code = ''
    lexical_analysis = LexicalAanalysis()
    line_by_line_result = lexical_analysis.analyze_code_by_lines(code)
    print(line_by_line_result)