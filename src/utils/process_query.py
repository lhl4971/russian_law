import re
from utils.abbreviation_map import ABBREVIATION_MAP

def process_text_with_case_preservation(text):
    """
    处理文本中的俄语缩写，进行不区分大小写的匹配，并保留原始文本的大小写。
    同时处理缩写前后没有空格的混合文本情况。
    """

    # 1. 构建所有缩写的正则表达式，不区分大小写
    regex_abbreviations = '|'.join(re.escape(abbr) for abbr in ABBREVIATION_MAP.keys())
    
    # 2. 定义非西里尔字母的字符集作为单词边界
    non_cyrillic_before = r'(?<![а-яА-Я])'  # 前面不是西里尔
    non_cyrillic_after  = r'(?![а-яА-Я])'   # 后面不是西里尔
    
    # 3. 组合最终的正则表达式    
    regex_pattern = re.compile(
        f'{non_cyrillic_before}({regex_abbreviations}){non_cyrillic_after}',
        re.IGNORECASE
    )

    def replacement_callback(match):
        """
        回调函数，根据匹配到的原始文本（match.group(1)）进行替换。
        """
        # 获取匹配到的原始缩写，例如 'РВП'
        original_abbr = match.group(1)
        # 将其转换为小写，从映射中查找全称
        full_name = ABBREVIATION_MAP.get(original_abbr.lower())
        
        # 我们可以根据需要决定如何格式化替换文本
        # 比如：返回"全称 (缩写)"
        return f'\"{full_name} ({original_abbr})\"'
        # 或者只返回全称
        # return full_name

    # 使用 re.sub() 和回调函数进行替换
    processed_query = regex_pattern.sub(replacement_callback, text)

    return processed_query


def preprocess_data(input_data):
    """
    处理 user_query 字段并保留其他字段。
    """
    if 'user_query' in input_data:
        input_data['user_query'] = process_text_with_case_preservation(input_data['user_query'])
    return input_data
