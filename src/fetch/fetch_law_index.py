import os
import re
import requests
import json
from bs4 import BeautifulSoup
import roman

def parse_index(text):
    """
    从 'Глава VI.1.' / 'Статья VII.3-1.' / 'Статья 16.3-1.' 中提取编号并转换为阿拉伯数字
    """
    # 提取 "Глава " 或 "Статья " 后面的部分
    m = re.search(r'^(Глава|Статья)\s+(.+)$', text, re.IGNORECASE)
    if not m:
        return None
    idx_part = m.group(2).strip()

    # 去掉末尾多余的点号
    idx_part = re.sub(r'\.+$', '', idx_part)

    # 按非数字分隔符切分（可能有罗马数字、小数点、横杠）
    def convert_token(token):
        token = token.strip()
        if not token:
            return ""
        # 如果是罗马数字，转成阿拉伯数字
        try:
            return str(roman.fromRoman(token.upper()))
        except roman.InvalidRomanNumeralError:
            return token  # 不是罗马数字就原样返回

    # 先从 idx_part 提取最前面的编号 token（可能是罗马数字或阿拉伯数字，允许 . 或 - 分隔）
    m = re.match(r'^\s*([IVXLCDM]+|\d+)(?:[.\-](?:\d+|[IVXLCDM]+))*', idx_part, re.IGNORECASE)
    if not m:
        # 无法识别编号时返回 None（或按需返回空字符串）
        return None

    num_token = m.group(0).strip()

    # 保留点号和横杠结构并对每个片段进行转换（罗马 -> 阿拉伯，非罗马保持原样）
    parts = re.split(r'([.-])', num_token)  # 只对编号部分分割
    converted_parts = []
    for part in parts:
        if part in ['.', '-']:
            converted_parts.append(part)
        else:
            converted_parts.append(convert_token(part))

    return ''.join(converted_parts)

def fetch_law_index(path, output_dir):
    BASE_URL = "https://www.consultant.ru"
    PAGE_URL = BASE_URL + path

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    resp = requests.get(PAGE_URL, headers=headers)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")

    result = {
        "law_index": None,
        "law_date": None,
        "law_title": None,
        "chapters": []
    }

    # 1. 文档抬头
    doc_title_tag = soup.find("h1") or soup.find("div", class_="b-document-title")
    if doc_title_tag:
        result["law_title"] = doc_title_tag.get_text(strip=True)
        
        # 编号匹配：查找"-ФЗ"之前的数字
        num_match = re.search(r'\s*(\d+)\s*-\s*ФЗ', result["law_title"])
        result["law_index"] = num_match.group(1) if num_match else None

        # 日期匹配：查找"от"之后的 дд.мм.гггг
        date_match = re.search(r'от\s*(\d{2}\.\d{2}\.\d{4})', result["law_title"])
        result["law_date"] = date_match.group(1) if date_match else None

    # 2. 解析章节与条款
    seen_urls = set()
    current_chapter = None

    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"]

        if not text:
            continue

        # 过滤掉已失效法律
        if "Утратила силу" in text:
            continue

        # 构造完整 URL
        full_url = href if href.startswith("http") else BASE_URL + href

        # 去重
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        if text.startswith("Глава"):
            current_chapter = {
                "chapter_index": parse_index(text),
                "chapter_title": text,
                "articles": []
            }
            result["chapters"].append(current_chapter)

        elif text.startswith("Статья"):
            if current_chapter is None:
                current_chapter = {
                    "chapter_index": None,
                    "chapter_title": "",
                    "articles": []
                }
                result["chapters"].append(current_chapter)

            current_chapter["articles"].append({
                "article_index": parse_index(text),
                "article_title": text,
                "url": full_url
            })

    # 3. 保存 JSON
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "law_index.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已提取并保存 {PAGE_URL} 至 {output_file}")
    return output_file


if __name__ == "__main__":
    # 示例：抓取法律目录
    fetch_law_index("/document/cons_doc_LAW_37868/", "./output")