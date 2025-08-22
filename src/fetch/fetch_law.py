import re
import json
import os
import requests
import datetime
from bs4 import BeautifulSoup
from fetch_law_index import fetch_law_index


def parse_article_document(lines, law_index, law_date, law_title, chapter_index, chapter_title, article_index, article_title):
    result = {
        "law_index": int(law_index), 
        "law_issue_year": int(datetime.datetime.strptime(law_date, "%d.%m.%Y").year),
        "law_title": law_title,
        "chapter_index": chapter_index,
        "chapter_title": chapter_title,
        "article_index": article_index,
        "article_title": article_title,
        "clauses": [],
        "unindexed": []
    }

    current_clause = None
    current_subclause = None

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        # 匹配二级条款 (如 1), a), б), 1.1), 1.1-1) 等)
        if re.match(r"^[0-9a-zA-Zа-яА-Я.\-]+\)", line) and current_clause:
            idx_match = re.match(r"^([0-9a-zA-Zа-яА-Я.\-]+)\)", line)
            sub_idx = idx_match.group(1) if idx_match else None
            current_subclause = {
                "subclause_index": sub_idx,
                "subclause_text": line.strip(),
                "unindexed": []
            }
            current_clause.setdefault("subclauses", []).append(current_subclause)
            continue

        # 匹配一级条款 (如 1., I., A., а., 1.1., 1.1-1. 等)
        if re.match(r"^[0-9a-zA-Zа-яА-Я.\-]+\.", line):
            idx_match = re.match(r"^([0-9a-zA-Zа-яА-Я.\-]+)\.", line)
            clause_idx = idx_match.group(1) if idx_match else None
            current_clause = {
                "clause_index": clause_idx,
                "clause_text": line.strip(),
                "subclauses": [],
                "unindexed": []
            }
            result["clauses"].append(current_clause)
            current_subclause = None
            continue

        # 普通段落
        if current_subclause:
            current_subclause["unindexed"].append(line)
        elif current_clause:
            current_clause["unindexed"].append(line)
        else:
            result["unindexed"].append(line)

    return result


def fetch_single_article(article_url, law_index, law_date, law_title, chapter_index, chapter_title, article_index, article_title, output_dir):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(article_url, headers=headers)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding

    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = []

    for p in soup.find_all("p"):
        a_tag = p.find("a", id=lambda x: x and x.startswith("dst"))
        if a_tag:
            text = p.get_text(strip=True)
            if text:
                paragraphs.append(text)

    structured = parse_article_document(
        paragraphs,
        law_index, 
        law_date,
        law_title,
        chapter_index,
        chapter_title,
        article_index,
        article_title
    )

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{article_index}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f"已保存 {filepath} ，共提取 {len(paragraphs)} 段正文")


def fetch_law(index_file, output_dir="articles"):
    # 假设你已经从 law_index.json 里读取了数据
    with open(index_file, "r", encoding="utf-8") as f:
        law_data = json.load(f)

    law_index = law_data["law_index"]
    law_date = law_data["law_date"]
    law_title = law_data["law_title"]

    # 遍历每个章节和文章
    for chapter in law_data["chapters"]:
        chap_idx = chapter["chapter_index"]
        chap_title = chapter["chapter_title"]
        for article in chapter["articles"]:
            art_idx = article["article_index"]
            art_title = article["article_title"]
            art_url = article["url"]
            fetch_single_article(
                art_url,
                law_index,
                law_date,
                law_title,
                chap_idx,
                chap_title,
                art_idx,
                art_title,
                output_dir
            )

def fetch_index_and_law(source_dir, law_name, output_dir="data/processed/laws"):
    law_index = fetch_law_index(source_dir, f"{output_dir}/{law_name}")
    fetch_law(law_index, f"{output_dir}/{law_name}/articles")

if __name__ == "__main__":
    SOURCE_DIR = "/document/cons_doc_LAW_11376/"
    LAW_NAME = "оn_the_procedure_for_leaving_and_entering"
    fetch_index_and_law(SOURCE_DIR, LAW_NAME)
