from typing import List, Dict, Any
from langchain_core.documents import Document

# --- 辅助函数：解析JSON并创建增强型文档 ---
def parse_law_json_to_docs(data: Dict[str, Any]) -> List[Document]:
    documents = []
    base_metadata = {
        "law_index": int(data["law_index"]),
        "law_date": data["law_date"],
        "chapter_index": data["chapter_index"],
        "article_index": data["article_index"]
    }

    # 处理 Unindexed
    for i, unindexed_text in enumerate(data.get("unindexed", [])):
        unindexed_metadata = base_metadata.copy()
        unindexed_metadata.update({
            "type": "unindexed_paragraph",
            "paragraph_order": i + 1, # 记录段落顺序
        })
        enhanced_content = (
            f"{data['law_title']}\n"
            f"{data['chapter_title']}\n"
            f"{data['article_title']}\n"
            f"{unindexed_text}"
        )
        documents.append(Document(page_content=enhanced_content, metadata=unindexed_metadata))

    for clause in data.get("clauses", []):
        # 处理 Clause
        clause_metadata = base_metadata.copy()
        clause_metadata.update({
            "type": "clause",
            "clause_index": clause.get("clause_index", ""),
        })
        enhanced_content = (
            f"{data['law_title']}\n"
            f"{data['chapter_title']}\n"
            f"{data['article_title']}\n"
            f"{clause['clause_text']}"
        )
        documents.append(Document(page_content=enhanced_content, metadata=clause_metadata))
        
        # 处理 Unindexed Clause
        for i, unindexed_text in enumerate(clause.get("unindexed", [])):
            unindexed_metadata = clause_metadata.copy()
            unindexed_metadata.update({
                "type": "unindexed_paragraph",
                "paragraph_order": i + 1, # 记录段落顺序
            })
            enhanced_content = (
                f"{data['law_title']}\n"
                f"{data['chapter_title']}\n"
                f"{data['article_title']}\n"
                f"{clause['clause_text']}\n"
                f"{unindexed_text}"
            )
            documents.append(Document(page_content=enhanced_content, metadata=unindexed_metadata))

        # 处理 Sub-clauses
        for subclause in clause.get("subclauses", []):
            subclause_metadata = clause_metadata.copy()
            subclause_metadata.update({
                "type": "subclause",
                "subclause_index": subclause.get("subclause_index", ""),
            })
            enhanced_content = (
                f"{data['law_title']}\n"
                f"{data['chapter_title']}\n"
                f"{data['article_title']}\n"
                f"{clause['clause_text']}\n"
                f"{subclause['subclause_text']}"
            )
            documents.append(Document(page_content=enhanced_content, metadata=subclause_metadata))

            # 处理 Unindexed Sub-clauses
            for i, unindexed_text in enumerate(subclause.get("unindexed", [])):
                unindexed_metadata = subclause_metadata.copy()
                unindexed_metadata.update({
                    "type": "unindexed_paragraph",
                    "paragraph_order": i + 1, # 记录段落顺序
                })
                enhanced_content = (
                    f"{data['law_title']}\n"
                    f"{data['chapter_title']}\n"
                    f"{data['article_title']}\n"
                    f"{clause['clause_text']}\n"
                    f"{subclause['subclause_text']}\n"
                    f"{unindexed_text}"
                )
                documents.append(Document(page_content=unindexed_text, metadata=unindexed_metadata))

    return documents