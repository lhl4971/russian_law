import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
import argparse
from fastmcp import FastMCP
from typing import List, Dict, Any
from langchain_chroma.vectorstores import Chroma
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever
from chains.lawyer_chain import get_rewrite_chain
from utils.retriever import get_self_query_retriever, get_bm25_retriever, get_ensemble_retriever, get_reranking_retriever
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from prompts import LAW_RETRIVING_REWRITE_PROMPT

# --- 参数解析 ---
def get_args():
    parser = argparse.ArgumentParser(description="Law MCP Server 配置")
    parser.add_argument(
        "--chroma_dir",
        type=str,
        default="data/chroma",
        help="ChromaDB 数据库存储路径 (默认: data/chroma)"
    )
    parser.add_argument(
        "--law_collection_name",
        type=str,
        default="law_articles",
        help="ChromaDB 法律条文集合名称 (默认: law_articles)"
    )
    parser.add_argument(
        "--doc_list_collection_name",
        type=str,
        default="required_documents_lists",
        help="ChromaDB 办理文件目录集合名称 (默认: required_documents_lists)"
    )
    parser.add_argument(
        "--use_reranker",
        action="store_true",
        help='Enable this to use reranker'
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="MCP 服务端口 (默认: 8000)"
    )
    return parser.parse_known_args()[0]

args = get_args()

embedding = HuggingFaceEmbeddings(
    model_name="ai-forever/ru-en-RoSBERTa",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

law_retriever = get_self_query_retriever(
    Chroma(
        collection_name=args.law_collection_name,
        persist_directory=args.chroma_dir,
        embedding_function=embedding
    )
)

if args.use_reranker:
    law_retriever = get_reranking_retriever(law_retriever)

rewrite_chain = get_rewrite_chain(LAW_RETRIVING_REWRITE_PROMPT)

# 创建 MCP 服务
mcp = FastMCP(name="LawMCPServer")

@mcp.tool()
def rewrite_query_for_law_search(user_query: str) -> str:
    """
    这是一个强大的工具，能将用户的口语化或非标准的俄语法律查询，重写为正式、精确的法律检索查询。

    当用户的输入包含缩写（如 "ВНЖ"）、口语化表达（如 "办居住证"）或模糊的法律术语时，
    Agent 应该优先使用此工具，以确保后续的向量检索能获得更高的匹配度和准确性。

    Args:
        user_query (str): 用户的原始查询文本，可以是任何非正式的表述。

    Returns:
        str: 返回经过重写后的正式俄语法律查询。
        例如："порядок получения вида на жительство"
    """
    rewritten_content = rewrite_chain.invoke({"user_query": user_query}).content
    return rewritten_content


@mcp.tool()
def search_law_articles(query: str, n_results: int = 20) -> List[Dict[str, Any]]:
    """
    一个强大的法律知识检索工具，结合了向量相似度检索和元数据过滤器。

    该工具能够根据用户的自然语言问题或精确的法律坐标，在法律数据库中查找并返回最相关的法律条文。它支持处理语义查询、精确过滤以及两者混合的查询，应当避免使用缩写、口语化表达或模糊的法律术语。

    Args:
        query (string): 用于检索的查询文本。这可以是用户的原始问题，也可以是包含法律标题、章节、条款编号等信息的混合查询。
        n_results (integer, optional): 指定要返回的最相关法律条文数量。默认值为 20。  
对于主题范围广泛、涉及多个情形或政策介绍类的问题（如“如何办理移民”），应适当增加返回条目的数量，以覆盖更多相关情形，通常建议在 30–50 之间。  
对于范围较窄、指向明确的具体问题（如“投资移民的最低资金要求”），可保持或减少返回条目的数量，以提高结果的精准度，一般为 10–20 条。

    Returns:
        List[Dict[str, Any]]: 返回一个包含多个字典的列表。每个字典代表一个独立的法律条文，并包含以下关键信息：
            - page_content (string): 法律条文的完整文本内容，已经包含其父级条款（如法律名称、章节、条款标题）作为上下文，以便直接使用。
            - metadata (dict): 一个包含丰富结构化信息的字典，例如法律文件的签发日期，编号，法律条文所属的章节，父条款编号等，可用于进一步分析或显示。

    示例：
    1. 纯语义查询: "Условия получения вида на жительство без разрешения на временное проживание"
    2. 混合查询: "В статье 8 Федерального закона 115, кто имеет право на получение вида на жительство?"
    3. 纯结构化过滤: "Содержание статьи 8 Федерального закона 'О правовом положении иностранных граждан в Российской Федерации' "
    """
    if isinstance(law_retriever, EnsembleRetriever):
        config = {
            "configurable": {
                "bm25_k_id": n_results * 2,
                "selfquery_search_kwargs": {"k": n_results * 2}
            }
        }
        docs = law_retriever.invoke(query, config=config)
        return docs[:n_results]
    elif isinstance(law_retriever, BaseRetriever) and not isinstance(law_retriever, Runnable):
        # 普通 BaseRetriever，传入 config
        docs = law_retriever.invoke(query, config={"configurable": {"search_kwargs_id": {"k": n_results}}})
        return docs[:n_results]
    elif isinstance(law_retriever, Runnable):
        # LCEL/RunnableParallel 或 Lambda 链
        docs = law_retriever.invoke({"query": query, "top_n": n_results})
        return docs
    else:
        raise ValueError(f"Unsupported retriever type: {type(law_retriever)}")


if __name__ == "__main__":
    mcp.run(transport="sse", port=args.port)
