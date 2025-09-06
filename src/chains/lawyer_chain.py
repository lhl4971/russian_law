import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from utils.process_query import preprocess_data

def get_rewrite_chain(rewrite_prompt: str):
    rewrite_llm = ChatOpenAI(
        model=os.getenv("STD_MIGRATION_MODEL"),
        api_key=os.getenv("STD_MIGRATION_API_KEY"),
        base_url=os.getenv("STD_MIGRATION_URL"),
        temperature=0
    )
    # 使用 ChatPromptTemplate 构建可复用的提示词
    rewrite_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", rewrite_prompt),
            ("human", "用户查询: {user_query}")
        ]
    )
    rewrite_chain = RunnableLambda(preprocess_data) | rewrite_prompt_template | rewrite_llm
    return rewrite_chain

def get_doc_list_chain(doc_list_prompt: str, doc_lists: dict):
    llm = ChatOpenAI(
        model=os.getenv("STD_MIGRATION_MODEL"),
        api_key=os.getenv("STD_MIGRATION_API_KEY"),
        base_url=os.getenv("STD_MIGRATION_URL"),
        temperature=0
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", doc_list_prompt),
            ("user", "Пользовательский запрос: \"{user_query}\"\n\nСписок вариантов:\n{candidates}\n")
        ]
    )

    def select_candidates(inputs: dict):
        doc_type = inputs["doc_type"]
        return {
            "user_query": inputs["user_query"],
            "candidates": [{"id": doc["id"], "text": doc["text"]} for doc in doc_lists.get(doc_type, [])],
        }

    candidate_selector = RunnableLambda(select_candidates)

    doc_list_chain = candidate_selector | prompt_template | llm | JsonOutputParser()

    return doc_list_chain
