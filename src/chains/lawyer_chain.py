import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
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
