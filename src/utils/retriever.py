import os
from langchain_openai import ChatOpenAI
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.runnables import ConfigurableField

# --- 定义元数据模式 ---
document_content_description = "法律条文的俄语文本内容，包含完整的法律、章节和条款上下文。例如：'Статья 2. Основные понятия ...'"
metadata_field_info = [
    AttributeInfo(
        name="law_index",
        description="法律的编号，例如标题中 'N 115-ФЗ' 中的 '115'。这通常是一个整数。",
        type="integer",
    ),
    AttributeInfo(
        name="law_issue_year",
        description=("法律的签发年份，必须严格输出为整数，例如标题中 'N 115-ФЗ от 25.07.2002' 中的 '2002'。这通常是一个整数。"),
        type="integer",
    ),
    AttributeInfo(
        name="chapter_index",
        description="法律章节的编号（Глава），如 '1', '2', '4.1'。如果原始文本中是罗马数字或罗马数字与阿拉伯数字的混合（如 'I', 'II', 'IV.1'），应转化为全阿拉伯数字。这个字段是**辅助性**的，用于组织和分类法律条文，但通常不作为单独查询的主要条件。",
        type="string",
    ),
    AttributeInfo(
        name="article_index",
        description="法律条的编号（Статья），例如 '6.2', '8' 或 '10'。**在同一部法律中，这个编号通常是唯一的，因此是定位特定法律条文最直接和可靠的标识符。** 如果法律条文没有编号，该字段可为空, 结尾如果存在一个 '.' 应该去掉他。例如，'Статья 8' 或 'ст.8' 的条文编号为 '8'。",
        type="string",
    ),
    AttributeInfo(
        name="clause_index",
        description="法律款的编号（Пункт），例如 '1' 或 '2.1'。该编号是相对于其所属的'Статья'。例如，'Пункт 1' 或 'п.1' 的款编号为 '1'，'п.2.3' 的款编号为 '2.3'。如果没有这个级别的编号，该字段可为空。",
        type="string",
    ),
    AttributeInfo(
        name="subclause_index",
        description="法律项的编号（Подпункт），例如 '1)', '2)', 'a)' 或 'б)'。该编号是相对于其所属的'Пункт'。例如，'Подпункт а)' 或 'пп.а' 的项编号为 'а'。如果没有这个级别的编号，该字段可为空。",
        type="string",
    ),
    AttributeInfo(
        name="type",
        description="文档片段的类型，可以是 'clause'（款）、'subclause'（项）或 'unindexed_paragraph'（没有编号的段落）。这有助于区分不同层级的文本片段，但该字段对于查询通常不重要。",
        type="string",
    ),
]


def get_self_query_retriever(vectorstore):
    query_llm = ChatOpenAI(
        model=os.getenv("STD_MIGRATION_MODEL"),
        api_key=os.getenv("STD_MIGRATION_API_KEY"),
        base_url=os.getenv("STD_MIGRATION_URL"),
        temperature=0
    )

    return SelfQueryRetriever.from_llm(
        llm=query_llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        search_type="mmr",
        search_kwargs={"k": 20}
    ).configurable_fields(
        search_kwargs=ConfigurableField(
            id="search_kwargs_id",
            name="Search Kwargs",
            description="控制返回文档的数量"
        )
    )
