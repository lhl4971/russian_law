import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate
import asyncio
from agents.prompts import lawyer_prompt
import argparse

# --- 解析参数 ---
def get_args():
    parser = argparse.ArgumentParser(description="Building a ChromaDB vector database for legal documents")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show tool calling and observation content (default: False)"
    )
    return parser.parse_args()


async def main():
    args = get_args()

    # 连接到部署在 127.0.0.1:8000/sse 的 MCP 工具
    client = MultiServerMCPClient(
        {
            "law": {
                "url": "http://127.0.0.1:8000/sse",
                "transport": "sse",
            }
        }
    )

    llm = ChatOpenAI(
        model=os.getenv("STD_MIGRATION_MODEL_AGENT"),
        api_key=os.getenv("STD_MIGRATION_API_KEY_AGENT"),
        base_url=os.getenv("STD_MIGRATION_URL_AGENT"),
        temperature=0
    )

    # 获取 MCP 工具列表
    tools = await client.get_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system", lawyer_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{messages}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 使用 ReAct 风格创建 LangGraph Agent
    agent = create_react_agent(llm, tools, prompt=prompt)

    # 流式调用 Agent
    async for event in agent.astream_events(
        {"messages": "РВПО换ВНЖ文件列表？"},
        version="v1"
    ):

        etype = event["event"]

        # 工具调用开始
        if etype == "on_tool_start":
            if args.verbose:
                print(f"\n[Action]: 调用工具 {event['name']}，参数={event['data']}")

        # 工具返回结果
        elif etype == "on_tool_end":
            if args.verbose:
                output = event["data"]["output"]
                if len(output.content) > 200:
                    output.content = output.content[:150] + " … " + output.content[-50:]
                print(f"[Observation]: 工具 {event['name']} 返回 -> {output}")

        # 模型输出 token
        elif etype == "on_chat_model_stream":
            delta = event["data"]["chunk"].content
            if delta:
                print(delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
