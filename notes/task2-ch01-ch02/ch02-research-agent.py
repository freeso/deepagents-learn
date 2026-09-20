"""第 2 章实战：构建一个研究助手（Tavily 真实搜索 + TodoListMiddleware）

课程 2.9 节：https://datawhalechina.github.io/deepagents-in-action/chapters/ch02-quickstart/
运行：source /Users/fangcaili/Documents/CatPaw/DeepAgents学习/.venv/bin/activate
      python ch02-research-agent.py
环境：SILICONFLOW_API_KEY / MODEL_NAME 在 ~/.zshrc；TAVILY_API_KEY 取自 research_deepagent/.env
注意：复杂多步任务建议 GLM-5.2（课程建议）；本脚本默认沿用 MODEL_NAME 环境变量
"""

import os
import sys
from pathlib import Path

from typing import Literal

from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware

# TAVILY_API_KEY：优先环境变量，其次读 research_deepagent/.env
if "TAVILY_API_KEY" not in os.environ:
    env_file = Path.home() / "research_deepagent" / ".env"
    for line in env_file.read_text().splitlines():
        if line.startswith(("TAVILY_API_KEY=", "SILICONFLOW_API_KEY=")):
            k, _, v = line.partition("=")
            os.environ.setdefault(k, v.strip())

# 1. 模型（课程默认免费 Qwen2.5-7B，复杂任务建议 zai-org/GLM-5.2）
model = ChatOpenAI(
    model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
)

# 2. 搜索客户端
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# 3. 搜索工具（课程原版：三要素完整）
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search for the given query.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        topic: The topic category for the search.
        include_raw_content: Whether to include raw page content.
    """
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# 4. 系统提示词（研究员人设）
research_instructions = """你是一位专业的研究员。
你的工作是进行深入研究，然后撰写一份完整的研究报告。

你可以使用 internet_search 工具搜索互联网获取信息。
"""

# 5. 创建 Agent（显式启用 TodoListMiddleware —— v0.7 不再默认开启任务规划）
agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=research_instructions,
    middleware=[TodoListMiddleware()],
)

# 6. 运行（问题可由命令行覆盖，方便复用）
question = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "什么是 LangGraph？它和 LangChain 是什么关系？"
)
print(f"[问题] {question}\n")
result = agent.invoke({"messages": [{"role": "user", "content": question}]})

# ---- 附：观察 Agent 背后做了什么（比课程多走一步） ----
from langchain_core.messages import ToolMessage  # noqa: E402

tool_calls: list[tuple[str, str]] = []
for msg in result["messages"]:
    # AIMessage.tool_calls：模型发起的工具调用（工具名 + 参数）
    for tc in getattr(msg, "tool_calls", None) or []:
        tool_calls.append((tc["name"], str(tc["args"])[:80]))
    # ToolMessage：工具实际执行的记录（工具名）
    if isinstance(msg, ToolMessage):
        tool_calls.append((f"  -> {msg.name}", f"结果 {len(str(msg.content))} 字符"))

print(f"\n[Agent 共发起 {sum(1 for n, _ in tool_calls if not n.startswith(' '))} 次工具调用]")
for name, detail in tool_calls:
    print(f"  {name:20s} {detail}")

print("\n[最终回答]")
print(result["messages"][-1].content)
