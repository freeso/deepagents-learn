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
print(f"[问题] {question}")
print(f"[模型] {model.model_name}")
print("[运行中] 以下为 Agent 实时动作流（invoke 是黑盒等待，stream 能看到每一步）：\n")

# ---- 流式运行：stream_mode="updates" 每个节点执行完就推一次更新 ----
from langchain_core.messages import ToolMessage  # noqa: E402

final_messages: list = []
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="updates",
):
    for node_name, node_update in chunk.items():
        msgs = (node_update or {}).get("messages", [])
        for msg in msgs:
            # 模型发起的工具调用：实时打印工具名和参数
            for tc in getattr(msg, "tool_calls", None) or []:
                print(f"  ⚙️  [{node_name}] 调用工具 {tc['name']}({str(tc['args'])[:100]})")
            # 工具执行结果：打印结果大小
            if isinstance(msg, ToolMessage):
                status = "✅" if msg.status == "success" else f"⚠️ {msg.status}"
                print(f"  {status} [{node_name}] {msg.name} 返回 {len(str(msg.content))} 字符")
        final_messages.extend(msgs)

# ---- 统计信息 ----
tool_call_count = sum(
    1 for m in final_messages if getattr(m, "tool_calls", None)
    for _ in m.tool_calls
)
print(f"\n[Agent 共发起 {tool_call_count} 次工具调用]")

print("\n[最终回答]")
print(final_messages[-1].content)
