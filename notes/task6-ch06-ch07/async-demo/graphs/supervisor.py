# Supervisor 主 Agent：声明一个 AsyncSubAgent，自动获得 5 把"遥控器"
#
# ⚠️ 关键点（踩坑高发区）：
# 1. graph_id="researcher" 必须与 langgraph.json 的 graphs 键名一字不差
# 2. 不写 url → 走 ASGI 进程内传输（与 supervisor 同一个 Agent Server）
# 3. 不要传 checkpointer：langgraph dev 服务端自己提供持久化
# 4. system_prompt 要写"派完活立刻返回，不许自己轮询"——
#    否则弱模型会 start 完立刻 check，同步等待把异步优势吃掉

import os

from langchain_openai import ChatOpenAI

from deepagents import AsyncSubAgent, create_deep_agent

model = ChatOpenAI(
    model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    api_key=os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    temperature=0,
    timeout=120,
    max_retries=3,
)

graph = create_deep_agent(
    model=model,
    system_prompt="""你是项目经理，负责通过异步子 Agent 完成长耗时调研。
你有 5 个遥控器（工具）：
- start_async_task：启动后台任务，立刻返回任务 ID
- check_async_task：查询任务进度，已完成则取回结果
- update_async_task：给运行中的任务追加要求
- cancel_async_task：取消运行中的任务
- list_async_tasks：列出所有任务及状态

严格遵守以下规则：
1. 用户提出调研类任务时：立刻调用 start_async_task 委派给 researcher，
   把返回的任务 ID 原样告诉用户，然后立即结束本轮回复。
   禁止等待结果、禁止自己连续调用 check_async_task。
2. 只有用户明确询问进度或结果时才调用 check_async_task。
3. 用户想补充或修改要求时：调用 update_async_task 更新已有任务，不要重新 start。
4. 用户要求取消时：调用 cancel_async_task。
5. 用户要求查看所有任务时：调用 list_async_tasks。
6. 与任务无关的闲聊直接回答，不调用任何工具。""",
    subagents=[
        AsyncSubAgent(
            name="researcher",
            description=(
                "长耗时后台调研专员：接收手机市场调研任务，"
                "在后台运行约 8 秒后返回确定性统计结论。"
            ),
            graph_id="researcher",  # 与 langgraph.json 注册名一致；不写 url → ASGI 进程内传输
        )
    ],
)
