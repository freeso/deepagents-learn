# 故意很慢的异步研究员：纯 LangGraph，不调 LLM（8 秒稳定可见）
#
# 为什么要"故意慢"：真实调研任务耗时不可控，实验要的是稳定复现
# "start 立即返回 → running → success" 这条生命周期曲线。
# 结论用 PHONE_DB 确定性计算（延续第 3/4/5 章的模拟数据），
# 避免弱模型把结果算错——异步机制本身才是本章的观察对象。

import asyncio

from langgraph.graph import END, START, MessagesState, StateGraph

PHONE_DB: dict[str, dict] = {
    "Aurora": {"price_cny": 4999, "battery_mah": 5000, "weight_g": 201},
    "Nimbus": {"price_cny": 3599, "battery_mah": 6000, "weight_g": 226},
    "Zenith": {"price_cny": 6299, "battery_mah": 4400, "weight_g": 172},
}


async def slow_research(state: MessagesState) -> dict:
    # 领任务：主 Agent 通过 update_async_task 追加的指令会出现在最新的人类消息里
    # （服务端 state 里消息可能是 dict，也可能是 LangChain 消息对象，两种都兼容）
    last_human = ""
    for m in reversed(state["messages"]):
        if isinstance(m, dict):
            if m.get("role") in ("human", "user"):
                last_human = str(m.get("content", ""))
                break
        elif getattr(m, "type", "") == "human":
            last_human = str(getattr(m, "content", ""))
            break

    # 模拟深度调研：8 秒"苦工期"
    await asyncio.sleep(8)

    # 结论是确定性代码算出来的，永远正确（min/max 一行一个）
    cheapest = min(PHONE_DB.items(), key=lambda kv: kv[1]["price_cny"])
    best_battery = max(PHONE_DB.items(), key=lambda kv: kv[1]["battery_mah"])
    lightest = min(PHONE_DB.items(), key=lambda kv: kv[1]["weight_g"])

    report = (
        f"[researcher 完成，模拟耗时 8s]\n"
        f"收到的任务：{str(last_human)[:120]}\n"
        f"关键结论（确定性统计）：最便宜 {cheapest[0]}（{cheapest[1]['price_cny']} 元）、"
        f"电池最大 {best_battery[0]}（{best_battery[1]['battery_mah']} mAh）、"
        f"最轻 {lightest[0]}（{lightest[1]['weight_g']} g）。\n"
        "异步子 Agent 的意义：主 Agent 拿到任务 ID 就返回，后台慢慢干，"
        "随时可查进度、可追加要求、可取消。"
    )
    return {"messages": [{"role": "ai", "content": report}]}


builder = StateGraph(MessagesState)
builder.add_node("slow_research", slow_research)
builder.add_edge(START, "slow_research")
builder.add_edge("slow_research", END)
graph = builder.compile()
