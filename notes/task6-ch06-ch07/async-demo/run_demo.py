# 异步行为验证脚本：SDK 连接本地 Agent Server，同一会话连续七轮对话
#
# 期望观察到的现象（对照课程"你应该看到什么"）：
#   1. 派活立刻返回任务 ID（本轮耗时 ≈ 一次 LLM 调用，远小于 8 秒）
#   2. 后台任务跑着时闲聊照常
#   3. check 看到 running
#   4. update 不重开任务、任务 ID 不变
#   5. 等 8 秒后再 check → success 且带研究员结论
#   6. list 看到全部任务清单
#   7. 新任务立刻取消 → cancelled
#
# 运行前提：先启动 Agent Server（见 README/笔记），然后 python run_demo.py

import asyncio
import time

from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:2024")
ASSISTANT = "supervisor"


def extract_reply(result: dict) -> tuple[str, list]:
    """从 run 结果里抽 supervisor 的最后一条回复文本 + 本次全部工具调用名。"""
    msgs = result.get("messages") or []
    tools = []
    reply = ""
    for m in msgs:
        mc = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        # 只统计本轮之后新增内容的工具名，简单起见全量收集
        tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        if tcs:
            tools.extend(tc.get("name") for tc in tcs)
        if (m.get("type") if isinstance(m, dict) else getattr(m, "type", "")) == "ai" and mc:
            reply = mc if isinstance(mc, str) else str(mc)
    return reply, tools


async def ask(thread_id: str, content: str, label: str) -> dict:
    """发送一轮对话，打印回复与耗时（耗时就是'阻塞感'的量化）。"""
    t0 = time.perf_counter()
    result = await client.runs.wait(
        thread_id,
        ASSISTANT,
        input={"messages": [{"role": "user", "content": content}]},
    )
    elapsed = time.perf_counter() - t0
    reply, _ = extract_reply(result)
    print(f"\n{'─' * 60}")
    print(f"[{label}] 耗时 {elapsed:.1f}s")
    print(f"用户: {content}")
    print(f"经理: {reply[:500]}")
    return result


async def main() -> None:
    thread = await client.threads.create()
    tid = thread["thread_id"]
    print(f"thread_id = {tid}")

    # ① 派活：应该秒回任务 ID，而不是卡 8 秒
    await ask(tid, "请在后台调研 Aurora、Nimbus、Zenith 三款手机的市场表现。", "第 1 轮·派活")

    # ② 后台跑着，前台闲聊：验证不阻塞
    await ask(tid, "等结果的间隙，用两句话告诉我异步子 Agent 和同步子 Agent 的区别。", "第 2 轮·闲聊")

    # ③ 查进度：应该看到 running 或处理中
    await ask(tid, "刚才那个后台任务现在进展如何？", "第 3 轮·查进度")

    # ④ 追加要求：应该调用 update_async_task，而不是重新开始
    await ask(tid, "补充一个要求：完成时请把结论整理成 3 条要点。", "第 4 轮·追加要求")

    # ⑤ 等 8 秒确保研究员干完活，再查：应该 success 且带结论
    print("\n……脚本主动等待 10 秒，让后台研究员干完活……")
    await asyncio.sleep(10)
    await ask(tid, "现在再看看后台任务的进展和结果。", "第 5 轮·取结果")

    # ⑥ 看全部任务：应该调用 list_async_tasks
    await ask(tid, "把我所有的后台任务列个清单。", "第 6 轮·任务总览")

    # ⑦ 再派一个新任务并立刻取消：验证 start + cancel 组合拳
    await ask(tid, "再派一个新调研任务，然后立刻取消它。", "第 7 轮·新任务+取消")

    print("\n" + "═" * 60)
    print("验证完毕：对照上面 7 轮的耗时与回复，确认'派活即返回、查改停皆可'。")
    print("关键判定：第 1 轮耗时 << 8 秒（异步生效）；第 5 轮拿到研究员结论。")


if __name__ == "__main__":
    asyncio.run(main())
