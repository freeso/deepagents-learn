# 第 4 章实战：任务规划（TodoListMiddleware）与中间件机制
#
# 三个实验：
#   实验一：开/不开 TodoListMiddleware 对比 —— 同一多步任务跑两遍，看行为差异（需 LLM）
#   实验二：todos 的持久化与跨 invoke 接续 —— Checkpointer + 同 thread_id（需 LLM）
#   实验三：SummarizationMiddleware 触发观察 —— 手动组装 LangChain 版，亲眼看摘要替换（需 LLM）
#
# 运行：source .venv/bin/activate && python ch04-experiments.py
# 可只跑部分实验：python ch04-experiments.py 1 2
#（免费档模型偶发长挂起时，方便分段重跑）
# 依赖环境变量：SILICONFLOW_API_KEY（可选 MODEL_NAME，默认 Qwen/Qwen2.5-7B-Instruct）
#
# 实验一/二使用【不联网的模拟手机参数库】代替 Tavily：
# 快、免费、结果确定，可控对比"有没有规划"的行为差异。

import os
import sys

# 命令行指定实验编号；空则全跑
ONLY = {a for a in sys.argv[1:] if a.isdigit()}


def should_run(n: int) -> bool:
    return not ONLY or str(n) in ONLY


# 复用 research_deepagent 里的 .env：手动解析（不引入 python-dotenv 依赖），
# shell 里已 export 的环境变量优先
def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"\''))


load_env_file(os.path.expanduser("~/research_deepagent/.env"))

from langchain_openai import ChatOpenAI  # noqa: E402

# SiliconFlow key：优先 shell 的 SILICONFLOW_API_KEY，其次 .env 里的 OPENAI_API_KEY（同一把 key）
api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("缺少 API key：请 export SILICONFLOW_API_KEY=... 后重试")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    temperature=0,
    # 免费档偶发长挂起：超时 + 自动重试，避免无限等待
    timeout=120,
    max_retries=3,
)


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_agent(agent, question: str, thread_id: str = "t1") -> dict:
    """运行 Agent 并流式打印每一步动作（write_todos/工具调用实时可见）。"""
    tool_calls_seen: list[str] = []
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
        config={"configurable": {"thread_id": thread_id}},
    ):
        for node_name, node_update in chunk.items():
            for msg in (node_update or {}).get("messages", []):
                for tc in getattr(msg, "tool_calls", None) or []:
                    if tc["name"] == "write_todos":
                        # 只打印状态摘要，不打印整个清单（太长）
                        items = tc["args"].get("todos", [])
                        statuses = "".join(
                            {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}.get(t.get("status"), "?")
                            for t in items
                        )
                        print(f"  📋 [{node_name}] write_todos：{len(items)} 项 [{statuses}]")
                    else:
                        print(f"  ⚙️  [{node_name}] {tc['name']}({str(tc['args'])[:80]})")
                    tool_calls_seen.append(tc["name"])
                if type(msg).__name__ == "ToolMessage" and msg.name != "write_todos":
                    print(f"  ✅ [{node_name}] {msg.name} -> {str(msg.content)[:70]}")
    values = agent.get_state({"configurable": {"thread_id": thread_id}}).values
    values["_tool_calls_seen"] = tool_calls_seen  # 附带统计，方便对比
    return values


def print_todos(values: dict, label: str) -> None:
    """打印 state 里的 todos 字段（TodoListMiddleware 注入的状态）。"""
    todos = values.get("todos") or []
    icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
    print(f"  🗂  {label}（state['todos']，共 {len(todos)} 项）：")
    for t in todos:
        print(f"      {icon.get(t.get('status'), '?')} {t.get('content', '')[:50]}")


# =====================================================================
# 模拟手机参数库：三个品牌的数据写死，工具永不抛异常（只返回数据或错误字符串）
# =====================================================================
PHONE_DB: dict[str, dict] = {
    "Aurora": {
        "price_cny": 4999, "battery_mah": 5000, "weight_g": 201,
        "camera": "5000万像素三摄", "screen": "6.7 英寸 OLED 120Hz", "os_years": 5,
    },
    "Nimbus": {
        "price_cny": 3599, "battery_mah": 6000, "weight_g": 226,
        "camera": "6400万像素双摄", "screen": "6.8 英寸 LCD 90Hz", "os_years": 3,
    },
    "Zenith": {
        "price_cny": 6299, "battery_mah": 4400, "weight_g": 172,
        "camera": "一英寸大底双摄", "screen": "6.3 英寸 OLED 144Hz", "os_years": 7,
    },
}


def get_phone_specs(brand: str) -> dict:
    """查询指定品牌手机的完整参数（价格/电池/重量/相机/屏幕/系统支持年限）。

    Args:
        brand: 品牌名，可选值：Aurora、Nimbus、Zenith。

    Returns:
        dict：该品牌的完整参数；品牌不存在时返回 {"error": "说明"}。
    """
    data = PHONE_DB.get(brand)
    if data is None:
        return {"error": f"没有找到品牌 {brand}，可选：Aurora、Nimbus、Zenith"}
    return {"brand": brand, **data}


# 实验一/二共用的任务与提示词
RESEARCH_TASK = (
    "请对比 Aurora、Nimbus、Zenith 三款手机的参数（价格、电池、相机、屏幕各维度都要覆盖），"
    "最后给出一份分人群（预算型/续航型/影像型）的购买建议，200 字左右。"
)

# 提示词与实验变体严格对应：
# ⚠️ 坑：A 组（未装 Todo）的提示词里不能提 write_todos，否则模型会幻觉调用不存在的工具
#（首轮实验中 Qwen 真的对未注册的 write_todos 发起了空参数调用）
NEUTRAL_PROMPT = """你是一位专业的数码产品研究员。请逐步查询所有需要的数据，对比分析后输出最终购买建议。"""

PLANNING_PROMPT = """你是一位专业的数码产品研究员。
面对复杂任务时，你会：
1. 先用 write_todos 制定研究计划（每款手机的查询、对比分析、写建议都应是独立条目）
2. 逐步执行每个步骤，开始前标 in_progress，完成后立即标 completed
3. 最后综合所有数据输出购买建议
"""


# =====================================================================
# 实验一：开/不开 TodoListMiddleware —— 同一任务跑两遍对比行为
# =====================================================================
def exp1() -> None:
    banner("实验一：TodoListMiddleware 开/关对比（同一多步任务）")

    from langchain.agents.middleware import TodoListMiddleware
    from langgraph.checkpoint.memory import InMemorySaver

    from deepagents import create_deep_agent

    # ---- A. 不开规划（v0.7 默认行为）----
    print(">>> A. 未加 TodoListMiddleware（默认）")
    plain_agent = create_deep_agent(
        model=model,
        tools=[get_phone_specs],
        # ⚠️ A 组用中性提示词：不能提 write_todos，否则模型会幻觉调用不存在的工具
        system_prompt=NEUTRAL_PROMPT,
        checkpointer=InMemorySaver(),  # get_state 统计需要
    )
    ra = run_agent(plain_agent, RESEARCH_TASK, thread_id="exp1-plain")
    print_todos(ra, "无 TodoListMiddleware")
    if not ra.get("todos"):
        print("      （没有 todos 字段或为空 —— v0.7 默认不注入规划能力）")

    # ---- B. 开规划 ----
    print("\n>>> B. 显式加了 middleware=[TodoListMiddleware()]")
    todo_agent = create_deep_agent(
        model=model,
        tools=[get_phone_specs],
        system_prompt=PLANNING_PROMPT,
        middleware=[TodoListMiddleware()],  # ← 与 A 组的核心区别
        checkpointer=InMemorySaver(),
    )
    rb = run_agent(todo_agent, RESEARCH_TASK, thread_id="exp1-todo")
    print_todos(rb, "有 TodoListMiddleware")

    # ---- 对比统计 ----
    print("\n  📊 行为对比：")
    for label, r in (("A 未开启规划", ra), ("B 开启规划", rb)):
        calls = r["_tool_calls_seen"]
        stats = {
            "get_phone_specs": calls.count("get_phone_specs"),
            "write_todos": calls.count("write_todos"),
        }
        final = r["messages"][-1].content
        covered = sum(b in final for b in PHONE_DB)  # 最终回答是否覆盖三款
        print(
            f"     {label}: 查参数 {stats['get_phone_specs']} 次 | "
            f"write_todos {stats['write_todos']} 次 | 清单 {len(r.get('todos') or [])} 项 | "
            f"最终回答覆盖品牌 {covered}/3"
        )
    print("\n📌 结论：开了规划后 Agent '先写工单再干活'；没开时也能完成任务（模型能力够的话），")
    print("   但缺少进度追踪的显式载体——对长任务/弱模型/产品 UI 展示进度，这个载体很关键。")


# =====================================================================
# 实验二：todos 持久化 —— Checkpointer + 同 thread_id 跨 invoke 接续
# =====================================================================
def exp2() -> None:
    banner("实验二：todos 跨 invoke 接续（Checkpointer + 同 thread_id）")

    from langchain.agents.middleware import TodoListMiddleware
    from langgraph.checkpoint.memory import InMemorySaver

    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=model,
        tools=[get_phone_specs],
        system_prompt=PLANNING_PROMPT,
        middleware=[TodoListMiddleware()],
        checkpointer=InMemorySaver(),  # 关键：没有它，第二次 invoke 找不到上次的 todos
    )

    # 第一轮：只让它做第一步就停（人为制造"未完成任务"）
    print(">>> 第 1 次 invoke：只查询参数，先不写建议")
    r1 = run_agent(
        agent,
        "请先用 write_todos 规划整个任务，但本轮只执行'查询三款手机参数'这一步，写完建议的步骤先停在 pending。",
        thread_id="exp2-thread",
    )
    print_todos(r1, "第 1 次调用后")

    # 第二轮：让它接着上次的清单继续
    print("\n>>> 第 2 次 invoke（同 thread_id）：继续完成剩余步骤")
    r2 = run_agent(agent, "继续，完成清单里剩下的所有步骤。", thread_id="exp2-thread")
    print_todos(r2, "第 2 次调用后")

    done = all(t.get("status") == "completed" for t in (r2.get("todos") or [{}])) and r2.get("todos")
    print(f"\n📌 结论：同 thread_id + Checkpointer 时，第 2 次 invoke 能看到第 1 次的清单并接着做完"
          f"（本轮是否全部完成：{'是' if done else '否——需验产物，勾完≠做对'}）。")
    if not done:
        print("   注意：清单没全勾/行为不稳定都是弱模型的正常表现——这正是'completed 是 Agent 自己打的勾'")
        print("   的含义，应用层必须验最终产物（本次看最终回答是否给出三人群建议即可）。")


# =====================================================================
# 实验三：SummarizationMiddleware 触发观察（LangChain 版，手动组装）
# =====================================================================
def exp3() -> None:
    banner("实验三：SummarizationMiddleware —— 亲眼看摘要替换发生")

    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware
    from langgraph.checkpoint.memory import InMemorySaver

    # 课程提醒：两个同名类的默认 trigger 都是 None（不触发）！必须手动设置
    agent = create_agent(
        model=model,
        tools=[get_phone_specs],
        checkpointer=InMemorySaver(),  # get_state 观察历史需要
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=("messages", 6),   # 历史超过 6 条消息就摘要
                keep=("messages", 2),      # 保留最近 2 条
            ),
        ],
    )

    rounds = [
        "查一下 Aurora 的参数",
        "再查 Nimbus",
        "Zenith 也查一下",
        "现在三款都查过了，请总结三款的价格",
        "刚才我让你查过哪几款手机？分别多少钱？",
    ]
    for i, q in enumerate(rounds, 1):
        # 只 invoke 不流式（每轮都很短）
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": q}]},
            stream_mode="updates",
            config={"configurable": {"thread_id": "exp3"}},
        ):
            pass
        values = agent.get_state({"configurable": {"thread_id": "exp3"}}).values
        msgs = values["messages"]
        types = [type(m).__name__ for m in msgs]
        # LangChain 版触发后：旧消息被替换成一条摘要消息（invoke 前 6 条变 3 条之类）
        print(f"  第{i}轮后 历史共 {len(msgs)} 条: {types}")
        if len(msgs) <= 4 and len(msgs) < (2 * i):  # 条数不随轮次增长 => 触发了摘要
            first = msgs[0]
            print(f"      ⚡ 疑似已触发摘要！历史第 1 条是 {type(first).__name__}:")
            print(f"      {str(first.content)[:150]!r}")

    print("\n📌 结论：trigger 到阈值后，LangChain 版 SummarizationMiddleware 会把旧消息")
    print("   替换成一条摘要（历史条数不再增长就是证据）；别忘了手动组装时 trigger 默认是 None，")
    print("   不设置它摘要永远不会发生——'图里有摘要节点'不等于'摘要已触发'。")


if __name__ == "__main__":
    for n, fn in enumerate((exp1, exp2, exp3), start=1):
        if should_run(n):
            fn()
    print("\n全部实验完成 ✅")
