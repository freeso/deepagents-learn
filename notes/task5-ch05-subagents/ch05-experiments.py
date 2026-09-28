# 第 5 章实战：子 Agent 与上下文隔离（Context Quarantine）
#
# 两个实验（实验一内含"草稿隔离 vs 档案共享"验证）：
#   实验一：主 Agent 亲自查 vs 委派子 Agent 查 —— 量化主上下文长度差异，
#           并验证"隔离的是草稿（消息流）、共享的是档案（Backend 文件）"
#   实验二：data-collector → data-analyzer → report-writer 三部门协作
#
# 运行：source .venv/bin/activate && python ch05-experiments.py
# 可只跑部分实验：python ch05-experiments.py 2
# 依赖环境变量：SILICONFLOW_API_KEY（可选 MODEL_NAME，默认 Qwen/Qwen2.5-7B-Instruct）
#
# 沿用第 3/4 章的【模拟手机参数库】：快、免费、结果确定。

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

api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("缺少 API key：请 export SILICONFLOW_API_KEY=... 后重试")

model = ChatOpenAI(
    model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    temperature=0,
    timeout=120,
    max_retries=3,
)

DEMO_DIR = os.path.join(os.path.dirname(__file__), ".tmp", "subagent-demo")


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_agent(agent, question: str, thread_id: str = "t1") -> dict:
    """流式运行 Agent：实时打印主 Agent 的每一步动作，最后回收完整状态。

    注意：子 Agent 的内部动作【不会】出现在这里——它们被隔离在子 Agent
    自己的上下文里，主 Agent 的流里只能看到一次 task(...) 调用。
    这正是 Context Quarantine 的直接体现。
    """
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
        config={"configurable": {"thread_id": thread_id}},
    ):
        for node_name, node_update in chunk.items():
            for msg in (node_update or {}).get("messages", []):
                for tc in getattr(msg, "tool_calls", None) or []:
                    print(f"  ⚙️  [{node_name}] {tc['name']}({str(tc['args'])[:70]})")
                if type(msg).__name__ == "ToolMessage":
                    print(f"  ✅ [{node_name}] {msg.name} -> {str(msg.content)[:70]}")
    return agent.get_state({"configurable": {"thread_id": thread_id}}).values


def context_stats(values: dict) -> dict:
    """统计主 Agent 上下文：消息条数、总字符数、工具调用次数。

    这就是 Context Quarantine 的"体温计"——同样的任务，
    委派得越好，主 Agent 的这几个数字越小。
    """
    msgs = values.get("messages", [])
    total_chars = sum(len(str(getattr(m, "content", ""))) for m in msgs)
    n_calls = sum(len(getattr(m, "tool_calls", None) or []) for m in msgs)
    return {"n_messages": len(msgs), "total_chars": total_chars, "n_tool_calls": n_calls}


# =====================================================================
# 模拟手机参数库（与第 3/4 章同一套，工具永不抛异常）
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


RESEARCH_TASK = (
    "请对比 Aurora、Nimbus、Zenith 三款手机的参数（价格、电池、相机、屏幕各维度都要覆盖），"
    "最后给出一份分人群（预算型/续航型/影像型）的购买建议，150 字左右。"
)

# =====================================================================
# 实验一：主 Agent 亲自查 vs 委派子 Agent 查 —— 量化 Context Quarantine
# =====================================================================
def exp1() -> None:
    banner("实验一：上下文隔离量化 —— 亲自干 vs 委派干")

    import shutil

    from langgraph.checkpoint.memory import InMemorySaver

    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # 委派对象：researcher 子 Agent（字典方式定义，三必填 + 精简工具）
    researcher = {
        "name": "researcher",
        # description 决定主 Agent 何时委派：要写行为，不要写名词（最佳实践 1）
        "description": "对多款手机做深度参数调研：需要多次查询参数库并整理原始数据时使用",
        # system_prompt 不继承主 Agent，必须自己写全（最佳实践 2/5：格式 + 字数限制）
        # ⚠️ 弱模型坑：step 列表太客气会跳步（首轮实验就没调 write_file）。
        # 要点：①第 2 步设为硬性关卡（不做完不许回复）②回复格式锚定路径开头
        "system_prompt": """你是手机参数研究员，可用工具：get_phone_specs（查参数）、write_file（写文件）。
严格按以下步骤执行，不许跳步：
1. 依次调用 get_phone_specs，查询任务要求的每一款手机；
2. 全部查完后，立刻调用 write_file 把所有原始查询结果（JSON 原文逐段粘贴）写入 /data/raw.txt；
3. 第 2 步完成之前，禁止输出最终回复。最后回复不超过 120 字：以"原始数据已写入 /data/raw.txt"开头，
   随后给出每款手机的关键数字（价格/电池容量）和一句话差异总结——
   要让项目经理拿到你的摘要就能直接写购买建议，不需要再去读原始文件。
禁止：把完整原始数据（全部 JSON）放进回复。""",
        "tools": [get_phone_specs],  # 最小权限：只给岗位需要的工具（最佳实践 3）
    }

    # ---- A 组：主 Agent 亲自干（无子 Agent）----
    print(">>> A. 主 Agent 亲自查（无 task 工具）")
    solo_agent = create_deep_agent(
        model=model,
        tools=[get_phone_specs],
        system_prompt="你是数码产品研究员。请逐步查询所有需要的数据，对比分析后输出最终购买建议。",
        checkpointer=InMemorySaver(),
    )
    ra = run_agent(solo_agent, RESEARCH_TASK, thread_id="exp1-solo")
    stats_a = context_stats(ra)
    print(f"  📊 A 组主上下文: {stats_a}")

    # ---- B 组：委派给 researcher 子 Agent ----
    print("\n>>> B. 主 Agent 委派给子 Agent 查")
    boss_agent = create_deep_agent(
        model=model,
        tools=[],  # 主 Agent 自己不带锤子，逼它学会委派
        system_prompt="""你是项目经理，你下面有一位专项研究员。
面对调研类任务时，不要自己动手，必须用 task 工具委派给 researcher 子 Agent。
委派的 description 里必须包含：①要调研的品牌清单；②"将全部原始查询结果用 write_file 写入 /data/raw.txt"。
研究员的摘要会包含每款手机的关键数字（价格/电池），基于摘要直接输出分人群购买建议——
不要自己调用 read_file/grep 去读原始文件，你只需要摘要里的关键数字。""",
        subagents=[researcher],
        backend=FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True),
        checkpointer=InMemorySaver(),
    )
    rb = run_agent(boss_agent, RESEARCH_TASK, thread_id="exp1-boss")
    stats_b = context_stats(rb)
    print(f"  📊 B 组主上下文: {stats_b}")

    # ---- 量化对比 ----
    print("\n  📊 Context Quarantine 效果：")
    print(f"     A 亲自干: 消息 {stats_a['n_messages']} 条 | 总字符 {stats_a['total_chars']} | 工具调用 {stats_a['n_tool_calls']} 次")
    print(f"     B 委派干: 消息 {stats_b['n_messages']} 条 | 总字符 {stats_b['total_chars']} | 工具调用 {stats_b['n_tool_calls']} 次")
    if stats_a["total_chars"] > 0:
        ratio = stats_b["total_chars"] / stats_a["total_chars"]
        print(f"     B 组主上下文字符量 = A 组的 {ratio:.0%}")

    # ---- 草稿隔离 vs 档案共享验证 ----
    print("\n  🔬 隔离边界验证：")
    raw_marker = "battery_mah"  # 原始数据的特征字段，只应出现在档案里
    solo_msgs_text = "".join(str(getattr(m, "content", "")) for m in ra.get("messages", []))
    boss_msgs_text = "".join(str(getattr(m, "content", "")) for m in rb.get("messages", []))
    print(f"     A 组主上下文含原始数据字段({raw_marker}): {raw_marker in solo_msgs_text}（自己查的，理应能看到）")
    print(f"     B 组主上下文含原始数据字段({raw_marker}): {raw_marker in boss_msgs_text}（应为 False——草稿被隔离）")

    # 扫描整个 demo 目录（弱模型可能不严格写 /data/raw.txt，落盘路径以实际为准）
    written = [
        os.path.relpath(os.path.join(root, f), DEMO_DIR)
        for root, _, files in os.walk(DEMO_DIR)
        for f in files
    ]
    print(f"     子 Agent 写入 Backend 的档案（扫描 {DEMO_DIR} 的相对路径）: {written or '（空——子 Agent 没调 write_file）'}")
    for rel in written:
        with open(os.path.join(DEMO_DIR, rel), encoding="utf-8", errors="replace") as f:
            text = f.read()
        # 验证不认字段名（弱模型常把 JSON 重排成中文行），认品牌覆盖数 + 关键数字
        covered = sum(b in text for b in PHONE_DB)
        has_key_numbers = all(str(v["price_cny"]) in text for v in PHONE_DB.values())
        print(f"     - {rel}（{len(text)} 字符，覆盖品牌 {covered}/3，关键数字齐全: {has_key_numbers}）"
              f"—— 共用 Backend，主 Agent 可随时 read_file 取回")

    print("\n📌 结论：委派后主 Agent 的工具调用从 3 次降为 1 次、原始数据不出现在其上下文里")
    print("   （中间过程全部隔离在子 Agent 的草稿里）；子 Agent 按约定写入 Backend 的文件")
    print("   落在共享档案柜（磁盘 root_dir），主 Agent 可随时取回——隔离的是草稿，共享的是档案。")
    print("   ⚠️ 源码佐证（deepagents/graph.py）：每个字典子 Agent 都会自动注入")
    print("   FilesystemMiddleware(backend=共享backend)，文件工具与 tools=[...] 参数是两条独立通道——")
    print("   指定 tools=[get_phone_specs] 只替换业务工具，不会丢掉 write_file 等文件工具。")


# =====================================================================
# 实验二：data-collector → data-analyzer → report-writer 三部门协作
# =====================================================================
def analyze_specs(specs: list[dict]) -> dict:
    """对多款手机的参数列表做确定性统计分析（纯代码，不用 LLM）。

    Args:
        specs: 手机参数字典列表，每项需含 brand/price_cny/battery_mah/weight_g。

    Returns:
        dict： cheapest（最便宜款）、best_battery（电池最大款）、
               lightest（最轻款）、avg_price（均价）等关键指标。
    """
    if not specs:
        return {"error": "没有收到任何参数数据"}
    cheapest = min(specs, key=lambda s: s.get("price_cny", 0))
    best_battery = max(specs, key=lambda s: s.get("battery_mah", 0))
    lightest = min(specs, key=lambda s: s.get("weight_g", 99999))
    prices = [s.get("price_cny", 0) for s in specs]
    return {
        "cheapest": f"{cheapest['brand']}（{cheapest['price_cny']} 元）",
        "best_battery": f"{best_battery['brand']}（{best_battery['battery_mah']} mAh）",
        "lightest": f"{lightest['brand']}（{lightest['weight_g']} g）",
        "avg_price": round(sum(prices) / len(prices), 1),
        "n_analyzed": len(specs),
    }


def exp2() -> None:
    banner("实验二：三部门协作 —— collector → analyzer → writer")

    from langchain.agents.middleware import TodoListMiddleware
    from langgraph.checkpoint.memory import InMemorySaver

    from deepagents import create_deep_agent

    subagents = [
        {
            "name": "data-collector",
            "description": "从参数库收集手机原始数据，需要查询多款手机完整参数时使用",
            "system_prompt": "你是数据收集专员。用 get_phone_specs 逐个查询任务要求的品牌，"
                             "以紧凑列表形式返回全部参数数据（每款一行：品牌/价格/电池/相机/屏幕），不要分析。",
            "tools": [get_phone_specs],
        },
        {
            "name": "data-analyzer",
            "description": "对已收集的手机参数做统计分析，提取关键洞察，输入是参数数据时使用",
            "system_prompt": "你是数据分析专员。用 analyze_specs 工具对收到的参数数据做统计分析，"
                             "然后输出 3-5 条关键发现（每条一句话，注明数据依据），控制在 200 字以内。",
            "tools": [analyze_specs],
        },
        {
            "name": "report-writer",
            "description": "根据分析发现撰写最终购买建议报告，输入是分析结论时使用",
            "system_prompt": "你是技术写作专员。根据收到的分析发现，撰写 150 字左右的分人群"
                             "（预算型/续航型/影像型）购买建议，直接输出报告正文。",
            "tools": [],  # 显式空列表 = 不继承主 Agent 工具（写作岗不需要锤子）
        },
    ]

    boss = create_deep_agent(
        model=model,
        tools=[],  # 经理不带锤子，只负责派活
        middleware=[TodoListMiddleware()],  # 第 4 章：先写工单再逐项委派
        system_prompt="""你是项目经理，团队里有三位专员：data-collector（收集）、
data-analyzer（分析）、report-writer（写作）。
面对调研任务，严格按流水线执行：
1. 先用 write_todos 制定三步计划
2. task 委派 data-collector 收集三款手机的参数，拿到收集结果
3. 把收集结果原样转交给 data-analyzer 做统计分析，拿到关键发现
4. 把分析发现转交给 report-writer 撰写购买建议，拿到报告
5. 直接把 report-writer 的报告作为你的最终回答输出
每一步都必须用 task 委派给对应专员，不要自己动手。""",
        subagents=subagents,
        checkpointer=InMemorySaver(),
    )

    values = run_agent(
        boss,
        "请按流程完成：对比 Aurora、Nimbus、Zenith 三款手机并输出分人群购买建议。",
        thread_id="exp2-pipeline",
    )
    stats = context_stats(values)
    print(f"\n  📊 经理的主上下文: {stats}")
    print("     （理应有 3 次 task 调用，每次只收到一份数据/发现/报告）")

    # 打印工单最终状态（第 4 章知识回顾：todos 是经理的锚）
    todos = values.get("todos") or []
    icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
    print(f"  🗂  经理的工单（{len(todos)} 项）：")
    for t in todos:
        print(f"      {icon.get(t.get('status'), '?')} {t.get('content', '')[:50]}")

    print("\n📌 结论：主 Agent 只做编排（3 次 task 转交），每个部门在独立上下文里干活，")
    print("   数据在各环节之间以'精炼交接'的形式流动——这就是课程说的多子 Agent 协作模式。")
    print("   若执行顺序有出入（弱模型正常现象），对照输出检查它实际委派了几次、给了谁。")


if __name__ == "__main__":
    for n, fn in enumerate((exp1, exp2), start=1):
        if should_run(n):
            fn()
    print("\n全部实验完成 ✅")
