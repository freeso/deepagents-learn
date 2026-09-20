# 第 3 章实战：虚拟文件系统与可插拔存储后端
#
# 五个实验，从"无 LLM 的纯后端验证"到"Agent 实际使用后端"：
#   实验一：FilesystemBackend 编程调用 —— 7 个底层操作 + 路径沙箱（无需 LLM）
#   实验二：StateBackend（默认）—— 同 thread 记得，换 thread 就忘（需 LLM）
#   实验三：FilesystemBackend —— Agent 写的文件真实落盘（需 LLM）
#   实验四：大结果自动卸载 —— 超限工具结果自动转存为文件引用（需 LLM）
#   实验五：CompositeBackend + FilesystemPermission —— 混合路由 + 敏感路径拒绝（需 LLM）
#
# 运行：source .venv/bin/activate && python ch03-experiments.py
# 可只跑部分实验：python ch03-experiments.py 1 4
#（免费档模型偶发长挂起时，方便分段重跑而不用从头再来）
# 依赖环境变量：SILICONFLOW_API_KEY（可选 MODEL_NAME，默认 Qwen/Qwen2.5-7B-Instruct）

import os
import shutil
import sys

# 命令行指定实验编号（如 "python ch03-experiments.py 1 4"）；空则全跑
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

model = ChatOpenAI(
    model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    temperature=0,
    # 免费档偶发长挂起：超时 + 自动重试，避免无限等待
    timeout=120,
    max_retries=3,
)

DEMO_DIR = os.path.join(os.path.dirname(__file__), ".tmp", "fs-demo")


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_agent(agent, question: str, thread_id: str = "t1") -> dict:
    """运行 Agent 并流式打印每一步动作（避免 invoke 黑盒等待）。

    注意：get_state 需要 checkpointer，调用方创建 agent 时须传入
    checkpointer=InMemorySaver() 实例（root graph 不支持 checkpointer=True）。
    """
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
        config={"configurable": {"thread_id": thread_id}},
    ):
        for node_name, node_update in chunk.items():
            for msg in (node_update or {}).get("messages", []):
                for tc in getattr(msg, "tool_calls", None) or []:
                    print(f"  ⚙️  [{node_name}] {tc['name']}({str(tc['args'])[:90]})")
                if type(msg).__name__ == "ToolMessage":
                    print(f"  ✅ [{node_name}] {msg.name} -> {str(msg.content)[:90]}")
    return agent.get_state({"configurable": {"thread_id": thread_id}}).values


# =====================================================================
# 实验一：FilesystemBackend 编程调用（不需要 LLM，确定性验证）
# =====================================================================
def exp1() -> None:
    banner("实验一：FilesystemBackend 编程调用 + 路径沙箱")

    from deepagents.backends import FilesystemBackend

    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    os.makedirs(DEMO_DIR, exist_ok=True)

    fs = FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True)

    print("1) write:", fs.write("/a/report.md", "# 季度报告\n第一行内容\n第二行内容"))
    # ReadResult.file_data 是 dict：{'content': ..., 'encoding': ...}
    print("2) read :", fs.read("/a/report.md").file_data["content"].splitlines()[:2])
    print("3) ls   :", [e["path"] for e in fs.ls("/").entries])
    # GlobResult 的字段叫 matches（不是 entries），每项也是 dict
    print("4) glob :", [m["path"] for m in fs.glob("**/*.md").matches])
    print("5) grep :", [m["path"] for m in fs.grep("季度").matches])

    # 沙箱验证：试图用 ../ 逃出 root_dir
    print("6) 越界写入 /../evil.txt ...")
    try:
        fs.write("/../evil.txt", "越权内容")
    except ValueError as e:
        print(f"   ✅ 被路径沙箱拦截：{e}")

    # edit_file 的底层：精确字符串替换
    fs.edit("/a/report.md", "第一行内容", "第一行内容（已修改）")
    print("7) edit 后 read :", fs.read("/a/report.md").file_data["content"].splitlines()[1])

    # delete
    fs.delete("/a/report.md")
    print("8) delete 后 glob:", [m["path"] for m in fs.glob("**/*.md").matches])

    print("\n📌 结论：FilesystemBackend 的 7 个能力（ls/read/write/edit/delete/glob/grep）")
    print("   可以不经过 LLM 直接编程调用；virtual_mode=True 的路径沙箱把 ../ 逃逸挡在 root_dir 内。")


# =====================================================================
# 实验二：StateBackend（默认后端）—— 同 thread 记得，换 thread 就忘
# =====================================================================
def exp2() -> None:
    banner("实验二：StateBackend —— Agent 的'草稿纸'（换 thread 即失忆）")

    from langgraph.checkpoint.memory import InMemorySaver

    from deepagents import create_deep_agent

    state_agent = create_deep_agent(
        model=model,
        system_prompt=(
            "你是一个记笔记的助手。当用户让你记录内容时，"
            "必须使用 write_file 工具写入 /workspace/notes.md，写完回复'已记录'。"
            "当用户让你回忆时，先使用 read_file 读取 /workspace/notes.md 再回答。"
        ),
        # get_state / 多轮对话需要 checkpointer；root graph 不能用 True，要传实例
        checkpointer=InMemorySaver(),
    )

    s1 = run_agent(
        state_agent,
        "请帮我记录：我最喜欢的语言是 Python。然后告诉我你把笔记存到哪了。",
        thread_id="thread-A",
    )
    files_a = s1.get("files", {})
    print(f"  📁 thread-A 的 state.files: {list(files_a.keys())}")

    s2 = run_agent(state_agent, "我最喜欢的语言是什么？", thread_id="thread-A")
    print(f"  💬 同 thread 回忆: {s2['messages'][-1].content[:80]}")

    s3 = run_agent(state_agent, "我最喜欢的语言是什么？", thread_id="thread-B")
    files_b = s3.get("files", {})
    print(f"  💬 新 thread 回忆: {s3['messages'][-1].content[:80]}")
    print(f"  📁 thread-B 的 state.files: {list(files_b.keys()) or '（空——文件跟着 thread 走）'}")
    print("\n📌 结论：StateBackend 的文件存在 LangGraph State 里，同一个 thread_id 内持久，")
    print("   换 thread 就'失忆'——这就是'临时草稿纸'的含义。")


# =====================================================================
# 实验三：FilesystemBackend —— Agent 写的文件真实落盘
# =====================================================================
def exp3() -> None:
    banner("实验三：FilesystemBackend —— 文件真实写到本地磁盘")

    from langgraph.checkpoint.memory import InMemorySaver

    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    os.makedirs(DEMO_DIR, exist_ok=True)

    disk_agent = create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True),
        system_prompt=(
            "你是文件管理助手。用户让你记录内容时，"
            "必须用 write_file 写入 /notes.md。"
        ),
        checkpointer=InMemorySaver(),
    )
    run_agent(
        disk_agent,
        "请记录：今天学完了第 3 章虚拟文件系统。写完后告诉我文件在哪。",
    )
    real_file = os.path.join(DEMO_DIR, "notes.md")
    print(f"  📂 磁盘上真实存在 {real_file}: {os.path.exists(real_file)}")
    if os.path.exists(real_file):
        print(f"  📄 内容: {open(real_file).read()[:100]!r}")
    print("\n📌 结论：换成 FilesystemBackend 后，Agent 写的文件直接落在磁盘上，")
    print("   会话结束后依然存在。注意：Agent 能读 root_dir 下所有文件（含 .env），")
    print("   所以敏感目录不要作为 root_dir。")


# =====================================================================
# 实验四：大结果自动卸载（> tool_token_limit_before_evict 即转存文件）
# =====================================================================
def exp4() -> None:
    banner("实验四：大结果自动卸载 —— 工具结果自动转存为文件引用")

    from langgraph.checkpoint.memory import InMemorySaver

    from deepagents import create_deep_agent

    def fetch_big_data(keyword: str) -> str:
        """返回某个关键词的超大模拟数据（501 行 x 5 倍冗余，超过 20000 token 卸载阈值），用于触发自动卸载。

        Args:
            keyword: 要查询的关键词。

        Returns:
            超长的模拟查询结果文本。
        """
        lines = [f"# 大数据结果：{keyword}", ""]
        for i in range(1, 502):
            lines.append(f"{i}. 记录{i}：{keyword} 的第 {i} 条详细数据，包含大量冗余描述文字。" * 5)
        return "\n".join(lines)

    evict_agent = create_deep_agent(
        model=model,
        tools=[fetch_big_data],
        system_prompt=(
            "你是数据分析助手。用户提问后，必须先调用 fetch_big_data 工具获取数据，"
            "然后用一句话概括数据总量，不要复述具体内容。"
        ),
        checkpointer=InMemorySaver(),
    )
    # create_deep_agent 内部默认构建 FilesystemMiddleware(tool_token_limit_before_evict=20000)，
    # 工具输出超过 20000 token 时自动写入虚拟文件系统并在对话中替换为路径引用 + 前 10 行预览。

    s4 = run_agent(evict_agent, "帮我查一下 LangGraph 的数据", thread_id="t4")
    all_files = list(s4.get("files", {}).keys())
    print(f"  📁 卸载后的 state.files: {all_files or '（未触发卸载——结果未超过阈值）'}")

    # 找到对话历史里被替换成文件引用的那条 ToolMessage
    for msg in s4["messages"]:
        if type(msg).__name__ == "ToolMessage" and msg.name == "fetch_big_data":
            preview = str(msg.content)[:150].replace("\n", " ")
            print(f"  📎 对话历史中的工具结果已被替换为: {preview}...")
            break
    print("\n📌 结论：工具输出超过阈值时自动'写入文件 + 对话里只留路径引用和预览'，")
    print("   这就是 Context Engineering 的核心：上下文永远保持精简，细节按需取回。")


# =====================================================================
# 实验五：CompositeBackend + FilesystemPermission —— 混合路由 + 敏感路径拒绝
# =====================================================================
def exp5() -> None:
    banner("实验五：CompositeBackend 混合路由 + FilesystemPermission 拒绝写敏感路径")

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from deepagents import FilesystemPermission, create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend

    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    os.makedirs(DEMO_DIR, exist_ok=True)

    store = InMemoryStore()

    composite_agent = create_deep_agent(
        model=model,
        backend=CompositeBackend(
            default=FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True),  # 草稿：磁盘
            routes={
                "/memories/": StoreBackend(
                    # 本地 invoke 时 rt.server_info 是 None，必须兜底
                    # ⚠️ 坑：三元表达式要各自成元组。若写成 (("local-user",) if ... else identity,)
                    # 尾逗号作用于整个三元表达式，None 分支会返回 (("local-user",),)
                    # ——元组套元组，运行时报 TypeError: component must be a string
                    namespace=lambda rt: (
                        ("local-user",) if rt.server_info is None else (rt.server_info.user.identity,)
                    ),
                ),
            },
        ),
        permissions=[
            FilesystemPermission(
                operations=["write", "edit"],
                paths=["/policies/**"],
                mode="deny",  # 敏感路径：直接拒绝写入
            ),
        ],
        store=store,
        checkpointer=InMemorySaver(),
        system_prompt=(
            "你是助手。用户让你记录到哪个路径，就用 write_file 写到哪个路径。"
            "如果工具报错，原样告诉用户错误信息。"
        ),
    )

    run_agent(composite_agent, "请把'用户偏好深色主题'记录到 /memories/preferences.txt")
    run_agent(composite_agent, "请把'临时草稿'记录到 /tmp-note.md")
    run_agent(composite_agent, "请把'公司密码'记录到 /policies/secret.txt")

    print("\n  各后端的落点验证：")
    print(f"  - /tmp-note.md 在磁盘: {os.path.exists(os.path.join(DEMO_DIR, 'tmp-note.md'))}")
    mem_items = store.search(("local-user",))
    print(f"  - Store(/memories/) 中的条目: {[i.key for i in mem_items] or '（空）'}")
    print(f"  - /policies/ 目录在磁盘: {os.path.exists(os.path.join(DEMO_DIR, 'policies'))}")
    print("\n📌 结论：CompositeBackend 按路径前缀路由到不同后端（草稿走磁盘、/memories/ 走持久 Store），")
    print("   FilesystemPermission 在工具调用前拦截敏感路径（first-match-wins，未命中默认放行）。")


if __name__ == "__main__":
    for n, fn in enumerate((exp1, exp2, exp3, exp4, exp5), start=1):
        if should_run(n):
            fn()
    print("\n全部实验完成 ✅")
