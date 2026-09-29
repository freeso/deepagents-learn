# 第 8 章实战：长期记忆 —— 让 Agent 拥有跨对话的记忆
#
# 三个实验（全程只用 InMemoryStore，无磁盘文件、免费 Qwen、结果确定性可验证）：
#   实验一：跨对话记忆全流程 + 用户隔离
#           对话1 记偏好 → 直接查 Store 验货（不信模型嘴）→ 全新 thread 对话2
#           系统提示词里自动出现偏好（memory= 注入铁证）→ 换 user_id 对照组看不到（隔离）
#   实验二：路径路由对照 + 追加不覆盖
#           /workspace/ 临时文件换线程就消失（StateBackend）vs /memories/ 跨线程还在（StoreBackend）；
#           第二条偏好追加后第一条不丢（edit_file 保留旧内容）
#   实验三：Agent 级共享 + 组织级只读
#           namespace 去掉 user_id 后不同用户共享同一份记忆（对比实验一的隔离）；
#           /policies/** deny（改合规政策被拦）+ 用户偏好写入放行（对照组）
#
# 运行：source .venv/bin/activate && python ch08-memory-experiments.py
# 可只跑部分实验：python ch08-memory-experiments.py 1 2
# 依赖环境变量：SILICONFLOW_API_KEY（可选 MODEL_NAME，默认 Qwen/Qwen2.5-7B-Instruct）

import re
import sys

ONLY = {a for a in sys.argv[1:] if a.isdigit()}


def should_run(n: int) -> bool:
    return not ONLY or str(n) in ONLY


def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


import os  # noqa: E402 （放在 load_env_file 定义后统一 import，保持与 ch07 相同风格）

load_env_file(os.path.expanduser("~/research_deepagent/.env"))

from dataclasses import dataclass  # noqa: E402

from langchain.agents.middleware import AgentMiddleware  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from deepagents import FilesystemPermission, create_deep_agent  # noqa: E402
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend  # noqa: E402
from deepagents.backends.utils import create_file_data  # noqa: E402
from deepagents.middleware.memory import MemoryMiddleware  # noqa: E402
from deepagents.profiles.harness.harness_profiles import (  # noqa: E402
    GeneralPurposeSubagentProfile,
    HarnessProfileConfig,
    register_harness_profile,
)

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

# 弱模型防护：关掉默认通用子 Agent，避免它动不动委派把实验搞乱（第 7 章同款）
register_harness_profile(
    "openai",
    HarnessProfileConfig(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_agent(agent, question: str, thread_id: str, context) -> dict:
    """流式运行一轮对话并打印动作；返回 get_state 的 values。"""
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
        context=context,
        config={"configurable": {"thread_id": thread_id}},
    ):
        for node_name, node_update in chunk.items():
            for msg in (node_update or {}).get("messages", []):
                for tc in getattr(msg, "tool_calls", None) or []:
                    print(f"  ⚙️  [{node_name}] {tc['name']}({str(tc['args'])[:80]})")
                if type(msg).__name__ == "ToolMessage":
                    ok = "✅" if "rror" not in str(msg.content)[:80] and "not found" not in str(msg.content)[:80] else "⛔"
                    print(f"  {ok} [{node_name}] {msg.name} -> {str(msg.content)[:70]}")
    return agent.get_state({"configurable": {"thread_id": thread_id}}).values


def tool_calls_of(values: dict) -> list[str]:
    """收集本轮全部工具调用名（按顺序）。"""
    names = []
    for m in values.get("messages", []):
        for tc in getattr(m, "tool_calls", None) or []:
            names.append(tc["name"])
    return names


def final_answer(values: dict) -> str:
    """取最后一条 AI 回复文本。"""
    for m in reversed(values.get("messages", [])):
        if type(m).__name__ == "AIMessage" and getattr(m, "content", ""):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


# ⚠️ 沿用第 7 章的关键发现：memory 注入也是 wrap_model_call 请求时临时拼接，
# 不持久化进 state 消息。且源码证实：用户自定义中间件被插在"核心栈之后、
# memory 尾巴之前"——若把 memory= 交给 create_deep_agent，偷拍员在 memory 外层，
# 拍不到注入结果。对策：手动构造 MemoryMiddleware 放在 CaptureSystemPrompt 前面
#（链上先外后内：memory 先改，capture 后拍）。
captured: dict[str, str] = {}


class CaptureSystemPrompt(AgentMiddleware):
    """截获每次模型调用实际收到的系统提示词（验证 memory 自动加载用）。"""

    def wrap_model_call(self, request, handler):
        captured["system"] = (
            request.system_message.content
            if isinstance(request.system_message.content, str)
            else str(request.system_message.content)
        )
        return handler(request)

    async def awrap_model_call(self, request, handler):
        return self.wrap_model_call(request, handler)


@dataclass
class UserContext:
    """本地调试用身份上下文（部署到 Server 时应改读 server_info）。"""

    user_id: str


# 实验里用的两条偏好（字符串写死 → 落没落盘、进没进提示词，一眼可辨）
PREF_A = "代码注释用中文"
PREF_B = "变量名用英文"

EMPTY_PREFS = "# 用户偏好\n（暂无记录）"

# 写入约定（弱模型硬性关卡）：先读 → edit_file 追加 → 成功后才说"已记住"
SYS_RULES = (
    "你是编程小助手，帮用户写简单的 Python 函数。\n"
    "当用户提出'记住'类请求时，必须严格执行以下流程：\n"
    "1. 先用 read_file 读取 /memories/preferences.md；\n"
    "2. 再用 edit_file 把新偏好追加到该文件末尾（保留已有内容，"
    "不要删除旧偏好，不要新建文件）；\n"
    "3. 第 2 步工具调用成功后，才允许回复'已记住'。\n"
    "禁止：用 write_file 覆盖整个偏好文件；没有调用工具就回复'已记住'。"
)


def seed_file(store: InMemoryStore, ns: tuple, key: str, content: str) -> None:
    """预置记忆文件（memory= 是读取配置，文件必须先存在，缺失会被静默跳过）。"""
    store.put(ns, key, create_file_data(content))


def stored_text(store: InMemoryStore, ns: tuple, key: str) -> str | None:
    """直接开仓验货：从 Store 读文件内容（不信模型嘴上的'已记住'）。"""
    item = store.get(ns, key)
    if item is None:
        return None
    return item.value.get("content")


def check(label: str, ok: bool) -> None:
    print(f"  {'✅' if ok else '❌'} {label}")


# =====================================================================
# 实验一：跨对话记忆全流程 + 用户隔离
# =====================================================================
def exp1() -> None:
    banner("实验一：跨对话记忆 —— 对话1存偏好，全新对话2自动想起来")

    store = InMemoryStore()
    ns_123 = ("user-123", "memories")
    ns_456 = ("user-456", "memories")
    seed_file(store, ns_123, "/preferences.md", EMPTY_PREFS)
    seed_file(store, ns_456, "/preferences.md", EMPTY_PREFS)  # 对照组也预置，排除"没文件"干扰

    composite = CompositeBackend(
        default=StateBackend(),
        routes={
            # namespace 带 user_id → 用户级记忆：A 的偏好 B 看不见
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.context.user_id, "memories"),
                store=store,
            ),
        },
    )
    agent = create_deep_agent(
        model=model,
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        backend=composite,
        middleware=[
            # 手动构造并放在 capture 外层：注入发生在链的更内侧，capture 才拍得到
            MemoryMiddleware(backend=composite, sources=["/memories/preferences.md"]),
            CaptureSystemPrompt(),
        ],
        system_prompt=SYS_RULES,
    )

    # ---- 对话 1：让 Agent 记住偏好 ----
    print(">>> 对话 1（thread-A，user-123）：记住偏好")
    v1 = run_agent(
        agent,
        f"请记住我的偏好：{PREF_A}，{PREF_B}",
        "exp1-write",
        UserContext(user_id="user-123"),
    )
    print(f"  本轮工具调用: {tool_calls_of(v1)}")

    print("\n  🔍 直接查 Store 验货（namespace=('user-123','memories')）：")
    saved = stored_text(store, ns_123, "/preferences.md")
    print(f"  文件内容: {saved!r}")
    check("偏好确实写进了 Store（不是模型嘴上说说）", saved is not None and PREF_A in saved and PREF_B in saved)

    keys = sorted(it.key for it in store.search(ns_123))
    print(f"  📌 Store 里的真实 key: {keys} ← 路由前缀 /memories/ 被 CompositeBackend 剥掉了")

    # ---- 对话 2：全新 thread，问一个不相干的问题 ----
    print("\n>>> 对话 2（thread-B 全新线程，user-123）：写个函数，看它是否记得偏好")
    v2 = run_agent(
        agent,
        "帮我写一个两数相加的 Python 函数",
        "exp1-read",
        UserContext(user_id="user-123"),
    )
    sys2 = captured.get("system", "")
    check("新线程的系统提示词里自动出现偏好（memory= 注入，Agent 没调 read_file）",
          PREF_A in sys2 and "<agent_memory>" in sys2)
    ans2 = final_answer(v2)
    has_code = "def " in ans2
    has_cn_comment = bool(re.search(r"#\s*[^\n]*[\u4e00-\u9fff]", ans2))
    print(f"  回复含函数定义: {has_code}；含中文注释（遵循偏好，弱模型可能跳过）: {has_cn_comment}")

    # ---- 对照组：换一个 user_id ----
    print("\n>>> 对照组（thread-C 全新线程，user-456）：同样的问题")
    v3 = run_agent(
        agent,
        "帮我写一个两数相加的 Python 函数",
        "exp1-other",
        UserContext(user_id="user-456"),
    )
    sys3 = captured.get("system", "")
    check("user-456 的系统提示词里没有 user-123 的偏好（用户级隔离）",
          PREF_A not in sys3 and "<agent_memory>" in sys3)

    print("\n📌 结论：/memories/ 路由到 StoreBackend 后，偏好跨线程存活；")
    print("   memory= 在每场新对话开始时自动把内容注入系统提示词（模型无需主动读）；")
    print("   namespace 里的 user_id 是隔离边界——换人即失忆。")


# =====================================================================
# 实验二：路径路由对照 + 追加不覆盖
# =====================================================================
def exp2() -> None:
    banner("实验二：临时桌面 vs 长期档案柜 —— 路由对照与追加不覆盖")

    store = InMemoryStore()
    ns = ("user-789", "memories")
    seed_file(store, ns, "/preferences.md", EMPTY_PREFS)

    composite = CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.context.user_id, "memories"),
                store=store,
            ),
        },
    )
    agent = create_deep_agent(
        model=model,
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        backend=composite,
        middleware=[
            MemoryMiddleware(backend=composite, sources=["/memories/preferences.md"]),
            CaptureSystemPrompt(),
        ],
        system_prompt=SYS_RULES,
    )
    user = UserContext(user_id="user-789")

    # ---- R1：写临时文件 + 记偏好 A（弱模型复合指令会吃掉一半，拆成两句）----
    print(">>> R1（thread-t1）：写 /workspace/draft.txt，再记住偏好A")
    run_agent(
        agent,
        "请用 write_file 把字符串'一次性草稿'写入 /workspace/draft.txt，写完告诉我文件路径。",
        "exp2-t1",
        user,
    )
    v1 = run_agent(agent, f"请记住我的偏好：{PREF_A}", "exp2-t1", user)
    files_t1 = v1.get("files") or {}
    print(f"  t1 状态里的文件: {sorted(files_t1)}")
    check("临时草稿进了 StateBackend（随对话存亡的'桌面'）", "/workspace/draft.txt" in files_t1)
    check("偏好A写进了 Store（'档案柜'）", PREF_A in (stored_text(store, ns, "/preferences.md") or ""))

    # ---- R2：全新线程追加偏好 B ----
    print("\n>>> R2（thread-t2 全新线程）：再记住偏好B")
    v2 = run_agent(agent, f"再记住一条偏好：{PREF_B}", "exp2-t2", user)
    content = stored_text(store, ns, "/preferences.md") or ""
    print(f"  文件内容: {content!r}")
    check("追加成功且偏好A没丢（edit_file 保留旧内容）", PREF_A in content and PREF_B in content)
    used_overwrite = "write_file" in tool_calls_of(v2)
    print(f"  ⚠️ 本轮是否误用 write_file 整体覆盖: {used_overwrite}（应为 False）")
    sys2 = captured.get("system", "")
    check("t2 开场系统提示词已含偏好A（Store → memory= 的跨线程回路打通）", PREF_A in sys2)

    # ---- R3：全新线程找临时文件 ----
    print("\n>>> R3（thread-t3 全新线程）：去找那份'一次性草稿'")
    v3 = run_agent(
        agent,
        "请用 read_file 读取 /workspace/draft.txt 并原样告诉我内容",
        "exp2-t3",
        user,
    )
    files_t3 = v3.get("files") or {}
    check("新线程状态里没有 draft.txt（StateBackend 线程隔离，'下班清理工位'）",
          "/workspace/draft.txt" not in files_t3)
    check("偏好文件不受影响（Store 跨线程仍在）",
          PREF_A in (stored_text(store, ns, "/preferences.md") or ""))

    print("\n📌 结论：同一个 write_file 工具，路径前缀决定命运——")
    print("   /workspace/ 走 StateBackend（对话结束即消失），/memories/ 走 StoreBackend（跨对话长存）。")
    print("   追加记忆必须用 edit_file；write_file 会整体覆盖（弱模型高发错误）。")


# =====================================================================
# 实验三：Agent 级共享 + 组织级只读
# =====================================================================
def exp3() -> None:
    banner("实验三：记忆的作用域 —— Agent 级共享与组织级只读")

    # ---- A 部分：namespace 去掉 user_id → 所有用户共享同一份记忆 ----
    print(">>> A. Agent 级记忆（namespace 固定，不含 user_id）")
    store_a = InMemoryStore()
    ns_shared = ("shared-agent", "memories")
    seed_file(store_a, ns_shared, "/preferences.md", EMPTY_PREFS)
    composite_a = CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(namespace=lambda rt: ("shared-agent", "memories"), store=store_a),
        },
    )
    agent_a = create_deep_agent(
        model=model,
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        backend=composite_a,
        middleware=[
            MemoryMiddleware(backend=composite_a, sources=["/memories/preferences.md"]),
            CaptureSystemPrompt(),
        ],
        system_prompt=SYS_RULES,
    )
    run_agent(agent_a, "请记住我的偏好：回答尽量简短", "exp3a-1", UserContext(user_id="user-111"))
    v2 = run_agent(agent_a, "根据你的记忆，我的偏好是什么？", "exp3a-2", UserContext(user_id="user-222"))
    sys_a2 = captured.get("system", "")
    check("user-222 的提示词里出现 user-111 写的偏好（Agent 级 = 全用户共享）",
          "回答尽量简短" in sys_a2)
    print(f"  user-222 的回答摘要: {final_answer(v2)[:60]}...")

    # ---- B 部分：组织级记忆 + 只读保护 ----
    print("\n>>> B. 组织级 /policies/（全员共享，deny 写入）")
    POLICY = "# 合规政策\n- 所有投资建议必须附带免责声明\n- 不得披露内部定价\n"
    store_b = InMemoryStore()
    ns_user = ("user-333", "memories")
    ns_org = ("org-acme", "policies")
    seed_file(store_b, ns_user, "/preferences.md", EMPTY_PREFS)
    seed_file(store_b, ns_org, "/compliance.md", POLICY)

    composite_b = CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.context.user_id, "memories"), store=store_b
            ),
            "/policies/": StoreBackend(namespace=lambda rt: ("org-acme", "policies"), store=store_b),
        },
    )
    agent_b = create_deep_agent(
        model=model,
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        backend=composite_b,
        permissions=[FilesystemPermission(operations=["write", "edit", "delete"], paths=["/policies/**"], mode="deny")],
        middleware=[
            MemoryMiddleware(
                backend=composite_b,
                sources=["/memories/preferences.md", "/policies/compliance.md"],
            ),
            CaptureSystemPrompt(),
        ],
        system_prompt=SYS_RULES,
    )
    user333 = UserContext(user_id="user-333")

    print("  B0. 先确认合规政策被自动加载：")
    run_agent(agent_b, "公司的合规政策对投资建议有什么要求？", "exp3b-0", user333)
    sys_b0 = captured.get("system", "")
    check("组织政策出现在系统提示词里（memory= 同时加载多个文件）",
          "免责声明" in sys_b0 and "/policies/compliance.md" in sys_b0)

    print("\n  B1. 试图篡改合规政策（应被 deny 拦截）：")
    print("  （⚠️ 首轮跑时弱模型把路径幻觉成 /memories/policies/...，压根没碰 /policies/**，")
    print("     deny 根本没触发——'文件没动'是假阳性。必须在指令里写死正确路径。）")
    vb1 = run_agent(
        agent_b,
        "请用 edit_file 把 /policies/compliance.md 里'- 不得披露内部定价'这一行整行删除。",
        "exp3b-1",
        user333,
    )
    policy_now = stored_text(store_b, ns_org, "/compliance.md") or ""
    check("政策文件原文未动（FilesystemPermission deny 生效）", policy_now == POLICY)
    denied_msgs = [
        str(m.content)[:80]
        for m in vb1.get("messages", [])
        if type(m).__name__ == "ToolMessage" and ("denied" in str(m.content).lower() or "权限" in str(m.content))
    ]
    if denied_msgs:
        print(f"  deny 报错（保真证据）: {denied_msgs[0]}")
        print(f"  ⚠️ 共被拦 {len(denied_msgs)} 次——弱模型被 deny 后不放弃，反复'读→试改→被拦'循环")
        print("     （deny 是硬墙拦得住，但费轮次；第 9 章的 interrupt 审批模式更适合敏感路径）")
    else:
        print("  deny 报错信息: （未观察到，需检查）")

    print("\n  B2. 对照组：普通偏好写入（应放行）：")
    run_agent(agent_b, "请记住我的偏好：回复保持简短", "exp3b-2", user333)
    check("deny 只锁 /policies/**，/memories/ 写入不受影响",
          "回复保持简短" in (stored_text(store_b, ns_user, "/preferences.md") or ""))

    print("\n📌 结论：namespace 决定'谁能看'——去掉 user_id 即 Agent 级共享；")
    print("   组织级记忆全员可读但设为只读，防止某个用户借 Agent 之手注入恶意指令。")


# =====================================================================
if __name__ == "__main__":
    if should_run(1):
        exp1()
    if should_run(2):
        exp2()
    if should_run(3):
        exp3()
    if not ONLY:
        print("\n（提示：可传实验编号单跑，如 python ch08-memory-experiments.py 1）")
