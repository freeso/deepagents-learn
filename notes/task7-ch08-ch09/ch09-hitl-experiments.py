# 第 9 章实战：Human-in-the-Loop —— 给 Agent 拴上安全绳
#
# 三个实验（免费 Qwen + 确定性"执行日志"验证，HITL 中断/恢复走 version="v2"）：
#   实验一：四种决策四连（approve / reject / edit / respond）
#           用"执行日志"做铁证：哪个工具真执行了、参数是原始的还是改过的，
#           不读模型嘴。reject 的反馈 message、respond 的"人肉工具返回值"逐一验证。
#   实验二：批量打包 + when 条件中断
#           一条指令三个敏感操作 → 审批循环（不管模型是打包成一批还是拆成多轮）；
#           write_file 挂 when 谓词：/workspace/ 内放行、/secrets/ 外才暂停。
#   实验三：文件权限 mode="interrupt" + 子 Agent 独立审批
#           /secrets/** 写入弹审批（approve 落盘 vs reject 不落盘）；
#           同一个 list_files 工具，主 Agent 调不拦、file-manager 子 Agent 调必拦。
#
# 运行：source .venv/bin/activate && python ch09-hitl-experiments.py
# 可只跑部分实验：python ch09-hitl-experiments.py 1 3
# 依赖环境变量：SILICONFLOW_API_KEY（可选 MODEL_NAME，默认 Qwen/Qwen2.5-7B-Instruct）
#
# ⚠️ HITL 三必须：Checkpointer 必配、恢复必须同 thread_id、invoke 必须 version="v2"

import os
import sys
import uuid

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


load_env_file(os.path.expanduser("~/research_deepagent/.env"))

from langchain.tools import tool  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from deepagents import FilesystemPermission, create_deep_agent  # noqa: E402
from deepagents.backends import FilesystemBackend  # noqa: E402
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

# 弱模型防护：关掉默认通用子 Agent（第 7/8 章同款），实验里只留我们自己定义的
register_harness_profile(
    "openai",
    HarnessProfileConfig(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp", "hitl-demo")


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check(label: str, ok: bool) -> None:
    print(f"  {'✅' if ok else '❌'} {label}")


# ---------------------------------------------------------------------
# HITL 基础工具：invoke 走 v2，返回 GraphOutput（.interrupts / .value）
# ---------------------------------------------------------------------
def invoke_v2(agent, payload, config):
    return agent.invoke(payload, config=config, version="v2")


def values_of(result):
    return result.value if hasattr(result, "value") else result


def final_answer(values) -> str:
    for m in reversed(values.get("messages", [])):
        if type(m).__name__ == "AIMessage":
            c = getattr(m, "content", "")
            if c:
                return c if isinstance(c, str) else str(c)
    return ""


def tool_messages_of(values, name: str) -> list[str]:
    """收集指定工具的全部 ToolMessage 文本（restore 后的执行/拒绝/人答反馈）。"""
    out = []
    for m in values.get("messages", []):
        if type(m).__name__ == "ToolMessage" and getattr(m, "name", "") == name:
            out.append(m.content if isinstance(m.content, str) else str(m.content))
    return out


def show_interrupts(result, tag: str = "") -> list[dict]:
    """打印中断内容（工具+参数+可选决策），返回 action_requests。"""
    if not result.interrupts:
        return []
    v = result.interrupts[0].value
    ars, cfgs = v["action_requests"], {c["action_name"]: c for c in v["review_configs"]}
    for a in ars:
        allowed = cfgs.get(a["name"], {}).get("allowed_decisions", "?")
        print(f"  ⏸️  {tag}待审批: {a['name']}({a['args']}) 可选决策: {allowed}")
    return ars


# =====================================================================
# 执行日志：确定性验证的锚点——工具真执行了什么，全记在这里，不读模型嘴
# =====================================================================
EXECUTED_LOG: list[tuple] = []


@tool
def delete_file(path: str) -> str:
    """删除指定路径的文件（模拟，实际只记日志）。"""
    EXECUTED_LOG.append(("delete_file", path))
    return f"已删除 {path}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（模拟，实际只记日志）。"""
    EXECUTED_LOG.append(("send_email", to, subject))
    return f"邮件已发送至 {to}"


@tool
def ask_user(question: str) -> str:
    """向用户提问。工具本身不会执行，真实回答由人工审批的 respond 决策提供。"""
    return "等待用户回答"


@tool
def get_phone_specs(brand: str) -> dict:
    """查询指定品牌手机的参数（低风险只读工具，不需要审批）。"""
    db = {"Aurora": {"price_cny": 4999}, "Nimbus": {"price_cny": 3599}}
    return db.get(brand, {"error": f"没有找到 {brand}"})


def ran(name: str, *kw) -> bool:
    """检查执行日志里是否有按关键参数匹配的记录。"""
    return any(entry[0] == name and all(k in entry for k in kw) for entry in EXECUTED_LOG)


# =====================================================================
# 实验一：四种决策四连 —— approve / reject / edit / respond
# =====================================================================
def exp1() -> None:
    banner("实验一：四种决策 —— 人插手的四种姿势")

    agent = create_deep_agent(
        model=model,
        tools=[delete_file, send_email, ask_user, get_phone_specs],
        interrupt_on={
            # 风险分层（课程最佳实践）：删除/邮件高风险（可改参数），提问型只 respond，只读不拦
            "delete_file": {"allowed_decisions": ["approve", "edit", "reject"]},
            "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
            "ask_user": {"allowed_decisions": ["respond"]},
            "get_phone_specs": False,
        },
        checkpointer=InMemorySaver(),  # HITL 必须
        system_prompt=(
            "你是运维助手。任何操作必须真实调用对应工具完成，"
            "禁止在未调用工具的情况下声称已完成操作。"
            "删除文件必须调用 delete_file，发邮件必须调用 send_email，"
            "向用户确认信息必须调用 ask_user。"
        ),
    )

    # ---- 轮 1：approve ——"就按你说的办" ----
    print(">>> 轮 1 approve：删除临时文件")
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    r = invoke_v2(agent, {"messages": [{"role": "user", "content": "请调用 delete_file 工具删除 /data/old-temp.txt"}]}, cfg)
    if show_interrupts(r, "轮1 "):
        r = invoke_v2(agent, Command(resume={"decisions": [{"type": "approve"}]}), cfg)
        check("approve 后工具真执行了（日志有记录）", ran("delete_file", "/data/old-temp.txt"))
        check("恢复后正常拿到最终回复", bool(final_answer(values_of(r))))
    else:
        check("轮 1 发生了中断（模型未嘴上答应）", False)

    # ---- 轮 2：reject ——"别办了，原因是……" ----
    print("\n>>> 轮 2 reject：删除核心数据库（带反馈 message）")
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    r = invoke_v2(agent, {"messages": [{"role": "user", "content": "请调用 delete_file 工具删除 /data/important.db"}]}, cfg)
    if show_interrupts(r, "轮2 "):
        r = invoke_v2(agent, Command(resume={"decisions": [{
            "type": "reject",
            "message": "用户拒绝删除：这是核心数据库。不要重试删除，请建议改为先备份。",
        }]}), cfg)
        check("reject 后工具没执行（日志无记录）", not ran("delete_file", "/data/important.db"))
        rej_msgs = tool_messages_of(values_of(r), "delete_file")
        check("拒绝原因作为反馈回到模型（ToolMessage 含 User rejected）",
              any("User rejected" in m for m in rej_msgs))
        print(f"  拒绝反馈原文: {rej_msgs[0][:70] if rej_msgs else '（无）'}...")
    else:
        check("轮 2 发生了中断（模型未嘴上答应）", False)

    # ---- 轮 3：edit ——"改成这样再办" ----
    print("\n>>> 轮 3 edit：发给老板的邮件，人工改收件人为团队邮箱")
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    r = invoke_v2(agent, {"messages": [{"role": "user", "content": "请把周报用邮件发给 boss@example.com，主题写'周报'，正文写'见附件'"}]}, cfg)
    ars = show_interrupts(r, "轮3 ")
    if ars:
        original = dict(ars[0]["args"])
        edited = {**original, "to": "team@example.com"}  # 只改收件人，保守修改
        r = invoke_v2(agent, Command(resume={"decisions": [{
            "type": "edit",
            "edited_action": {"name": "send_email", "args": edited},
        }]}), cfg)
        check(f"edit 后按改后的参数执行（发给了 {edited['to']} 而非 {original.get('to')}）",
              ran("send_email", "team@example.com") and not ran("send_email", "boss@example.com"))
    else:
        check("轮 3 发生了中断", False)

    # ---- 轮 4：respond ——"工具不用跑了，我亲自回答" ----
    print("\n>>> 轮 4 respond：ask_user 问报表粒度，人工代答")
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    r = invoke_v2(agent, {"messages": [{"role": "user", "content": "我想做一份手机行情分析报告，时间粒度用按月还是按季度？请先用 ask_user 工具问我。"}]}, cfg)
    show_interrupts(r, "轮4 ")
    answer = "按季度统计，并排除测试数据。"
    r = invoke_v2(agent, Command(resume={"decisions": [{"type": "respond", "message": answer}]}), cfg)
    resp_msgs = tool_messages_of(values_of(r), "ask_user")
    check("respond 的话成了 ask_user 的成功返回值", answer in (resp_msgs[0] if resp_msgs else ""))
    check("最终回复采纳了人的答案（含'季度'）", "季度" in final_answer(values_of(r)))

    print("\n📌 结论：四种决策语义各异——approve 原样执行、reject 不执行且原因回传给模型、")
    print("   edit 按改后参数执行（收件人被换掉）、respond 是'人肉工具返回值'。")
    print("   全部验证来自执行日志和 ToolMessage，与模型嘴上说的无关。")


# =====================================================================
# 实验二：批量打包 + when 条件中断
# =====================================================================
def exp2() -> None:
    banner("实验二：一条指令三个敏感操作 + when 条件放行")

    import shutil

    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    os.makedirs(DEMO_DIR, exist_ok=True)

    def writes_outside_allowed(request) -> bool:
        """when 谓词：/workspace/ 内的写入放行，其他路径才暂停审批。"""
        path = request.tool_call["args"].get("file_path", "")
        return not path.startswith("/workspace/")

    agent = create_deep_agent(
        model=model,
        tools=[delete_file, send_email],
        backend=FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True),
        interrupt_on={
            "delete_file": {"allowed_decisions": ["approve", "reject"]},
            "write_file": {"allowed_decisions": ["approve", "edit", "reject"], "when": writes_outside_allowed},
            "send_email": {"allowed_decisions": ["approve", "reject"]},
        },
        checkpointer=InMemorySaver(),
        system_prompt=(
            "你是运维助手，必须严格按顺序操作：用户要求删除文件时，"
            "先逐个调用 delete_file 完成每一次删除（系统会逐次审批），"
            "全部删除尝试结束后才允许调用 send_email 发通知邮件。"
            "禁止调换顺序，禁止在邮件正文里声称删除了没有实际调用工具删除的文件。"
            "写文件用 write_file 工具。"
        ),
    )

    # ---- Round A：审批循环（不依赖模型的打包行为） ----
    print(">>> A. 删除 junk1 + junk2、给 ops 发邮件通知（junk1 放行、junk2 拒绝、邮件放行）")
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    r = invoke_v2(agent, {"messages": [{"role": "user", "content": "请删除 /data/junk1.txt 和 /data/junk2.txt，然后给 ops@example.com 发邮件通知清理完成，主题'清理通知'。"}]}, cfg)

    rounds, packed_sizes = 0, []
    budget = 12  # 防死循环
    while getattr(r, "interrupts", None) and budget > 0:
        budget -= 1
        rounds += 1
        ars = show_interrupts(r, f"A{rounds} ")
        packed_sizes.append(len(ars))
        decisions = []
        for a in ars:
            name, args = a["name"], a["args"]
            if name == "delete_file" and args.get("path") == "/data/junk1.txt":
                decisions.append({"type": "approve"})
            elif name == "delete_file":
                decisions.append({"type": "reject", "message": "该文件仍在使用，禁止删除，不要重试。"})
            elif name == "send_email":
                decisions.append({"type": "approve"})
            elif name == "write_file":
                decisions.append({"type": "approve"})
            else:
                decisions.append({"type": "approve"})
        r = invoke_v2(agent, Command(resume={"decisions": decisions}), cfg)
    print(f"  📊 共触发 {rounds} 次中断，每次打包的工具数: {packed_sizes}（弱模型可能拆步、强模型会打包）")
    check("junk1 已删除", ran("delete_file", "/data/junk1.txt"))
    check("junk2 未删除（被拒且没有重试）", not ran("delete_file", "/data/junk2.txt"))
    check("邮件正常发送", ran("send_email", "ops@example.com"))

    # ---- Round B：when 条件中断 ----
    print("\n>>> B1. 写 /workspace/note.txt（when 谓词应放行，零打扰）")
    print("  （⚠️ 实测发现：弱模型写完文件后会主动加戏——给用户发'操作通知'邮件，")
    print("     触发 send_email 审批。白名单只精准放行 write_file，加戏动作照样被拦。）")
    cfg_b1 = {"configurable": {"thread_id": str(uuid.uuid4())}}
    r1 = invoke_v2(agent, {"messages": [{"role": "user", "content": "请用 write_file 把'普通笔记'写入 /workspace/note.txt，不要发任何邮件。"}]}, cfg_b1)
    ws_interrupted, budget = False, 4
    while getattr(r1, "interrupts", None) and budget > 0:
        budget -= 1
        ars = show_interrupts(r1, "B1 ")
        # 语义正确的断言：/workspace/ 内的 write_file 调用绝不允许进中断
        ws_interrupted = ws_interrupted or any(
            a["name"] == "write_file" and str(a["args"].get("file_path", "")).startswith("/workspace/")
            for a in ars
        )
        # 模型加戏的额外操作（如自发通知邮件）：一律拒绝，促使其收敛收尾
        r1 = invoke_v2(agent, Command(resume={"decisions": [
            {"type": "reject", "message": "不需要此额外操作，请直接汇报原任务结果。"} for _ in ars
        ]}), cfg_b1)
    check("工作区内写入零打扰（该 write_file 从未进中断）", not ws_interrupted)
    check("文件确实落盘", os.path.exists(os.path.join(DEMO_DIR, "workspace", "note.txt")))

    print("\n>>> B2. 写 /secrets/cred.json（when 谓词应拦截）")
    cfg_b2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
    r2 = invoke_v2(agent, {"messages": [{"role": "user", "content": "请用 write_file 把'模拟密钥 mk-123'写入 /secrets/cred.json"}]}, cfg_b2)
    show_interrupts(r2, "B2 ")
    r2 = invoke_v2(agent, Command(resume={"decisions": [{
        "type": "reject",
        "message": "密钥禁止落盘。不要写入 /secrets/，如需保存请存到 /workspace/ 并注明是测试数据。",
    }]}), cfg_b2)
    check("/secrets/cred.json 未落盘（reject 生效）", not os.path.exists(os.path.join(DEMO_DIR, "secrets", "cred.json")))
    fallback_written = os.path.exists(os.path.join(DEMO_DIR, "workspace", "cred.json"))
    print(f"  （观察）模型听劝改存 /workspace/cred.json: {fallback_written}（reject message 里给了替代方案）")

    print("\n📌 结论：审批循环对'打包一批'和'拆成多轮'都健壮（decisions 与 action_requests 对号）；")
    print("   when 谓词实现'低风险零打扰、高风险必拦'——工作区直通、越界弹审批。")
    print("   reject 的 message 写清'下一步怎么办'，模型才不会像撞 deny 墙那样反复重试。")


# =====================================================================
# 实验三：文件权限 mode="interrupt" + 子 Agent 独立审批
# =====================================================================
def exp3() -> None:
    banner("实验三：文件权限中断 + 子 Agent 独立审批（同一工具两套安全级别）")

    import shutil

    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # ---- A 部分：FilesystemPermission mode="interrupt" ----
    print(">>> A. /secrets/** 写入弹审批（第 8 章 deny 的亲兄弟：硬墙 → 弹窗）")
    agent = create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True),
        permissions=[FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="interrupt")],
        checkpointer=InMemorySaver(),
        system_prompt="你是运维助手，写文件用 write_file 工具。",
    )
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    r = invoke_v2(agent, {"messages": [{"role": "user", "content": "请用 write_file 把'mk-456'写入 /secrets/token.json"}]}, cfg)
    show_interrupts(r, "A ")
    r = invoke_v2(agent, Command(resume={"decisions": [{"type": "approve"}]}), cfg)
    check("approve 后真的落盘", open(os.path.join(DEMO_DIR, "secrets", "token.json"), encoding="utf-8").read() == "mk-456")

    print("\n>>> A2. 换一份密钥写（这次 reject）")
    cfg2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
    r2 = invoke_v2(agent, {"messages": [{"role": "user", "content": "请用 write_file 把'mk-789'写入 /secrets/token2.json"}]}, cfg2)
    show_interrupts(r2, "A2 ")
    r2 = invoke_v2(agent, Command(resume={"decisions": [{"type": "reject", "message": "禁止再写入任何密钥文件。"}]}), cfg2)
    check("reject 后没有落盘", not os.path.exists(os.path.join(DEMO_DIR, "secrets", "token2.json")))
    check("前一次 approve 的文件不受影响", os.path.exists(os.path.join(DEMO_DIR, "secrets", "token.json")))

    # ---- B 部分：子 Agent 独立 interrupt_on ----
    print("\n>>> B. 同一个 list_files：主 Agent 调不拦，file-manager 子 Agent 调必拦")

    @tool
    def list_files() -> str:
        """列出手头的文件清单。"""
        EXECUTED_LOG.append(("list_files",))
        return "文件清单：a.txt、b.txt、c.txt"

    agent_b = create_deep_agent(
        model=model,
        tools=[list_files],
        interrupt_on={"list_files": False},  # 主 Agent 自己调：低风险直通
        subagents=[{
            "name": "file-manager",
            "description": "盘点、整理手头的文件清单时委派给它",
            "system_prompt": "你是文件管理员，收到任务后必须立刻调用 list_files 盘点并汇报清单。",
            "tools": [list_files],
            "interrupt_on": {"list_files": True},  # 子 Agent 调：必须审批
        }],
        checkpointer=InMemorySaver(),
        system_prompt="你是文件管家。凡是盘点文件清单的任务，必须用 task 工具委派给 file-manager 子 Agent。",
    )

    EXECUTED_LOG.clear()
    print("  B1. 委派给子 Agent 盘点：")
    cfg_b1 = {"configurable": {"thread_id": str(uuid.uuid4())}}
    rb1 = invoke_v2(agent_b, {"messages": [{"role": "user", "content": "请让 file-manager 子 Agent 帮我盘点手头的文件清单。"}]}, cfg_b1)
    sub_ars = show_interrupts(rb1, "B1 ")
    check("子 Agent 的调用触发审批（中断上浮到主 invoke）",
          any(a["name"] == "list_files" for a in sub_ars))
    if sub_ars:
        rb1 = invoke_v2(agent_b, Command(resume={"decisions": [{
            "type": "reject", "message": "该文件库涉密，禁止盘点，直接向用户说明无法执行。",
        }]}), cfg_b1)
    check("reject 后子 Agent 的工具没执行", ("list_files",) not in EXECUTED_LOG)

    print("\n  B2. 对照组：主 Agent 自己盘点（不委派）：")
    EXECUTED_LOG.clear()
    cfg_b2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
    rb2 = invoke_v2(agent_b, {"messages": [{"role": "user", "content": "请你亲自直接用 list_files 盘点一下文件清单，不要委派给子 Agent。"}]}, cfg_b2)
    check("主 Agent 自己调同一个工具：没有中断", not rb2.interrupts)
    check("工具直接执行了", ("list_files",) in EXECUTED_LOG)

    print("\n📌 结论：文件权限 mode='interrupt' 与 interrupt_on 合并成同一套中断/恢复流程；")
    print("   子 Agent 的 interrupt_on 独立成篇——同一个工具，主 Agent 免检、子 Agent 必检，")
    print("   中断从子图一路上浮到主 invoke，审批员坐在主控台就能拦下属的动作。")


# =====================================================================
if __name__ == "__main__":
    if should_run(1):
        exp1()
    if should_run(2):
        exp2()
    if should_run(3):
        exp3()
    if not ONLY:
        print("\n（提示：可传实验编号单跑，如 python ch09-hitl-experiments.py 1）")
