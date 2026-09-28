# 第 7 章实战：Skills —— 可复用的 Agent 能力包
#
# 三个实验（Skill 文件由本脚本现场生成到 .tmp/skills-demo/，可重复运行）：
#   实验一：渐进式披露三级加载实测
#           L1 启动只进"书脊"（name+description）→ L2 匹配后才 read_file 正文
#           → L3 正文引用的 references 评分标准被按需读取（分数写死，读没读一眼可辨）
#   实验二：精准匹配 + 同名分层覆盖（last wins）
#   实验三：Skill 只读保护 —— deny /skills/** 写入（改手册被拦）+ 别处写入放行（对照组）
#
# 运行：source .venv/bin/activate && python ch07-skills-experiments.py
# 可只跑部分实验：python ch07-skills-experiments.py 1 2
# 依赖环境变量：SILICONFLOW_API_KEY（可选 MODEL_NAME，默认 Qwen/Qwen2.5-7B-Instruct）

import os
import shutil
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


load_env_file(os.path.expanduser("~/research_deepagent/.env"))

from langchain.agents.middleware import AgentMiddleware  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("缺少 API key：请 export SILICONFLOW_API_KEY=... 后重试")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    temperature=0,
    timeout=120,
    max_retries=3,
)

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp", "skills-demo")


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_agent(agent, question: str, thread_id: str = "t1") -> dict:
    """运行 Agent 并流式打印动作；返回 get_state 的 values（需要 checkpointer）。"""
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
        config={"configurable": {"thread_id": thread_id}},
    ):
        for node_name, node_update in chunk.items():
            for msg in (node_update or {}).get("messages", []):
                for tc in getattr(msg, "tool_calls", None) or []:
                    print(f"  ⚙️  [{node_name}] {tc['name']}({str(tc['args'])[:80]})")
                if type(msg).__name__ == "ToolMessage":
                    ok = "✅" if "Error" not in str(msg.content)[:60] else "⛔"
                    print(f"  {ok} [{node_name}] {msg.name} -> {str(msg.content)[:70]}")
    return agent.get_state({"configurable": {"thread_id": thread_id}}).values


def tool_calls_of(values: dict) -> list[dict]:
    """从消息历史里收集全部工具调用 [{name, args}]。"""
    calls = []
    for m in values.get("messages", []):
        for tc in getattr(m, "tool_calls", None) or []:
            calls.append({"name": tc["name"], "args": tc["args"]})
    return calls


def system_message_of(values: dict) -> str:
    """取系统提示词文本（消息历史的第一条 system 消息）。"""
    for m in values.get("messages", []):
        if type(m).__name__ == "SystemMessage":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


# ⚠️ 重要发现：SkillsMiddleware 的'书脊'是通过 wrap_model_call 在请求时临时注入
# 系统提示词的，不会持久化进 state 消息——get_state 里翻不到！要验证 L1 必须
# 用截获器中间件抓模型实际看到的系统提示词。
captured: dict[str, str] = {}


class CaptureSystemPrompt(AgentMiddleware):
    """截获每次模型调用实际收到的系统提示词（验证 L1 用）。"""

    def wrap_model_call(self, request, handler):
        captured["system"] = (
            request.system_message.content
            if isinstance(request.system_message.content, str)
            else str(request.system_message.content)
        )
        return handler(request)

    async def awrap_model_call(self, request, handler):
        return self.wrap_model_call(request, handler)


# =====================================================================
# 模拟手机参数库（延续前几章）：Skill 正文会指示 Agent 调这个工具
# ⚠️ 坑：Skill 正文里引用的工具必须真的注册（tools=[get_phone_specs]），
# 否则弱模型会报错后幻觉去找不存在的数据文件
# =====================================================================
PHONE_DB: dict[str, dict] = {
    "Aurora": {"price_cny": 4999, "battery_mah": 5000, "weight_g": 201},
    "Nimbus": {"price_cny": 3599, "battery_mah": 6000, "weight_g": 226},
    "Zenith": {"price_cny": 6299, "battery_mah": 4400, "weight_g": 172},
}


def get_phone_specs(brand: str) -> dict:
    """查询指定品牌手机的完整参数（价格/电池/重量）。

    Args:
        brand: 品牌名，可选值：Aurora、Nimbus、Zenith。

    Returns:
        dict：该品牌的完整参数；品牌不存在时返回 {"error": "说明"}。
    """
    data = PHONE_DB.get(brand)
    if data is None:
        return {"error": f"没有找到品牌 {brand}，可选：Aurora、Nimbus、Zenith"}
    return {"brand": brand, **data}


# =====================================================================
# Skill 文件现场生成（两个职责不同的 Skill + 一份参考评分标准）
# =====================================================================
SKILL_ANALYSIS = """---
name: phone-analysis
description: 当用户要求对比、分析手机参数或推荐购买哪款手机时，使用此技能。按固定流程查询参数并输出【手机分析报告】。
---

# 手机参数分析规范

## 执行步骤（必须严格照做）
1. 对用户提到的每一款手机，调用 get_phone_specs 查询参数；
2. 读取评分标准文件 /skills/phone-analysis/references/scoring.md，报告中的分数必须以它为准；
3. 输出报告：第一行必须是【手机分析报告】，随后列出每款手机的得分和一句结论。
禁止：跳过第 2 步直接输出报告。
"""

SKILL_REPORT = """---
name: report-writing
description: 当用户要求撰写技术报告、周报或正式文档时使用此技能。规定报告的结构与用词规范。
---

# 技术报告写作规范

## 格式要求
1. 报告第一行必须是【技术报告】；
2. 使用"背景/分析/结论"三段式结构；
3. 语言正式，不使用口语。
"""

SCORING_REF = """# 评分标准（结论已定，直接引用，不要自己算）

- Aurora 得分 8.5：旗舰均衡型
- Nimbus 得分 7.8：续航之王
- Zenith 得分 9.0：影像旗舰
"""


def build_skill_tree() -> None:
    """生成实验一/三用的标准 Skill 树；实验二再额外生成分层覆盖结构。"""
    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    base = os.path.join(DEMO_DIR, "skills", "phone-analysis")
    os.makedirs(os.path.join(base, "references"), exist_ok=True)
    with open(os.path.join(base, "SKILL.md"), "w") as f:
        f.write(SKILL_ANALYSIS)
    with open(os.path.join(base, "references", "scoring.md"), "w") as f:
        f.write(SCORING_REF)
    rw = os.path.join(DEMO_DIR, "skills", "report-writing")
    os.makedirs(rw, exist_ok=True)
    with open(os.path.join(rw, "SKILL.md"), "w") as f:
        f.write(SKILL_REPORT)


def build_override_tree() -> None:
    """实验二：shared / project 两层各放一个同名 phone-analysis（正文标记不同）。"""
    for layer, marker in (("shared", "【共享版报告】"), ("project", "【项目版报告】")):
        d = os.path.join(DEMO_DIR, "skills", layer, "phone-analysis")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "SKILL.md"), "w") as f:
            f.write(
                "---\n"
                "name: phone-analysis\n"
                "description: 当用户要求对比、分析手机参数或推荐购买哪款手机时，使用此技能。"
                f"输出报告第一行必须是{marker}\n"
                "---\n\n"
                "# 手机参数分析规范\n\n"
                f"输出报告时，第一行必须是{marker}，随后用一句话评价 Aurora 这款手机。\n"
                "不需要查询任何数据，直接输出即可。\n"
            )


from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from deepagents import (  # noqa: E402
    FilesystemPermission,
    GeneralPurposeSubagentProfile,
    create_deep_agent,
)
from deepagents.backends import FilesystemBackend  # noqa: E402
from deepagents.profiles.harness.harness_profiles import (  # noqa: E402
    HarnessProfileConfig,
    register_harness_profile,
)

# 关掉默认的通用子 Agent（task 工具）：本实验专注 Skills 机制，
# 别让弱模型跑去委派添乱。注意 profile 按 provider 键注册，传模型实例时
# 查表用的是 provider="openai"（我们自建的 ChatOpenAI 都会命中）。
register_harness_profile(
    "openai",
    HarnessProfileConfig(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)


# =====================================================================
# 实验一：渐进式披露三级加载实测
# =====================================================================
def exp1() -> None:
    banner("实验一：三级加载 —— 书脊先行 / 匹配读正文 / 按需读附录")

    agent = create_deep_agent(
        model=model,
        tools=[get_phone_specs],  # Skill 正文里要求调用的工具必须注册
        backend=FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True),
        skills=["/skills/"],
        middleware=[CaptureSystemPrompt()],  # 截获模型实际看到的系统提示词（验证 L1）
        checkpointer=InMemorySaver(),
        # ⚠️ 坑：system_prompt 别重复 Skill 执行细节，否则分不清'元数据加载'和'泄题'
        # 硬性关卡模式（第 5 章验证过）：弱模型有工具就直奔工具、跳过手册，
        # 必须把"先读 SKILL.md"设为调用其他工具的前置条件
        system_prompt=(
            "你是一位严谨的手机顾问。开始任务前，先查看可用的技能列表；"
            "若某技能与任务相关，必须先用 read_file 读取它的 SKILL.md，"
            "读完之前禁止调用任何其他工具。"
            "然后严格按正文步骤执行，包括正文提到的参考文件。"
        ),
    )

    values = run_agent(
        agent,
        # 显式点名"按技能规范"——弱模型不点名的提问会直奔工具、跳过手册
        "请按照可用的技能规范，对比 Aurora 和 Nimbus，推荐一款更值得买的。",
        thread_id="exp1",
    )

    sys_text = captured.get("system", "")
    calls = tool_calls_of(values)
    final = values["messages"][-1].content
    read_paths = [c["args"].get("file_path", "") for c in calls if c["name"] == "read_file"]

    print("\n  🔬 三级加载逐层验证（基于模型实际收到的系统提示词）：")
    l1_name = "phone-analysis" in sys_text
    l1_desc = "推荐购买哪款手机" in sys_text          # description 片段
    l1_body = "必须先读再算分" not in sys_text and "跳过第 2 步" not in sys_text  # 正文片段不该在
    print(f"     L1 书脊先行：系统提示词含 Skill 名 {l1_name} / 含 description {l1_desc} /"
          f" 不含正文 {l1_body}")
    l2 = "/skills/phone-analysis/SKILL.md" in read_paths
    print(f"     L2 匹配读正文：read_file 读取了 SKILL.md -> {l2}")
    l3 = "/skills/phone-analysis/references/scoring.md" in read_paths
    print(f"     L3 按需读附录：read_file 读取了 scoring.md -> {l3}")

    scores_ok = "8.5" in final and "7.8" in final   # 用户只问了 Aurora 和 Nimbus
    header_ok = "【手机分析报告】" in final
    print(f"     产物验证：最终回答含写死分数（8.5/7.8，来自附录）{scores_ok} / "
          f"按规范以【手机分析报告】开头 {header_ok}（弱模型对格式要求常见跳步，如实记录）")
    print(f"     最终回答开头预览: {str(final)[:200]}")
    print(f"     全部 read_file 路径: {read_paths}")
    print(f"     未读的另一个 Skill（精准性）: report-writing 的 SKILL.md "
          f"{'没被读 ✅' if '/skills/report-writing/SKILL.md' not in read_paths else '被误读 ❌'}")

    print("\n📌 结论：启动时上下文里只有'书脊'（name+description），正文和附录都是")
    print("   Agent 判断相关后才通过 read_file 自己取的——50 个 Skill 也不怕撑爆上下文，")
    print("   这就是 Progressive Disclosure。产物里出现写死分数，证明附录真的被读了并用上了。")


# =====================================================================
# 实验二：精准匹配 + 同名分层覆盖（last wins）
# =====================================================================
def exp2() -> None:
    banner("实验二：同名 Skill 分层覆盖 —— shared vs project（last wins）")
    build_override_tree()

    agent = create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True),
        skills=["/skills/shared/", "/skills/project/"],  # 后面的覆盖前面的
        middleware=[CaptureSystemPrompt()],
        checkpointer=InMemorySaver(),
        system_prompt=(
            "你是一位严谨的手机顾问。开始任务前，先查看可用的技能列表；"
            "若某技能与任务相关，先用 read_file 读取它的 SKILL.md，"
            "然后严格按正文要求输出。不要查询或编造任何其他数据文件。亲自完成，不要委派。"
        ),
    )

    values = run_agent(agent, "请按照可用的手机分析技能，输出一份对 Aurora 的简评。", thread_id="exp2")
    final = values["messages"][-1].content
    sys_text = captured.get("system", "")

    has_project = "【项目版报告】" in final
    has_shared = "【共享版报告】" in final
    print(f"\n  🔬 覆盖验证：")
    print(f"     skills 列表 = ['/skills/shared/', '/skills/project/']（同名 last wins）")
    print(f"     最终回答以【项目版报告】规范输出: {has_project}")
    print(f"     是否误用了【共享版报告】: {has_shared}")
    print(f"     系统提示词里的 phone-analysis 指向: "
          f"{'项目版' if '【项目版报告】' in sys_text else ('共享版' if '【共享版报告】' in sys_text else '都不含（书脊只到描述层）')}")

    print("\n📌 结论：同名 Skill 时列表中靠后的路径胜出——这就是分层覆盖机制：")
    print("   组织级 /skills/org/ 放通用规范，项目级 /skills/project/ 放定制版，")
    print("   同名即覆盖，团队积累的知识可以按层级'就近生效'。")


# =====================================================================
# 实验三：Skill 只读保护 —— deny 写入 + 对照组
# =====================================================================
def exp3() -> None:
    banner("实验三：Skill 权限 —— deny /skills/** 写入（企业知识库场景）")

    agent = create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=DEMO_DIR, virtual_mode=True),
        skills=["/skills/"],
        permissions=[
            FilesystemPermission(
                operations=["write", "edit", "delete"],
                paths=["/skills/**"],
                mode="deny",  # 手册只能读不能改；改用 "interrupt" 则变成人工审批
            ),
        ],
        checkpointer=InMemorySaver(),
        system_prompt="你是助手。用户让你改文件就尝试改，工具报错就把错误原样转告用户。"
                      "写入新文件用 write_file 工具。",
    )

    print(">>> A. 让它修改 Skill 手册（应被 deny 拦截）")
    run_agent(
        agent,
        "请把 /skills/phone-analysis/SKILL.md 文件里的【手机分析报告】改成【新版报告】。",
        thread_id="exp3",
    )

    print("\n>>> B. 对照组：往普通路径写便签（应放行）")
    run_agent(agent, "请用 write_file 把'明天复核评分标准'写入 /notes/todo.md", thread_id="exp3b")

    skill_file = os.path.join(DEMO_DIR, "skills", "phone-analysis", "SKILL.md")
    with open(skill_file, encoding="utf-8") as f:
        untouched = "【手机分析报告】" in f.read() and "【新版报告】" not in f.read()
    note_file = os.path.join(DEMO_DIR, "notes", "todo.md")
    note_ok = os.path.exists(note_file)

    print("\n  🔬 权限验证：")
    print(f"     A. Skill 手册未被篡改（原文仍在/改动未落盘）: {untouched}")
    print(f"     B. 对照组普通路径写入成功落盘: {note_ok}")

    print("\n📌 结论：FilesystemPermission(mode='deny') 让 Skill 库变成'只读知识库'——")
    print("   Agent 能发现、能读、能执行，但改不了手册；普通路径不受影响。")
    print("   把 mode 换成 'interrupt' 就是'修改需人工审批'（第 9 章 HITL 的预演）。")


if __name__ == "__main__":
    build_skill_tree()
    if should_run(1):
        exp1()
    if should_run(2):
        exp2()
    if should_run(3):
        exp3()
    print("\n全部实验完成 ✅")
