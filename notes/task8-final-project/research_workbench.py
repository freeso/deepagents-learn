# -*- coding: utf-8 -*-
"""深研工坊（Deep Research Workbench）—— Task 8 结营综合项目

一个终端版"调研 → 成稿 → 审批 → 归档"的 Agent 应用，综合使用课程六项能力：

  ch04 任务规划   write_todos 自动拆解研究任务（TodoListMiddleware）
  ch05 子 Agent   资料收集委派给 data-collector，主上下文只见摘要不见原始 JSON
  ch07 Skills     report-writing 技能书脊自动注入，匹配后按需读正文
  ch08 长期记忆   /memories/ 路由到 JSON 落盘 Store，偏好跨会话、跨进程存活
  ch09+11 HITL    定稿写入 /reports/final/** 必须人工审批（FilesystemPermission interrupt）
  ch03/11 权限    /policies/** 与 /skills/** 禁止修改（deny），草稿区 /reports/drafts/** 放行

虚拟文件布局（CompositeBackend 路由，前缀剥离后各归各家）：

  /workspace/**  StateBackend          每次会话清空的"草稿桌"
  /memories/**   JsonFileStore(.tmp)   用户偏好，跨进程长存
  /skills/**     agent_data/skills/    技能手册（只读，随仓库提交）
  /policies/**   agent_data/policies/  发布政策（只读，随仓库提交）
  /reports/**    .tmp/workshop/reports/ 初稿 drafts/ 免审批，定稿 final/ 需审批

用法：
  python research_workbench.py               # 交互聊天（演示入口）
  python research_workbench.py --reset       # 清空运行时数据后退出
  python research_workbench.py --memcheck    # 跨进程记忆自检（新进程+假模型，零 API 成本）

依赖环境变量：SILICONFLOW_API_KEY（可选 MODEL_NAME，默认 Qwen2.5-7B）
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "agent_data")
RUNTIME_DIR = os.path.join(BASE_DIR, ".tmp", "workshop")
STORE_PATH = os.path.join(RUNTIME_DIR, "store.json")
REPORTS_DIR = os.path.join(RUNTIME_DIR, "reports")

DEFAULT_USER = "cherry"


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

from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.base import (  # noqa: E402
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchOp,
)
from langgraph.types import Command  # noqa: E402

from deepagents import FilesystemPermission, create_deep_agent  # noqa: E402
from deepagents.backends import (  # noqa: E402
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)
from deepagents.backends.utils import create_file_data  # noqa: E402
from deepagents.middleware.memory import MemoryMiddleware  # noqa: E402
from deepagents.profiles.harness.harness_profiles import (  # noqa: E402
    GeneralPurposeSubagentProfile,
    HarnessProfileConfig,
    register_harness_profile,
)

# 弱模型防护（第 5/7/8/9 章同款）：关掉默认通用子 Agent，只留我们定义的
try:
    register_harness_profile(
        "openai",
        HarnessProfileConfig(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
        ),
    )
except Exception:  # 已注册过则忽略（同进程多次 build 时）
    pass


# =====================================================================
# JsonFileStore：把 Store 落到 JSON 文件 —— 记忆跨进程存活的物理基础
#（课程第 8 章用的是 InMemoryStore（进程即失忆），这里升级为落盘版）
# =====================================================================
class JsonFileStore(BaseStore):
    """极简 JSON 文件持久化 Store：put 即落盘，重启进程即重读。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.data: dict[tuple[str, ...], dict[str, dict]] = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            self.data = {tuple(k.split("/")): v for k, v in raw.items()}

    def _flush(self) -> None:
        with open(self.path, "w") as f:
            json.dump(
                {"/".join(k): v for k, v in self.data.items()},
                f,
                ensure_ascii=False,
                indent=1,
            )

    @staticmethod
    def _to_item(ns: tuple[str, ...], key: str, rec: dict) -> Item:
        return Item(
            value=rec["value"],
            key=key,
            namespace=ns,
            created_at=datetime.fromtimestamp(rec["created_at"], timezone.utc),
            updated_at=datetime.fromtimestamp(rec["updated_at"], timezone.utc),
        )

    def batch(self, ops):
        out = []
        for op in ops:
            if isinstance(op, PutOp):
                bucket = self.data.setdefault(op.namespace, {})
                prev = bucket.get(op.key)
                now = datetime.now(timezone.utc).timestamp()
                bucket[op.key] = {
                    "value": op.value,
                    "created_at": prev["created_at"] if prev else now,
                    "updated_at": now,
                }
                self._flush()
                out.append(None)
            elif isinstance(op, GetOp):
                rec = self.data.get(op.namespace, {}).get(op.key)
                out.append(self._to_item(op.namespace, op.key, rec) if rec else None)
            elif isinstance(op, SearchOp):
                pref = op.namespace_prefix
                hits = [
                    self._to_item(ns, k, r)
                    for ns, bucket in self.data.items()
                    if ns[: len(pref)] == pref
                    for k, r in bucket.items()
                ]
                out.append(hits[: op.limit])
            elif isinstance(op, ListNamespacesOp):
                out.append(sorted(self.data.keys()))
            else:
                msg = f"unsupported op: {op!r}"
                raise ValueError(msg)
        return out

    async def abatch(self, ops):
        return self.batch(ops)


# =====================================================================
# 数据工具：确定性模拟数据（免费、可复现），执行全部记日志（验证锚点）
# 注意：这两个工具只挂在子 Agent 身上，主 Agent 没有 → 必须委派（第 5 章设计）
# =====================================================================
EXECUTED_LOG: list[tuple] = []

PHONE_DB: dict[str, dict] = {
    "Aurora": {
        "price_cny": 4999, "battery_mah": 5000, "weight_g": 201,
        "screen_inch": 6.4, "camera_mp": 50,
    },
    "Nimbus": {
        "price_cny": 3599, "battery_mah": 6000, "weight_g": 226,
        "screen_inch": 6.7, "camera_mp": 12,
    },
    "Zenith": {
        "price_cny": 6299, "battery_mah": 4400, "weight_g": 172,
        "screen_inch": 6.2, "camera_mp": 108,
    },
}

REVIEW_DB: dict[str, list[str]] = {
    "Aurora": ["夜景照片有涂抹感，但白天很稳", "手感大小正好，单手无压力", "价格略贵，等一个促销"],
    "Nimbus": ["续航真能撑两天，重度使用也行", "拍照日常够用，别期待效果", "机身偏重，躺床上刷手机砸脸"],
    "Zenith": ["拍演唱会月亮都清楚，长焦离谱", "电池小了点，出门必带充电宝", "一分钱一分货，贵有贵的道理"],
}


@tool
def get_phone_specs(brand: str) -> dict:
    """查询指定品牌手机的完整参数（价格/电池/重量/屏幕/相机）。

    Args:
        brand: 品牌名，可选值：Aurora、Nimbus、Zenith。

    Returns:
        dict：该品牌的完整参数；品牌不存在时返回 {"error": "说明"}。
    """
    EXECUTED_LOG.append(("get_phone_specs", brand))
    data = PHONE_DB.get(brand)
    if data is None:
        return {"error": f"没有找到品牌 {brand}，可选：Aurora、Nimbus、Zenith"}
    return {"brand": brand, **data}


@tool
def get_user_reviews(brand: str, limit: int = 3) -> list[str]:
    """查询指定品牌手机的用户口碑评论。

    Args:
        brand: 品牌名，可选值：Aurora、Nimbus、Zenith。
        limit: 最多返回几条，默认 3。

    Returns:
        list[str]：用户评论原话列表；品牌不存在时返回错误说明列表。
    """
    EXECUTED_LOG.append(("get_user_reviews", brand))
    reviews = REVIEW_DB.get(brand)
    if reviews is None:
        return [f"没有找到品牌 {brand} 的评论，可选：Aurora、Nimbus、Zenith"]
    return reviews[:limit]


def ran(name: str, *keywords: str) -> bool:
    """执行日志里是否出现过指定工具调用（且参数包含全部关键词，子串匹配）。"""
    return any(
        e[0] == name and all(any(k in str(x) for x in e) for k in keywords)
        for e in EXECUTED_LOG
    )


# =====================================================================
# 发布工具：唯一能把报告写进 /reports/final/ 的通道。
# 挂 interrupt_on 审批：批准前工具不会执行（第 9 章机制），
# 工具内部用普通文件 IO 复制初稿，不长两手的写 content 又绕开弱模型长参数短板。
# （agent 的 write_file 对 /reports/final/** 是 deny —— 不许绕过审批）
# =====================================================================
@tool
def publish_report(draft_path: str) -> str:
    """发布报告：把草稿区 /reports/drafts/ 下的初稿发布到最终报告区 /reports/final/（同名文件）。发布动作需要人工审批。

    Args:
        draft_path: 初稿路径，必须以 /reports/drafts/ 开头。

    Returns:
        str: 发布成功提示；路径非法或初稿不存在时返回 Error 说明。
    """
    EXECUTED_LOG.append(("publish_report", draft_path))
    rel = draft_path.removeprefix("/reports/")
    src = os.path.join(REPORTS_DIR, rel)
    if not (rel.startswith("drafts/") and os.path.isfile(src)):
        return (
            f"Error: 初稿 {draft_path} 不存在（/reports/drafts/ 下没有这个文件），"
            "说明撰稿人 report-writer 还没有真实写入初稿。请重新委派 report-writer，"
            "等它真正用 write_file 写入 /reports/drafts/ 并汇报路径后，再来发布。"
        )
    dst = os.path.join(REPORTS_DIR, "final", os.path.basename(src))
    with open(src) as f:
        content = f.read()
    with open(dst, "w") as f:
        f.write(content)
    return f"发布成功：/reports/final/{os.path.basename(src)}"


# =====================================================================
# 身份上下文与系统提示词
# =====================================================================
@dataclass
class UserContext:
    user_id: str


EMPTY_PREFS = "# 用户偏好\n（暂无记录）"

SYS_MAIN = (
    "你是深研工坊的总编辑，负责把调研需求变成合乎规范的报告。你是编排者："
    "不亲自查数据、不亲自写报告正文。\n"
    "铁律：研究类任务从拆任务到发布是一个连续工作流。"
    "在报告真实发布成功之前，禁止向用户输出任何进度汇报或后续计划说明"
    "（'接下来我将…'这类回复是违规的）；只有全部步骤完成后才允许回复最终结论。\n"
    "研究类任务（对比分析、写报告）必须逐步执行以下流程：\n"
    "1. 先用 write_todos 制定任务清单，至少包含：委派资料收集、委派撰稿、"
    "发布审批三项；\n"
    "2. 委派子 Agent data-collector 收集资料（任务描述只写需要哪些品牌的"
    "什么资料），拿到它的【资料摘要】后立即进入第 3 步——"
    "禁止重复委派 data-collector（资料已经在手，重复委派是违规）；\n"
    "3. 把资料摘要全文连同调研主题一起委派给子 Agent report-writer 撰写初稿。"
    "委派的任务描述里必须把【资料摘要】逐字附上（所有数字都要在，"
    "只写'根据资料摘要撰写'而不附原文是违规），等它回复'初稿已完成：<路径>'；\n"
    "4. 调用 publish_report 工具发布（参数就是撰稿人给出的初稿路径），"
    "发布需要用户审批，批准后工具才会执行成功；"
    "禁止用 write_file 自己往 /reports/final/ 写文件；\n"
    "5. publish_report 返回成功后，用如下格式回复：\n"
    "『报告已发布：<定稿路径>；结论摘要：<一句话>』。"
    "这是研究任务唯一合法的最终回复，中间步骤一律连续执行，不得中途停下汇报。\n"
    "当用户要求'记住'偏好时：先用 read_file 读取 /memories/preferences.md，"
    "再用 edit_file 把新偏好追加到文件末尾（保留已有内容，禁止 write_file 整体覆盖），"
    "工具调用成功后才允许回复'已记住'。\n"
    "当用户指出某个操作尚未真实完成时（如目录里没有文件），必须立刻用真实的工具调用来完成，"
    "禁止重复声称已完成、禁止辩解。\n"
    "禁止：在任何未调用工具的情况下声称已完成操作；"
    "修改 /policies/ 或 /skills/ 目录下的任何文件。"
)

# 监工回话：模型半途播报/虚报/打转时，用地面真值把它拍回工作流（验证与演示共用）
NUDGE = (
    "继续执行，注意不要打转：\n"
    "- 如果 data-collector 已经返回过【资料摘要】，不要再重复委派它；"
    "- 下一步应把资料摘要委派给 report-writer 撰写初稿（写入 /reports/drafts/）；"
    "- 初稿就位后调用 publish_report 发布；\n"
    "- 每个动作都要用真实的工具调用完成，文件真实存在之前禁止声称已发布。"
)

COLLECTOR_PROMPT = (
    "你是资料收集员，负责查询手机数据库并回传摘要。接到任务后必须严格照做：\n"
    "1. 对任务指定的每一个品牌，分别调用 get_phone_specs 和 get_user_reviews，"
    "各调用一次即可，禁止重复查询；\n"
    "2. 全部查询完成后，回复摘要，格式铁律：第一行必须是'【资料摘要】'四个字；"
    "之后每个品牌一段中文，包含价格（X 元）、电池（X mAh）、重量（X g）三个数字"
    "和一句最有代表性的用户评论原话；最后一行必须是'资料收集完毕'。\n"
    "禁止：回复中出现任何 JSON、花括号或原始字段名；跳过查询编造数字；"
    "只回复'已完成'不给出摘要。"
)

WRITER_PROMPT = (
    "你是撰稿人，负责把调研资料写成报告初稿。任务描述里会给出【资料摘要】（含真实数字）。"
    "接到任务后必须严格照做：\n"
    "1. 如果任务描述里没有附【资料摘要】或摘要里缺少数字，禁止编写，"
    "直接回复'任务描述缺少资料摘要，请附上后重新委派'；\n"
    "2. 用 read_file 读取 /skills/report-writing/SKILL.md，严格按其中步骤执行；\n"
    "3. 用 read_file 读取 /policies/report-policy.md，全文遵守；\n"
    "4. 按 SKILL 里的报告模板，用 write_file 把初稿写入 /reports/drafts/report-<主题>.md。"
    "所有数字必须来自【资料摘要】原文，一个都不许编；"
    "禁止出现模板占位符（如'（明确的推荐对象…'）和凭空编造的评分/评测文件；\n"
    "5. write_file 返回成功信息之后，才能回复，回复第一行必须是"
    "'初稿已完成：/reports/drafts/<主题>.md'，再附两行内容摘要。\n"
    "禁止：不读规范就写；write_file 没有真实执行就声称完成。"
)

DATA_COLLECTOR = {
    "name": "data-collector",
    "description": "收集手机参数与用户口碑数据时委派给它，它会查询数据库并返回要点摘要（含关键数字）。",
    "system_prompt": COLLECTOR_PROMPT,
    "tools": [get_phone_specs, get_user_reviews],
}

REPORT_WRITER = {
    "name": "report-writer",
    "description": "撰写报告初稿时委派给它。把资料摘要交给它，它会按写作规范和政策成稿并写入 /reports/drafts/。",
    "system_prompt": WRITER_PROMPT,
    "tools": [],  # 文件工具自动注入（第 5 章源码结论）
}


# =====================================================================
# 中间件：偷拍员（记录每次模型调用实际收到的系统提示词，验证注入用）
# =====================================================================
CAPTURED_PROMPTS: list[str] = []


class CaptureSystemPrompt(AgentMiddleware):
    """截获模型实际收到的系统提示词（追加记录，不覆盖）。"""

    def wrap_model_call(self, request, handler):
        content = (
            request.system_message.content
            if isinstance(request.system_message.content, str)
            else str(request.system_message.content)
        )
        CAPTURED_PROMPTS.append(content)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        return self.wrap_model_call(request, handler)


def captured_all() -> list[str]:
    return list(CAPTURED_PROMPTS)


def captured_since(mark: int) -> list[str]:
    return CAPTURED_PROMPTS[mark:]


# =====================================================================
# 工厂：组装整套工坊
# =====================================================================
def build_model():
    api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("缺少 API key：请 export SILICONFLOW_API_KEY=... 后重试")
    return ChatOpenAI(
        model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        temperature=0,
        timeout=120,
        max_retries=3,
    )


def ensure_runtime() -> None:
    os.makedirs(os.path.join(REPORTS_DIR, "drafts"), exist_ok=True)
    os.makedirs(os.path.join(REPORTS_DIR, "final"), exist_ok=True)


def reset_runtime() -> None:
    """清空运行时数据（store.json + 报告目录），回到全新装机状态。"""
    import shutil

    shutil.rmtree(REPORTS_DIR, ignore_errors=True)
    if os.path.exists(STORE_PATH):
        os.remove(STORE_PATH)
    ensure_runtime()


def build_workbench(
    user_id: str = DEFAULT_USER,
    model=None,
    store: JsonFileStore | None = None,
):
    """组装工坊 Agent。user_id 决定记忆的隔离边界；model 默认真实 Qwen。"""
    ensure_runtime()
    model = model or build_model()
    store = store or JsonFileStore(STORE_PATH)

    # 记忆冷启动：新用户先铺一张空偏好表（memory 源缺失会被静默跳过）
    ns = (user_id, "memories")
    if store.get(ns, "/preferences.md") is None:
        store.put(ns, "/preferences.md", create_file_data(EMPTY_PREFS))

    composite = CompositeBackend(
        default=StateBackend(),  # /workspace/** 等未匹配路径 → 会话级草稿桌
        routes={
            "/skills/": FilesystemBackend(
                root_dir=os.path.join(DATA_DIR, "skills"), virtual_mode=True
            ),
            "/policies/": FilesystemBackend(
                root_dir=os.path.join(DATA_DIR, "policies"), virtual_mode=True
            ),
            "/reports/": FilesystemBackend(
                root_dir=REPORTS_DIR, virtual_mode=True
            ),
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.context.user_id, "memories"),
                store=store,
            ),
        },
    )

    agent = create_deep_agent(
        model=model,
        tools=[publish_report],  # 主 Agent 只掌握发布权，且发布必经审批
        subagents=[DATA_COLLECTOR, REPORT_WRITER],
        interrupt_on={
            "publish_report": {"allowed_decisions": ["approve", "edit", "reject"]}
        },
        backend=composite,
        skills=["/skills/"],
        permissions=[
            # 三区制（规则按序首中即生效）：
            # 政策区/技能区/发布区禁写硬墙（发布唯一通道是经审批的 publish_report
            # 工具，工具执行不走文件权限中间件）；草稿区默认放行，撰稿人在那里工作
            FilesystemPermission(
                operations=["write"],
                paths=["/policies/**", "/skills/**", "/reports/final/**"],
                mode="deny",
            ),
        ],
        context_schema=UserContext,
        checkpointer=InMemorySaver(),  # HITL 必须
        middleware=[
            TodoListMiddleware(),  # ch04 任务规划
            # 手动构造 MemoryMiddleware 并放在偷拍员外层（第 8 章源码结论：
            # memory= 交给 create_deep_agent 的注入位置比用户中间件更深，拍不到）
            MemoryMiddleware(backend=composite, sources=["/memories/preferences.md"]),
            CaptureSystemPrompt(),
        ],
        system_prompt=SYS_MAIN,
    )
    return agent


# =====================================================================
# 通用小工具（供 verify.py 与交互入口共用）
# =====================================================================
def invoke_v2(agent, payload, cfg, user_id: str = DEFAULT_USER):
    cfg = {**cfg, "recursion_limit": 60} if "recursion_limit" not in cfg else cfg
    return agent.invoke(
        payload, config=cfg, context=UserContext(user_id), version="v2"
    )


def values_of(result):
    return result.value if hasattr(result, "value") else result


def final_answer(values) -> str:
    for m in reversed(values.get("messages", [])):
        if type(m).__name__ == "AIMessage":
            c = getattr(m, "content", "")
            if c:
                return c if isinstance(c, str) else str(c)
    return ""


def tool_calls_of(values) -> list[dict]:
    calls = []
    for m in values.get("messages", []):
        for tc in getattr(m, "tool_calls", None) or []:
            calls.append({"name": tc["name"], "args": tc["args"]})
    return calls


def show_interrupts(result, tag: str = "") -> list[dict]:
    if not result.interrupts:
        return []
    v = result.interrupts[0].value
    ars = v["action_requests"]
    for a in ars:
        print(f"  ⏸️  {tag}待审批: {a['name']}({str(a['args'])[:100]})")
    return ars


def read_report(rel_dir: str) -> tuple[str, str]:
    """读取 reports 下某子目录里最新修改的报告（返回 文件名, 内容）。"""
    d = os.path.join(REPORTS_DIR, rel_dir)
    if not os.path.exists(d):
        return "", ""
    names = [n for n in os.listdir(d) if not n.startswith(".")]
    if not names:
        return "", ""
    newest = max(names, key=lambda n: os.path.getmtime(os.path.join(d, n)))
    with open(os.path.join(d, newest)) as f:
        return newest, f.read()


def auto_run(
    agent,
    question: str,
    thread: str,
    user_id: str = DEFAULT_USER,
    decider=None,
    goal_check=None,
    max_steps: int = 5,
):
    """带监工的一轮任务：模型停下来就验收地面真值，目标没达成就自动催办。

    - goal_check: () -> bool，返回 True 才算完成；None 表示拿到回复即完成
    - decider: action_request -> decision，处理审批中断；None 表示中断直接
      返回给调用方（交互模式下留给人工决策）
    弱模型适配：Qwen2.5-7B 常在长流程中途停下播报甚至虚报完成，
    监工不读模型嘴，只认磁盘上的事实。
    """
    cfg = {"configurable": {"thread_id": thread}}
    payload = {"messages": [{"role": "user", "content": question}]}
    r = None
    for step in range(max_steps):
        if step:
            print(f"  ⏩ 监工 Round {step}")
        r = invoke_v2(agent, payload, cfg, user_id)
        while getattr(r, "interrupts", None):
            if decider is None:
                return r  # 交互模式：交给人工
            ars = show_interrupts(r, "[监工] ")
            r = invoke_v2(
                agent, Command(resume={"decisions": [decider(a) for a in ars]}), cfg, user_id
            )
        if goal_check is None or goal_check():
            return r
        print("  🔁 监工：目标未达成（地面真值检查失败），发催办指令")
        payload = {"messages": [{"role": "user", "content": NUDGE}]}
    return r


# =====================================================================
# 交互聊天（演示入口）
# =====================================================================
HELP = """命令：
  直接输入内容     与工坊对话（研究任务会自动走 规划→委派→成稿→审批 流程）
  /new            开启新会话（thread 换新，检查点里旧会话不再接续）
  /user <id>      切换用户身份（记忆按用户隔离，换个 id 就是另一份记忆）
  /files          查看报告目录里已生成的文件
  /exit           退出
审批时输入：a = 批准；r <原因> = 拒绝并说明原因；e = 跳过（本轮不决策）
"""


def chat() -> None:
    print("=" * 60)
    print("深研工坊 Deep Research Workbench")
    print("规划 / 委派 / 技能 / 记忆 / 审批 / 权限 全在一个终端里")
    print("=" * 60)
    print(HELP)

    user_id = DEFAULT_USER
    agent = build_workbench(user_id)
    thread = str(uuid.uuid4())

    while True:
        try:
            q = input(f"\n[{user_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/exit", "/quit"):
            break
        if q == "/new":
            thread = str(uuid.uuid4())
            print(f"  ↪ 新会话 {thread[:8]}（同一用户，长期记忆依然生效）")
            continue
        if q.startswith("/user"):
            parts = q.split(maxsplit=1)
            if len(parts) < 2:
                print("  用法：/user <id>")
                continue
            user_id = parts[1].strip()
            agent = build_workbench(user_id)
            thread = str(uuid.uuid4())
            print(f"  ↪ 已切换到用户 {user_id}（新 Agent、新记忆空间）")
            continue
        if q == "/files":
            for sub in ("drafts", "final"):
                d = os.path.join(REPORTS_DIR, sub)
                names = os.listdir(d) if os.path.isdir(d) else []
                print(f"  /reports/{sub}/: {sorted(names) or '（空）'}")
            continue

        cfg = {"configurable": {"thread_id": thread}}
        try:
            r = invoke_v2(agent, {"messages": [{"role": "user", "content": q}]}, cfg, user_id)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ 出错：{e}")
            continue

        # 审批循环：定稿写入 /reports/final/** 会触发 interrupt
        while getattr(r, "interrupts", None):
            ars = show_interrupts(r, "")
            print("  请决策 [a]批准 / [r <原因>]拒绝 / [e]跳过 > ", end="")
            decisions = []
            raw = input().strip()
            if raw == "e":
                break
            for a in ars:
                if raw.startswith("r"):
                    reason = raw[1:].strip() or "用户未说明原因"
                    decisions.append({"type": "reject", "message": f"用户拒绝：{reason}"})
                else:
                    decisions.append({"type": "approve"})
            try:
                r = invoke_v2(
                    agent, Command(resume={"decisions": decisions}), cfg, user_id
                )
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ 恢复失败：{e}")
                break

        values = values_of(r)
        ans = final_answer(values)
        if ans:
            print(f"\n{ans}")
        print(f"\n  （本会话工具调用 {len(tool_calls_of(values))} 次 | 报告目录 /reports/）")


# =====================================================================
# 跨进程记忆自检（--memcheck）：新开 OS 进程 + 假模型，零 API 成本
# =====================================================================
class _NoopFakeModel(FakeMessagesListChatModel):
    """假模型：bind_tools 直接返回自身（绕过 NotImplementedError，第 9 章同款技巧）。"""

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


def memcheck(user_id: str = DEFAULT_USER) -> int:
    """验证 store.json 里的偏好能在全新进程里自动注入系统提示词。"""
    fake = _NoopFakeModel(responses=[AIMessage(content="好的")])
    agent = build_workbench(user_id=user_id, model=fake)
    CAPTURED_PROMPTS.clear()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    agent.invoke(
        {"messages": [{"role": "user", "content": "你好"}]},
        config=cfg,
        context=UserContext(user_id),
        version="v2",
    )
    sys_prompt = "\n".join(CAPTURED_PROMPTS[-1:])
    store = JsonFileStore(STORE_PATH)
    saved = store.get((user_id, "memories"), "/preferences.md")
    saved_text = (saved.value.get("content") if saved else "") or ""
    real_prefs = [ln for ln in saved_text.splitlines()
                  if ln.strip() and "暂无记录" not in ln]
    injected = "<agent_memory>" in sys_prompt and all(p in sys_prompt for p in real_prefs)
    print(
        json.dumps(
            {
                "user_id": user_id,
                "saved_preferences": saved_text,
                "memory_injected": injected,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if injected else 1


if __name__ == "__main__":
    if "--reset" in sys.argv:
        reset_runtime()
        print("运行时数据已清空（store.json + 报告目录）")
    elif "--memcheck" in sys.argv:
        idx = sys.argv.index("--memcheck")
        uid = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else DEFAULT_USER
        raise SystemExit(memcheck(uid))
    else:
        chat()
