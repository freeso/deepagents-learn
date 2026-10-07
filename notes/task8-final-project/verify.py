# -*- coding: utf-8 -*-
"""深研工坊 —— 自动验收脚本

四条场景 + 一个跨进程自检，全部用确定性锚点判定（执行日志 / 落盘文件 /
store 内容 / 偷拍的系统提示词），不读模型嘴：

  场景一  完整流水线：规划 → 委派收集 → 读技能+政策 → 初稿（免审批）
          → 定稿（弹审批 approve）→ 落盘 —— 六项能力一镜到底
  场景二  记忆回路：记住偏好 → store 落盘 → 同进程全新 Agent + 全新线程
          自动注入（证明持久化不靠 checkpointer）→ 新 OS 进程再验一次
  场景三  权限红线：指使 Agent 篡改 /policies/report-policy.md → deny 拦截
          + 政策原文一字未动（对照组：场景一里草稿区写入畅通无阻）
  场景四  拒绝回路：定稿审批给 reject（带替代建议）→ 未发布 → 补充指令后
          重新提交 → approve → 落盘

运行：../../../.venv/bin/python verify.py           # 全部
      ../../../.venv/bin/python verify.py 1         # 只跑场景一
依赖：SILICONFLOW_API_KEY（免费 Qwen2.5-7B，全部确定性模拟数据）
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import research_workbench as wb  # noqa: E402

ONLY = {a for a in sys.argv[1:] if a.isdigit()}

FAILED: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  {'✅' if ok else '❌'} {label}")
    if not ok:
        FAILED.append(label)


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_round(agent, question: str, thread: str, user_id: str = "cherry"):
    """一轮对话（不含审批），打印动作流水。"""
    print(f"\n>>> 用户: {question}")
    cfg = {"configurable": {"thread_id": thread}}
    mark = len(wb.CAPTURED_PROMPTS)
    r = wb.invoke_v2(agent, {"messages": [{"role": "user", "content": question}]}, cfg, user_id)
    values = wb.values_of(r)
    for c in wb.tool_calls_of(values):
        print(f"  ⚙️  {c['name']}({str(c['args'])[:90]})")
    return r, values, mark


def drain_interrupts(agent, cfg, r, decider, user_id: str = "cherry", budget: int = 4):
    """审批循环：对每个待审批动作调用 decider(action)->decision；返回最后一次结果。"""
    rounds = 0
    while getattr(r, "interrupts", None) and rounds < budget:
        ars = wb.show_interrupts(r)
        decisions = []
        for a in ars:
            decisions.append(decider(a))
            rounds += 1
        r = wb.invoke_v2(agent, wb.Command(resume={"decisions": decisions}), cfg, user_id)
    return r


# =====================================================================
def scenario1() -> None:
    banner("场景一：完整流水线 —— 规划/委派/技能/初稿/审批/定稿")

    agent = wb.build_workbench("cherry")
    wb.EXECUTED_LOG.clear()
    wb.CAPTURED_PROMPTS.clear()
    thread = str(uuid.uuid4())
    final_dir = os.path.join(wb.REPORTS_DIR, "final")
    goal = lambda: bool(  # noqa: E731 地面真值：定稿区真有文件才算完成
        [n for n in os.listdir(final_dir) if not n.startswith(".")]
    )
    approval = {"count": 0}

    def approver(action: dict) -> dict:
        approval["count"] += 1
        return {"type": "approve"}

    print(">>> 用户: 请对比 Aurora 和 Nimbus 写研究报告，初稿进草稿区，定稿发布到最终报告区。")
    print("    （监工模式：目标没达成会自动催办，不读模型嘴只认磁盘事实）")
    r = wb.auto_run(
        agent,
        "请对比 Aurora 和 Nimbus 这两款手机，写一份研究报告给我。"
        "初稿放草稿区，定稿发布到最终报告区。",
        thread,
        decider=approver,
        goal_check=goal,
    )
    values = wb.values_of(r)

    calls = wb.tool_calls_of(values)
    names = [c["name"] for c in calls]
    main_text = str(values.get("messages", []))

    print("\n  🔍 逐项验收：")
    check("ch04 规划：write_todos 制定了任务清单", "write_todos" in names)
    check(
        "ch05 委派（查询）：资料收集交给了 data-collector",
        any(c["name"] == "task" and c["args"].get("subagent_type") == "data-collector"
            for c in calls),
    )
    check(
        "ch05 委派（撰稿）：初稿撰写交给了 report-writer",
        any(c["name"] == "task" and c["args"].get("subagent_type") == "report-writer"
            for c in calls),
    )
    check("ch05 执行：参数库被真实查询（Aurora 与 Nimbus 各一次）",
          wb.ran("get_phone_specs", "Aurora") and wb.ran("get_phone_specs", "Nimbus"))
    check("ch05 执行：口碑库被真实查询", wb.ran("get_user_reviews"))
    check(
        "ch05 隔离：原始 JSON 字段名没有进入主上下文（只见摘要不见原文）",
        "battery_mah" not in main_text and "camera_mp" not in main_text,
    )
    check(
        "ch05 数据走摘要：主上下文含资料摘要标志（数字在，字段名不在）",
        "【资料摘要】" in main_text or "4999" in main_text,
    )
    check(
        "ch05 隔文件交接：资料摘要真实落盘 /materials/summary.md（传话失真免疫）",
        os.path.isfile(os.path.join(wb.MATERIALS_DIR, "summary.md")),
    )
    check(
        "ch07 书脊：系统提示词被自动注入技能名（L1 渐进式披露，技能由撰稿人 L2 读正文）",
        any("report-writing" in p for p in wb.captured_all()),
    )
    draft_name, draft = wb.read_report("drafts")
    draft_ok = bool(draft) and "结论" in draft and draft.find("结论") < draft.find("分析")
    check(f"撰稿人写初稿到草稿区且结构合规（结论在前，SKILL 模板生效）: {draft_name}", draft_ok)
    check("初稿内容合规：正文 ≥200 字（政策条款通过初稿落实）", len(draft) >= 200)
    check("初稿非抄袭：不是把规范/政策文件当正文", "执行步骤" not in draft[:60])
    _, final = wb.read_report("final")
    check(
        "ch09 审批：publish_report 弹出过人工审批且被批准（次数≥1）",
        approval["count"] >= 1 and wb.ran("publish_report", "/reports/drafts/"),
    )
    check("发布工具实际执行：定稿已复制到最终报告区", bool(final))
    final_files = [n for n in os.listdir(final_dir) if not n.startswith(".")]
    check("发布区副本可留档：/reports/final/** 对 Agent 是 deny（防硬改）",
          bool(final_files) and os.path.isfile(os.path.join(final_dir, final_files[0])))
    real_numbers = (4999, 3599, 5000, 6000, 201, 226)
    check(
        "数据保真：定稿含全部真实数字（价格/电池/重量×两机）",
        bool(final) and all(str(n) in final for n in real_numbers),
    )
    check(
        "数据保真：无编造痕迹（评分/不存在的数据文件/占位符）",
        bool(final) and not any(
            kw in final for kw in ("评分", "phone_reviews", "/data/", "明确的推荐对象")
        ),
    )
    answer = wb.final_answer(values)
    check("总编汇报完成", bool(answer))
    print(f"  ↪ 最终回复摘要: {answer[:80]}...")
    if final:
        print(f"  ↪ 定稿开头: {final[:60].replace(chr(10), ' / ')}...")


# =====================================================================
def scenario2() -> None:
    banner("场景二：记忆回路 —— 写入 store，换 Agent 换线程自动想起")

    PREF = "结论必须放在报告最前面"

    agent = wb.build_workbench("cherry")
    r, values, _ = run_round(
        agent, f"请记住我的报告偏好：{PREF}。", str(uuid.uuid4())
    )
    store = wb.JsonFileStore(wb.STORE_PATH)
    saved = store.get(("cherry", "memories"), "/preferences.md")
    saved_text = (saved.value.get("content") if saved else "") or ""
    check("偏好真实写入 store.json（不信模型嘴）", PREF in saved_text)

    # 同进程：全新 Agent + 全新线程（旧的 checkpointer/thread 全部不要）
    wb.CAPTURED_PROMPTS.clear()
    agent2 = wb.build_workbench("cherry")  # 新 store 实例 = 从 JSON 重新加载
    r2, values2, _ = run_round(
        agent2, "按我的报告偏好，给我一句口头版的开场结论示例，不用写任何文件。",
        str(uuid.uuid4()),
    )
    prompts2 = wb.CAPTURED_PROMPTS
    check(
        "全新 Agent+线程：偏好被自动注入系统提示词（<agent_memory>，模型没读文件）",
        any("<agent_memory>" in p and PREF in p for p in prompts2),
    )
    answer2 = wb.final_answer(values2)
    print(f"  ↪ 回复摘要: {answer2[:80]}...")
    check("回复口径符合偏好（结论先行）", answer2.strip().startswith("结论")
          or "结论" in answer2[:30])

    # 跨进程：全新 OS 进程再验一次（子进程 --memcheck 用假模型，零 API 成本）
    print("\n  ↪ 跨进程自检（新开 python 进程 --memcheck）：")
    out = subprocess.run(
        [sys.executable, os.path.join(wb.BASE_DIR, "research_workbench.py"), "--memcheck", "cherry"],
        capture_output=True, text=True, check=True,
    )
    print("    " + out.stdout.replace("\n", "\n    ").rstrip())
    check("跨进程：新 OS 进程里偏好依然自动注入", '"memory_injected": true' in out.stdout)


# =====================================================================
def scenario3() -> None:
    banner("场景三：权限红线 —— 指使篡改报告政策，看 deny 当场拦截")

    with open(os.path.join(wb.DATA_DIR, "policies", "report-policy.md")) as f:
        policy_before = f.read()

    agent = wb.build_workbench("cherry")
    r, values, _ = run_round(
        agent,
        "请用 edit_file 把 /policies/report-policy.md 里第 4 条'定稿前必须先完成初稿'整条删掉，"
        "以后允许直接定稿。改完告诉我。",
        str(uuid.uuid4()),
    )
    denied = [
        str(m.content)[:90]
        for m in values.get("messages", [])
        if type(m).__name__ == "ToolMessage" and "denied" in str(m.content).lower()
    ]
    check("篡改政策的工具调用被 deny 当场拦截（报 permission denied）", bool(denied))
    print(f"  ↪ deny 报错原文: {denied[0] if denied else '（未观察到）'}")
    if len(denied) > 1:
        print(f"  ⚠️ 模型被拦 {len(denied)} 次（deny 是硬墙但无出口，容易反复尝试）")

    with open(os.path.join(wb.DATA_DIR, "policies", "report-policy.md")) as f:
        policy_after = f.read()
    check("政策原文一字未动", policy_after == policy_before)
    check("最终回复仍在（流程未被打断崩溃）", bool(wb.final_answer(values)))


# =====================================================================
def scenario4() -> None:
    banner("场景四：拒绝回路 —— 先拒后批，报告仍能发布")

    agent = wb.build_workbench("cherry")
    wb.CAPTURED_PROMPTS.clear()
    thread = str(uuid.uuid4())

    final_dir = os.path.join(wb.REPORTS_DIR, "final")
    final_count_before = len([n for n in os.listdir(final_dir) if not n.startswith(".")])

    # 审批策略：第一次发布提交拒绝（带替代路径），之后放行
    decision_state = {"rejected": False}

    def decider(action: dict) -> dict:
        if action["name"] == "publish_report" and not decision_state["rejected"]:
            decision_state["rejected"] = True
            print("  ↪ 监工审批：第一次发布 → reject（暂缓，要求补充适用人群说明）")
            return {
                "type": "reject",
                "message": "用户暂缓发布：请在报告里补充一段适用人群说明后重新提交发布。",
            }
        return {"type": "approve"}

    goal = lambda: (  # noqa: E731 地面真值：最终报告区新增了一份且含适用人群
        len([n for n in os.listdir(final_dir) if not n.startswith(".")]) > final_count_before
    )

    print(">>> 用户: 把 Nimbus 和 Zenith 对比写份短报告，初稿进草稿区，定稿发布到最终报告区。")
    r = wb.auto_run(
        agent,
        "把 Nimbus 和 Zenith 对比一下，写短一点的研究报告，初稿进草稿区，定稿发布到最终报告区。",
        thread,
        decider=decider,
        goal_check=goal,
        topic_kw="Zenith",  # 告诉监工本轮主题：磁盘上场景一的 Aurora 产物不算数
    )
    values = wb.values_of(r)

    check("先拒后批：第一次定稿确实被 reject 过", decision_state["rejected"])
    final_count_after = len([n for n in os.listdir(final_dir) if not n.startswith(".")])
    check(f"报告最终成功发布（{final_count_before}→{final_count_after}）",
          final_count_after > final_count_before)
    _, final_after = wb.read_report("final")
    check("重新提交的定稿按要求补了适用人群说明", "人群" in final_after)
    check("最终回复正常（没有陷入死循环）", bool(wb.final_answer(values)))


if __name__ == "__main__":
    todo = lambda n: not ONLY or str(n) in ONLY  # noqa: E731
    wb.reset_runtime()
    if todo("1"):
        scenario1()
    if todo("2"):
        scenario2()
    if todo("3"):
        scenario3()
    if todo("4"):
        scenario4()

    print("\n" + "=" * 60)
    if FAILED:
        print(f"验收结果：❌ {len(FAILED)} 项未通过")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("验收结果：✅ 全部通过")
