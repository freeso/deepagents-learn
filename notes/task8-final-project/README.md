# 深研工坊 Deep Research Workbench —— Task 8 结营项目

> 状态：🚧 进行中——四场景验收正在迭代调优，首次全绿后本行会更新为可复现的运行说明。**（注：结构设计与四场景验收框架已就绪）**

一个终端版的"调研 → 委派 → 成稿 → 审批 → 归档"Agent 应用：你把一个研究需求丢进去，它自动拆任务、把资料收集委派给专职子 Agent、按《报告写作规范》技能成稿、把初稿写进草稿区，提交定稿时停下等你审批——通过后归档发布，并且**跨进程记得你的写作偏好**。

全程只用免费的 Qwen2.5-7B 与确定性模拟数据（手机参数/口碑库），无任何真实副作用，所有行为用确定性锚点验收。

## 怎么跑

```bash
# 环境：项目根目录的 .venv（见仓库顶层说明），API key 走 SILICONFLOW_API_KEY
cd notes/task8-final-project

../../../.venv/bin/python verify.py            # 自动验收（四场景 + 跨进程自检，全绿即通过）
../../../.venv/bin/python verify.py 1          # 只跑场景一
../../../.venv/bin/python research_workbench.py        # 交互聊天（结营演示入口）
../../../.venv/bin/python research_workbench.py --memcheck   # 零成本跨进程记忆自检
```

## 六项课程能力在项目里做什么

| 能力（章节） | 项目里的角色 | 验证锚点 |
| --- | --- | --- |
| 任务规划 ch04 | `TodoListMiddleware`：研究任务自动拆四步（委派/读规范/初稿/定稿） | `write_todos` 出现在工具调用链 |
| 子 Agent ch05 | `data-collector` 专职查参数库和口碑库；主 Agent **没有**数据工具，只能委派 | 查询日志真实执行；主上下文无原始 JSON 字段（隔离） |
| Skills ch07 | `report-writing` 技能：书脊自动注入，主 Agent 匹配后按需读正文 | 偷拍的系统提示词含技能名 + 主链 `read_file SKILL.md` |
| 长期记忆 ch08 | `/memories/` 路由到 **JSON 落盘 Store**（课程 InMemoryStore 的持久化升级） | store.json 落盘；新 Agent / 新线程 / **新 OS 进程**都自动注入偏好 |
| HITL ch09 | 定稿写入 `/reports/final/**` 自动弹人工审批（approve/reject） | interrupt 出现；approve 后落盘、reject 后不落盘且听劝重写 |
| 权限 ch03/ch11 | `/policies/**`、`/skills/**` deny 禁写；草稿区放行；规则首中即生效 | 篡改政策被拦 + 原文一字未动；初稿免审批直接落盘 |

## 架构与虚拟文件布局

```
用户（终端 / 验收脚本）
      │
      ▼
┌── 主 Agent（总编辑）─────────────────────────────┐
│ ① write_todos 拆任务（ch04）                      │
│ ② task 委派 data-collector（ch05）                │
│      └─ 子 Agent 查库 → 原始 JSON 放 /workspace/  │
│         只回带关键数字的中文摘要（上下文隔离）      │
│ ③ 读 /skills/report-writing/SKILL.md（ch07 L2）   │
│    └─ 技能要求再读 /policies/report-policy.md     │
│ ④ 初稿 → /reports/drafts/（放行）                │
│ ⑤ 定稿 → /reports/final/（ch09 interrupt 审批）  │
│ ⑥ 记偏好 → /memories/（ch08 落盘 Store）          │
└── 权限墙（ch11）：/policies、/skills 禁写 ────────┘
```

`CompositeBackend` 按路径前缀路由到四个“仓库”，前缀剥离后各归各家：

| 虚拟路径 | 后端 | 磁盘位置 | 生命周期 |
| --- | --- | --- | --- |
| `/workspace/**` | StateBackend | （内存 state） | 会话结束即清空（“草稿桌”） |
| `/memories/**` | StoreBackend ← JsonFileStore | `.tmp/workshop/store.json` | **跨进程长存** |
| `/skills/**` | FilesystemBackend | `agent_data/skills/`（随仓库提交） | 只读 |
| `/policies/**` | FilesystemBackend | `agent_data/policies/`（随仓库提交） | 只读（deny） |
| `/reports/**` | FilesystemBackend | `.tmp/workshop/reports/{drafts,final}` | 草稿放行，定稿审批 |

## 一个关键工程点：记忆持久化

课程第 8 章的 `InMemoryStore` 进程一退就失忆。本项目写了一个 ~60 行的 `JsonFileStore`（实现 `BaseStore.batch` 契约，`put` 即落盘），使“重启程序还记得你的偏好”成为真实能力——验收里的跨进程自检就是为此设计的（`--memcheck`：新开 OS 进程 + 假模型，零 API 成本验证注入）。

另两个源码级细节（前两章实验的结论在本项目复用）：`MemoryMiddleware` 必须手动构造并放在偷拍中间件**外层**，否则拍不到注入结果；权限规则按声明顺序**首中即生效**，deny 政策区、interrupt 定稿区、放行草稿区三条规则各安其位。

## 演示

结营演示实录见 [demo-transcript.md](demo-transcript.md)（五分钟版：完整流水线 → 记忆跨进程 → 权限红线）。

## 目录

```
task8-final-project/
├── README.md                  # 本文件
├── research_workbench.py      # 主程序（Agent 工厂 + 交互聊天 + memcheck）
├── verify.py                  # 自动验收（四场景，确定性锚点）
├── demo-transcript.md         # 演示实录
├── agent_data/                # 只读静态资料（随仓库提交）
│   ├── skills/report-writing/SKILL.md
│   └── policies/report-policy.md
└── .tmp/workshop/             # 运行时数据（gitignore：store.json、报告产物）
```
