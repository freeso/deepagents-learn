# DeepAgents 学习笔记

Datawhale 开源课程 [Deep Agents 实战](https://github.com/datawhalechina/deepagents-in-action) 组队学习笔记。

- 课程地址：https://github.com/datawhalechina/deepagents-in-action
- 课程文档站：https://datawhalechina.github.io/deepagents-in-action
- 视频学习参考：[B站合集](https://space.bilibili.com/28357052/lists/7757577?type=season)
- 组队学习计划：见课程组内部文档（22 天 8 个 Task，详见下表）

## 组队学习计划（2026-09-14 开始，共 22 天）

课程说明：Deep Agents 是 LangChain 开源的 Agent 开发框架，适合构建能够拆解复杂任务、持续调用工具并自主推进工作的智能体。本课程以 **0.7 版本**为基线。

> 截止时间均为次日 03:00（即当天晚上加班也算第二天）

| Task | 内容 | 天数 | 截止时间 | 状态 |
| --- | --- | --- | --- | --- |
| Task 1 | 环境准备（Python、模型 API、Git、LangSmith 环境自检） | 1 天 | 09-15 | ✅ |
| Task 2 | 第 1 章 Agent Framework、Runtime 与 Harness；第 2 章快速上手 | 3 天 | 09-18 | ✅ |
| Task 3 | 第 3 章 虚拟文件系统与存储后端 | 3 天 | 09-21 | ✅ |
| Task 4 | 第 4 章 任务规划与分解 | 3 天 | 09-24 | ⬜ |
| Task 5 | 第 5 章 子 Agent 与上下文隔离 | 2 天 | 09-26 | ⬜ |
| Task 6 | 第 6 章 异步子 Agent、第 7 章 Skills | 4 天 | 09-30 | ⬜ |
| Task 7 | 第 8 章 长期记忆、第 9 章 Human-in-the-Loop | 4 天 | 10-04 | ⬜ |
| Task 8 | 综合项目开发、Demo 结营 | 2 天 | 10-06 | ⬜ |

状态图例：⬜ 未开始 / 🔵 进行中 / ✅ 已完成

### 各 Task 作业要求（评审标准）

| Task | 评审要求 | 优秀作业标准 |
| --- | --- | --- |
| Task 1 | 完成 Python、模型 API、Git 和 LangSmith 环境自检，并提交结果 | 环境自检清单、运行日志和问题说明完整 |
| Task 2 | 能运行第一个 Deep Agent，添加自定义工具并查看一次 Trace | 可运行代码、清晰 Trace 和简短实验总结 |
| Task 3 | 使用文件工具，并完成至少两种 Backend 的对比实验 | 文件系统实验记录、对比结论和可复现代码 |
| Task 4 | 使用 Todo 机制拆解复杂任务，并完成一个小型研究文档或 Agent | 规划过程、执行结果和问题复盘齐全 |
| Task 5 | 创建两个职责不同的子 Agent，并说明上下文隔离方式 | 多 Agent 协作示例、角色设计和运行结果清晰 |
| Task 6 | 体验并行任务、追加指令、取消或恢复，编写并调用一个最小可用的 SKILL.md | 异步任务记录、自定义 Skill 和调用日志完整 |
| Task 7 | 跨两轮对话验证记忆，并为敏感工具配置审批、修改和拒绝流程 | 两轮对话记录、记忆说明和审批流程演示完整 |
| Task 8 | 综合使用至少 3 项课程能力，提交可运行项目和 README，完成演示与反馈 | 项目可运行、README 清晰，并有完整演示材料 |

## 笔记组织方式

- 按 Task 组织，每个 Task 一个子目录（作业成果放 `deliverables/`）：

```
notes/
├── task1-env-setup/          # Task 1 环境准备
├── task2-ch01-ch02/          # Task 2 第1-2章
├── task3-ch03-filesystem/    # Task 3 第3章
├── task4-ch04-planning/      # Task 4 第4章
├── task5-ch05-subagents/     # Task 5 第5章
├── task6-ch06-ch07/          # Task 6 第6-7章
├── task7-ch08-ch09/          # Task 7 第8-9章
├── task8-final-project/      # Task 8 综合项目
└── template.md               # 笔记模板
```

- 新建笔记请复制 `notes/template.md` 模板
- 开始/完成某个 Task 后，更新上方状态列

## 学习进度时间线

- 09-14 开营，Task 1 环境准备启动
- 09-14 Task 1 完成：doctor 22 项全过、前后端启动验证、准备篇技能安装、LangSmith CLI 就绪（四个坑见 task1 笔记）；准备篇 5.3 Trace 瓶颈分析实操完成（见 task2 目录）
- 09-14 第 1 章白话笔记完成（三层架构 + Context Engineering，见 task2 目录）；macOS 升级后恢复 embedded seekdb 模式验证通过
- 09-17 第 2 章快速上手完成：Hello World + 计算器双实验跑通；踩到小模型链式调用坑并三层修复（工具防御/docstring/prompt），笔记与可复现代码在 task2 目录。**Task 2 完成**
- 09-18 第 3 章虚拟文件系统完成：5 个实验覆盖 4 种 Backend（State/Filesystem/Store/Composite）+ 权限拦截 + 大结果自动卸载验证，含 8 个踩坑记录，见 task3 目录。**Task 3 完成**（提前 3 天）
