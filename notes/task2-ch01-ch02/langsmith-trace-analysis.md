# LangSmith Trace 瓶颈分析报告（准备篇 5.3 实操 / Task 2 素材）

> - 分析日期：2026-09-14
> - 分析工具：`langsmith-trace` 技能 + LangSmith CLI（v0.2.54）

## 基本信息

| 项目 | 值 |
| --- | --- |
| LangSmith 项目 | deepagents-course |
| trace_id | 01a0a00b-4634-7152-810e-8a5018e8a6d6 |
| 根 Run | LangGraph (chain)，总耗时 201.52s |
| 错误状态 | 全部 run error: null（无错误） |
| 模型 | deepseek-ai/DeepSeek-V4-Flash |
| 研究问题 | LangGraph 1.0 相比 0.x 新增了什么（带引用） |

## Trace 层级结构（四层区分）

```text
LangGraph (chain) [201.52s]                    ← ① 根 research 流程
├── PatchToolCallsMiddleware.before_agent
├── model (chain) ×7 轮                        ← 根 Agent 的 LLM 调用
│   └── ...middleware stack...
│       └── ChatOpenAI (llm)                   ← ③ 叶子模型调用
├── tools (chain)
│   └── think_tool / write_file / read_file    ← ④ 实际工具调用（本地，<1ms）
├── tools (chain) [111.55s]
│   └── task (tool)                            ← ② research-agent 子 Agent 包装层
│       └── research-agent (chain) [111.55s]
│           ├── model ×6 轮
│           │   └── ChatOpenAI (llm)           ← ③ 叶子模型调用（子 Agent 内）
│           ├── tavily_search ×6               ← ④ 实际工具调用（网络搜索）
│           ├── think_tool ×1
│           └── read_file ×1
```

正确排除了三类**非瓶颈包装层**：根 Trace 聚合层（201.52s）、task 子 Agent 包装（111.55s，输入为 subagent_type: "research-agent"）、各 Middleware 的 awrap_* 透传层。

## 最慢的叶子模型调用（run_type=llm）

| 排名 | run_id | 耗时 | 位置 | 输入概要 |
| --- | --- | --- | --- | --- |
| 1 | …add6… | 29.14s | research-agent 第 3 轮 | System + 摘要（含 4 轮搜索全文，约 30K+ tokens） |
| 2 | …291e… | 18.18s | research-agent 第 2 轮 | 摘要 + 搜索结果全文 |
| 3 | …b34a… | 17.84s | 根 Agent 第 4 轮 | 完整对话历史 + 工具结果 |

## 最慢的实际工具调用（排除 task 包装）

| 排名 | run_id | 耗时 | 工具 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | …79d8… | 11.31s | tavily_search | query "LangGraph 1.0 announcement blog.langchain.dev release"，返回 3 条结果含完整网页正文（langchain.com 博客 >15K 字符，clickittech.com >30K 字符） |
| 2 | …79d9… | 9.86s | tavily_search | GitHub release notes 查询 |
| 3 | …b6c9… | 9.49s | tavily_search | LangGraph 1.0 发布博客 |

## 根因链

```text
tavily_search 返回完整网页正文（单条 >30K 字符）
  → 大量文本灌入 LLM 上下文（约 30K+ tokens）
    → LLM prefill 阶段耗时飙升（首 token 延迟 2.8s，总耗时 29s）
      → 6 轮搜索 × ~9s + 6 轮 LLM × ~15s ≈ 135s
        → 占 research-agent 111.55s 的绝大部分，形成正反馈循环
```

## 优化建议（按证据对应）

1. **工具层截断**：tavily_search 返回前对每条结果截断（如保留前 2000 字符），同时降低工具传输耗时和 LLM prefill 耗时——预计总耗时 201s → 80-100s
2. **搜索参数收敛**：`search_depth="basic"` 替代 advanced、`max_results` 限制在 3 条以内、如支持 `include_answer=True` 用 AI 摘要替代全文
3. **上下文管理**：调低 SummarizationMiddleware 的摘要阈值，减少历史消息体积
4. **模型侧**：延迟敏感场景可换更快的模型变体或降低 max_tokens（但根因在上下文长度，模型不是第一优先级）

## 对照课程 5.4 检查项自评

| 检查项 | 完成情况 |
| --- | --- |
| Trace 选择 | ✅ 最近一次已完成的 research Trace（201.52s，无错误） |
| 层级区分 | ✅ 明确区分根流程 / 子 Agent / LLM / 工具四层 |
| 模型瓶颈 | ✅ 最慢叶子 ChatOpenAI 29.14s（run_id 有据） |
| 工具瓶颈 | ✅ 最慢实际工具 tavily_search 11.31s（已排除 task 包装） |
| 证据 | ✅ 给出 run_id、耗时、输入输出检查结果（run get --include-io） |
| 建议 | ✅ 与证据对应（截断正文 / 收敛搜索参数），非空泛的"换更快模型" |

## 我的收获

- Agent 的耗时大头往往不在"网络慢"，而在**上下文工程**：工具返回什么粒度的数据，直接决定了 LLM 的 prefill 成本
- task / middleware 这类包装层在 Trace 树上耗时很大，但它们是"容器"不是"执行单元"——分析瓶颈必须下钻到 run_type=llm / tool 的叶子节点
- 这正好预习了核心篇的两个主题：第 3 章（虚拟文件系统与上下文管理——工具返回多少内容进上下文）和第 5 章（子 Agent 与上下文隔离——task 包装层的意义）
