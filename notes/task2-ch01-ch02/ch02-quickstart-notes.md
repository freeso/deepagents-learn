# 第 2 章笔记：快速上手 — 5 分钟构建你的第一个 Deep Agent

> - 篇章：认知篇（Task 2）
> - 课程链接：https://datawhalechina.github.io/deepagents-in-action/chapters/ch02-quickstart/
> - 学习日期：2026-09-17
> - 实验代码：[ch02-experiments.py](./ch02-experiments.py)（可复现）

## 一、本章做了什么

从安装到运行，完成第一个能干活（调工具）的 Deep Agent。核心就三步：

```python
agent = create_deep_agent(
    model=model,            # ChatOpenAI + base_url 接硅基流动
    tools=[get_weather],    # 普通 Python 函数
    system_prompt="...",    # 人设
)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
print(result["messages"][-1].content)
```

## 二、实验记录（均已跑通）

### 实验 1：Hello World 天气 Agent

- 工具：`get_weather(city: str)`，返回固定天气
- 提问「北京今天天气怎么样？」→ 正确调用工具 → 回答「今天北京天气晴朗！」
- 几行代码 Agent 就跑起来了，验证了 Harness 的「开箱即用」

### 实验 2：计算器 Agent（工具三要素）

- 两个工具：`calculate(expression)` + `convert_currency(amount, from_currency, to_currency="CNY")`
- 提问「100 美元换算成人民币，再乘以 1.08 通胀系数」
- 正确完成链式多步调用，最终回答 **777.6 元**（720 × 1.08 = 777.6 ✅）

### ⚠️ 实验 2 踩到的真实坑：小模型不可靠

第一次运行直接崩溃。Trace 显示：Qwen2.5-7B 想一步到位做链式计算，把 `convert_result`（一个变量名）当"数学表达式"传给了 `eval`，`NameError` 直接炸穿整个 invoke。

三层修复（越靠前越根本）：

1. **工具侧防御**：`calculate` 返回类型改 `str`，eval 前清空 builtins（`eval(expr, {"__builtins__": {}}, {})`），非法输入返回错误说明而不是抛异常——工具错误变成模型的「可读反馈」，它能自我纠正
2. **Docstring 收紧**：明确写「只接受纯数字算式，不接受变量名」——工具说明书越清晰，模型用得越准
3. **System Prompt 补规则**：要求「多步任务分步调用工具」

这个坑完美印证了课程「模型选择」一节的警告：**7B 小模型跑简单任务没问题，但链式多步工具调用容易翻车；复杂场景要上 GLM-5.2 级别的模型**。同时也演示了课程没细讲的一点：**工具函数应该永远不抛异常，把错误信息作为返回值喂回给模型**，这是 Agent 自我修复能力的基础。

## 三、核心知识点

### 工具三要素

| 要素 | 作用 | 缺失后果 |
| --- | --- | --- |
| 参数类型标注 | 告诉 Agent 传什么类型 | 传错类型 |
| Docstring | 告诉 Agent 何时用这个工具 | 不知道何时调用 |
| 默认值 | 标记可选参数 | 全部必填，增加出错面 |

口诀：**类型标注决定输入形态，docstring 决定调用时机，默认值减少必填项**。

### create_deep_agent() 自动给了什么

除了你传入的 `tools`，Agent 还自动获得：文件系统（read_file/write_file/grep/glob...）、子 Agent 委派（task）、上下文管理（SummarizationMiddleware）——这些就是第 1 章说的「Harness 固化共性」。一次 invoke 背后可能 10+ 次工具调用。

### v0.7 关键变化（与第 1 章呼应）

- 任务规划**不再默认启用**：要 `write_todos` 就显式传 `middleware=[TodoListMiddleware()]`。短任务省 token，长任务主动加脚手架
- 基础提示词默认为空：人设完全由应用自己定义
- 我在 Trace 分析里看到的 `FilesystemMiddleware / SubAgentMiddleware / SummarizationMiddleware` 就是这套中间件栈

### 模型接入三种方式

1. **OpenAI 兼容**（推荐）：`ChatOpenAI(model=..., base_url="https://api.siliconflow.cn/v1")` —— 换 base_url + key 就能切国内任意平台
2. Anthropic 兼容：`ChatAnthropic(base_url=...)`，硅基流动也支持
3. 字符串格式：`create_deep_agent(model="anthropic:claude-sonnet-4-6")`，适合直连官方 API

### LangSmith

设 `LANGSMITH_TRACING=true` 即自动上报（已在 Task 1 验证过全链路），无需改代码。

## 四、实战：研究助手（Tavily 真实搜索 + TodoListMiddleware）

代码：[ch02-research-agent.py](./ch02-research-agent.py)，问题可通过命令行参数覆盖。

### 双模型对比实验（同题：「什么是 LangGraph？它和 LangChain 是什么关系？」）

| | Qwen2.5-7B（免费） | GLM-5.2（753B） |
| --- | --- | --- |
| 工具调用 | 1 次（只写了 todos 就开答） | 4 次（中英双语各 2 轮搜索，共回传 ~28K 字符） |
| 是否真的搜索 | ❌ 一次都没搜 | ✅ 4 次 internet_search |
| 回答质量 | 凭训练记忆硬编，细节错误（如把 LangGraph 说成「基于 LangChain 技术的平台」） | 结构化报告：四大支柱、Pregel 起源、双向依赖关系、选型对比表 |
| 耗时 | 快但没用 | 稍慢但完整 |

### 两个关键观察

1. **小模型的失败方式很隐蔽**：Qwen2.5-7B 不是报错，而是「装作完成了任务」——调了一次 write_todos 就直接用训练记忆编答案，连一次搜索都没做。如果没有工具调用统计，单看最终回答根本发现不了。这就是课程反复强调的：**多步编排/中间件场景必须上大模型**，以及为什么 LangSmith Trace 重要——不看 Trace 你不知道 Agent 到底干了什么。
2. **TodoListMiddleware 的真实行为**：GLM 那次也没调 write_todos（任务简单，模型自行判断不需要规划），印证了课程 v0.7 提醒：「模型会根据任务决定是否实际调用工具，不能把流程图当固定协议」。

## 五、小结

1. Deep Agent 最小骨架 = 模型 + 工具函数 + 提示词，几行代码起步
2. 工具设计是 Agent 质量的第一决定因素：三要素 + 永不抛异常
3. 模型能力要与任务复杂度匹配：学习用免费 7B，复杂编排上大模型——实战对比实验给出了实证（见上表）
4. 下一章深入虚拟文件系统——本章 invoke 背后自动发生的「上下文管理」的正式展开

## 六、TODO

- [x] 研究助手实战（双模型对比，见上）
- [ ] 用 GLM-5.2 重跑实验 2，对比小模型行为差异（已部分实证：见实战对比）
- [ ] 研究 Agent 完整版（Tavily 真实搜索）已在 research_deepagent 模板验证过，不重复做
