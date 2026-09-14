# Task 1 踩坑记录：依赖安装、技能安装与 LangSmith CLI 四连坑

> - 篇章：准备篇（AgentSeek 环境搭建）
> - 相关章节：AgentSeek 准备篇（上）；课程实验模板 `deepagents/research`
> - 日期：2026-09-14
> - 环境：macOS 14.4 (arm64) · CPython 3.14 · uv 0.12.13

## 一、问题现象

用 AgentSeek 创建研究模板后，执行依赖同步：

```bash
agentseek task sync   # 内部等价于 uv sync
```

报错：

```
error: Distribution `pylibseekdb==1.4.0.post1 @ registry+https://pypi.org/simple`
can't be installed because it doesn't have a source distribution or wheel
for the current platform

hint: You're on macOS (`macosx_14_0_arm64`), but `pylibseekdb` (v1.4.0.post1)
only has wheels for: manylinux_2_28_aarch64, manylinux_2_28_x86_64, macosx_15_0_arm64
```

## 二、根因分析

依赖链是：

```
agentseek-api[embedded]==0.2.3
  └─ langchain-oceanbase[pyseekdb]==0.6.3
       └─ pyseekdb
            └─ pylibseekdb   ← 崩在这里
```

`pylibseekdb` 是 OceanBase seekdb（AI 原生搜索数据库）的 Python 绑定，官方要求：

- Linux x86_64 / aarch64（glibc ≥ 2.28）
- **macOS arm64 ≥ 15.6**（wheel 只发布 `macosx_15_0_arm64` 及以上）

我的系统是 macOS 14.4，新旧版本（1.2.0 也一样）都没有对应 wheel，**降级也解决不了**。这是模板对旧 macOS 的兼容缺陷，不是配置错误。

## 三、解决方案：切换到 SQLite 后端

读 agentseek-api 0.2.3 源码（`agentseek_api/core/database.py` 的 `DatabaseManager.initialize()`）发现它有两条持久化路径：

| 配置 | 路径 | 依赖 |
| --- | --- | --- |
| `SEEKDB_EMBED=true` | 嵌入式 seekdb（OceanBaseCheckpointSaver + OceanBaseStore） | 需要 pylibseekdb |
| `SEEKDB_EMBED=false` + `METADATA_DB_BACKEND=sqlite` | SQLite（SqliteCheckpointSaver + SqliteStore，LangGraph 检查点为内存 InMemorySaver） | 只需要 aiosqlite，已内置 |

对本地学习/实验来说 SQLite 后端功能等价（checkpoint 持久化、跨会话 run 记录都有），完全够用。

### 具体修改（3 个文件）

**1. `pyproject.toml`**：去掉 embedded extra，并加 uv 配置：

```toml
dependencies = [
    # ...
    "agentseek-api==0.2.3",   # 原来是 agentseek-api[embedded]==0.2.3
]

[[tool.uv.index]]
url = "https://pypi.org/simple"
default = true

[tool.uv]
environments = [                 # 只解析当前平台，避免 win32 分支解析失败
    "sys_platform == 'darwin' and platform_machine == 'arm64'",
    "sys_platform == 'linux'",
]
```

**2. `.agentseek/lifecycle.toml`**：`SEEKDB_EMBED` 默认值 `true` → `false`，新增：

```toml
[env.METADATA_DB_BACKEND]
required = false
default = "sqlite"

[env.METADATA_DB_URL]
required = false
default = "sqlite+aiosqlite:///.agentseek/metadata.db"
```

**3. `.env.example`**：同步更新持久化段，`SEEKDB_EMBED=false` + 上面两个变量，并注明原因和恢复方法（升级到 macOS 15.6+ 后可改回 embedded）。

### 额外坑：uv 镜像环境变量

在配置了企业内网 PyPI 镜像的终端里跑 `uv sync` / `uv run` 会报 `agentseek-api was not found in the package registry`——因为内网镜像没有收录这个新包，且 `UV_DEFAULT_INDEX` 环境变量优先级高于 pyproject 里的 `[[tool.uv.index]]`。解决：

```bash
unset UV_DEFAULT_INDEX UV_INDEX_URL PIP_INDEX_URL UV_INSECURE_HOST
uv sync
```

注意：改完 `pyproject.toml` 后如果 lock 里有旧的 win32 解析残留（`grep -c win32 uv.lock` 数量异常），需要 `rm uv.lock && uv sync` 重新生成。

### 额外坑 2：`npx skills add` 从 GitHub 克隆超时

准备篇（下）执行 `npx skills add ob-labs/agentseek --list` 时，skills CLI 直接完整克隆 GitHub 仓库，国内网络下 300 秒超时失败：

```
Clone timed out after 300s.
```

解决：绕开 CLI 的完整克隆，浅克隆到本地再喂本地路径：

```bash
git clone --depth 1 https://github.com/ob-labs/agentseek.git /tmp/agentseek
npx skills add /tmp/agentseek --skill langchain-dev-guide --skill langsmith-trace -g -y
```

- `--depth 1` 只拉最新一层（约 67M），几十秒完成
- `-g` 装到全局（`~/.agents/skills/`，对 Codex、Cursor 等以 universal 方式生效；Claude Code、OpenClaw 以 symlink 方式生效）
- 安装时若出现 `PromptScript does not support global skill installation` 报错可忽略，不影响主流载体
- 备选方案：加 `SKILLS_CLONE_TIMEOUT_MS=600000` 放宽超时；或有代理时 `git config --global http.proxy` 一劳永逸

装完验证：`ls ~/.agents/skills/` 应能看到 `langchain-dev-guide` 和 `langsmith-trace`。

> 补充：课程准备篇（下）推荐的是**项目级安装**（不加 `-g`，装到项目的 `.agents/skills/`）。全局和项目级不冲突，编码助手优先读项目级。项目级安装用同样的浅克隆方式：
> ```bash
> cd research_deepagent
> npx skills add /tmp/agentseek --skill langchain-dev-guide --skill langsmith-trace -y
> # 装到 ./research_deepagent/.agents/skills/，skills list 可验证
> ```
> 更新命令：项目级 `npx skills update -p`，全局 `npx skills update -g`。

### 额外坑 3：LangSmith CLI「not authenticated」

新终端执行 `langsmith --format pretty project list` 报：

```
not authenticated; run 'langsmith auth login', set LANGSMITH_API_KEY, or pass --api-key
```

原因：CLI 从**环境变量**读凭证，`.env` 文件不会自动加载，新开终端后 shell 里没有 `LANGSMITH_API_KEY`。

解决（课程 5.1 节的做法，在项目目录下）：

```bash
set -a
source .env
set +a
```

`set -a` 让 source 进来的变量自动导出为环境变量，`set +a` 恢复。只对当前会话有效，新终端要重新 source。长期方案：`langsmith auth login`（推荐，key 存本地凭证）或把 export 写进 `~/.zshrc`。

安装方式注意：LangSmith CLI 要用官方脚本 `curl -fsSL https://cli.langsmith.com/install.sh | sh`（装到 `/usr/local/bin/langsmith`）。**PyPI 的 `langsmith` 包是 Python SDK，不提供这个 CLI**，不要用 `uv tool install langsmith` 装。

### 额外坑 4：`trace list --name` 报 400 Bad Request

CLI v0.2.54 执行：

```bash
langsmith trace list --project deepagents-course --name research --include-metadata --limit 5
```

报错：

```
400 Bad Request: Failed to parse filter: Invalid argument count:
"search" requires 1 arguments, got 2
```

原因：该版本 `--name` 参数生成的过滤语法有 bug（`search` 算子参数数量拼错），是 CLI 缺陷，不是使用姿势问题。

解决：去掉 `--name`，用 NAME 列肉眼过滤：

```bash
langsmith trace list --project deepagents-course --include-metadata --limit 5
# 研究模板的根 Trace 名字显示为 LangGraph (chain)，复制 TRACE ID 进入下一步
langsmith trace get <trace-id> --project deepagents-course --include-metadata
```

需要按名字精确过滤时改用 `run list`（其 filter 解析正常）：`langsmith run list --project ... --include-metadata`。官方出新版后可重跑安装脚本升级。

## 四、验证结果

- `uv sync` 成功，105 个依赖全部安装（deepagents 0.7.13、agentseek-api 0.2.3）
- `pylibseekdb` 已不在环境中（`find_spec` 确认）
- `DatabaseManager` 可导入
- `research_deepagent.agent` 可导入，`graph` 为 `CompiledStateGraph`

## 五、经验总结

- 遇到 wheel 平台不兼容，先查 PyPI JSON API（`https://pypi.org/pypi/<pkg>/json`）确认各版本支持的平台矩阵，判断"降级"是否可行，不要盲目试
- `agentseek-api[embedded]` 的 embedded 是可选 extra，不代表框架必需 seekdb——读源码里的配置分支比猜靠谱
- 企业内网 PyPI 镜像对小众新包常有延迟，`uv` 场景下项目级 `[[tool.uv.index]]` 是好习惯，但注意环境变量优先级问题
- 这个坑也侧面说明了 DeepAgents 生态的**存储后端可插拔**设计（第 3 章会正式学到）：换掉 seekdb 对上层应用零改动

## 六、TODO

- [ ] 填 .env（OPENAI_API_KEY、TAVILY_API_KEY）后跑 `agentseek doctor` + `agentseek dev`
- [ ] 浏览器打开 http://127.0.0.1:5174 冒烟测试研究 Agent
