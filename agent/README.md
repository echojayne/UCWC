# UCWC Tool-Calling Agent

一个基于 LLM 的 UCWC 工具调用型 Agent 框架。当前版本保留通用 ReAct loop，但工具集已经替换为 UCWC/NL2SQL 工具，不再注册文件读取、文件编辑或 shell 命令工具。

## 项目结构

```
.
├── main.py             # 入口：组装 UCWC Agent、启动 CLI 交互循环
├── agent.py            # Agent 核心：ReAct 主循环逻辑
├── llm.py              # LLM 客户端：封装 OpenAI SDK，连通 DeepSeek API
├── protocol.py         # 数据类型：UserMessage / AssistantMessage / ToolMessage / ToolCall
├── context.py          # 上下文管理：对话历史记录与待处理工具调用追踪
├── tools.py            # 工具注册表：工具定义、注册与执行
├── ucwc_tools.py       # UCWC 工具：Schema link、读表、写表、SQL 生成/执行/修复、验证、commit
├── ucwc_prompts.py     # UCWC session-level 控制提示词
├── config.toml         # LLM 配置：API Key、模型、推理参数
├── pyproject.toml      # 项目元信息与依赖
└── prompt_factory/     # 提示词工厂（系统提示词模板库）
```

## Agent Loop 工作流程

```
用户输入
  │
  ▼
┌─────────────────────────────────────────┐
│  while step < max_steps (默认 20):      │
│                                         │
│  1. LLM.complete(                      │
│       system_prompt,                   │
│       conversation_history,  ← 完整历史 │
│       tool_definitions       ← 工具列表 │
│     )                                  │
│                                         │
│  2. 记录 assistant 消息到对话历史       │
│                                         │
│  3. 有 tool_calls?                     │
│     ├─ 是 → 执行工具 → 记录结果 → 继续  │
│     └─ 否 → return content（最终输出）   │
└─────────────────────────────────────────┘
  │
  ▼
最终回答返回给用户
```

核心思想：LLM 在每一步可以选择**调用工具执行操作**（如读取文件、运行命令），也可以选择**直接给出文本回答**。工具执行的结果被追加到对话历史中，循环继续，LLM 可以在下一轮基于新信息继续推理，直到任务完成。

## 可用工具

| 工具 | 功能 | 安全措施 |
|------|------|----------|
| 工具 | 功能 |
|------|------|
| `ucwc_schema_link` | 读取 SQLite schema、字段、行数和推荐 join keys |
| `ucwc_read_table` | 对指定表做 bounded read |
| `ucwc_sql_generate` | 根据 query goal 生成 UCWC 状态查询 SQL |
| `ucwc_sql_execute` | 校验并执行 read-only SELECT/WITH |
| `ucwc_sql_correct` | 根据失败 SQL 和 query goal 生成修复 SQL |
| `ucwc_verify_config_plan` | 调用确定性 UCWC verifier |
| `ucwc_commit_config_plan` | 仅在 verifier 通过后写入 `config_history` |
| `ucwc_write_table` | 受限写入 `config_history` |

## 快速开始

### 环境要求

- Python >= 3.11
- 有效的 DeepSeek API Key（或兼容 OpenAI 接口的其它服务）

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd custom-implementation

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install openai
```

### 配置

编辑 `config.toml`，填入你的 API Key：

```toml
[llmcfg]
api_key = "your-api-key-here"
base_url = "https://api.deepseek.com"
model = "deepseek-v4-pro"

[llmcfg.request]
stream = false
reasoning_effort = "xhigh"

[llmcfg.request.extra_body.thinking]
type = "enabled"
```

### 运行

```bash
python main.py --db-path ../outputs/scenario_smoke/network_state.sqlite --target-ue ue_001
```

交互示例：

```
You: 列出当前目录有哪些文件
Assistant: 当前目录包含 main.py, agent.py, llm.py, protocol.py 等文件...

You: 读一下 agent.py
Assistant: agent.py 定义了 Agent 类和 AgentConfig，核心逻辑在 run() 方法中...

You: exit
```

输入 `exit` 或 `quit` 退出，按 `Ctrl+C` 也可退出。

## 核心设计

### 1. ReAct 模式

Agent 不直接输出最终答案，而是通过"思考 → 行动 → 观察"的循环逐步完成任务：

- **思考**：LLM 分析当前对话历史，决定下一步（调用工具 or 输出答案）
- **行动**：执行工具调用，如读取文件、运行命令
- **观察**：将工具返回结果追加到对话历史，进入下一轮思考

### 2. 安全性

所有工具有严格的沙箱限制：

- 文件操作路径必须位于 workspace 根目录内，禁止符号链接遍历
- 命令执行不使用 shell，参数长度和数量受限，有独立超时控制
- 所有输出有大小截断，防止 LLM 上下文溢出

### 3. 对话状态管理

`ContextManager` 保证对话历史的完整性：

- 待处理的工具调用必须全部完成后才能记录新的用户/助手消息
- 自动检测重复 ID、名称不匹配等异常

### 4. 工具扩展

添加新工具只需三步：

1. 实现一个 `handler(arguments: dict) -> str` 函数
2. 构造 `ToolDefinition(name, description, parameters, handler)`
3. 在 `main.py` 的 `build_agent()` 中调用 `tools.register(your_tool)`

## License

MIT
