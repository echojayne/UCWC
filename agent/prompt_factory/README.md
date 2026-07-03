# Prompt Factory

该目录将原始提示词目录整理为可独立复用的 Markdown 模板。

## 整理原则

- 按用途拆分主题，避免把互相冲突的系统规则一次性拼接。
- 合并完全重复的模板，过滤测试断言、解析错误和无意义源码碎片。
- 移除产品名、厂商名、模型版本、仓库路径和内部来源标识。
- 保留模板变量、工具名称和行为约束，方便继续组合与二次加工。

## 目录

| 文件 | 主题 | 条目数 |
| --- | --- | ---: |
| [01-base-agent.md](./01-base-agent.md) | 基础代理 | 4 |
| [02-personality.md](./02-personality.md) | 交互风格 | 2 |
| [03-collaboration.md](./03-collaboration.md) | 协作模式 | 5 |
| [04-agent-orchestration.md](./04-agent-orchestration.md) | 多代理编排 | 20 |
| [05-context.md](./05-context.md) | 上下文管理 | 14 |
| [06-tools.md](./06-tools.md) | 工具使用 | 24 |
| [07-tool-discovery.md](./07-tool-discovery.md) | 工具发现 | 2 |
| [08-skills.md](./08-skills.md) | 技能系统 | 4 |
| [09-permissions.md](./09-permissions.md) | 权限与沙箱 | 15 |
| [10-review.md](./10-review.md) | 审查与风控 | 21 |
| [11-goals.md](./11-goals.md) | 目标管理 | 5 |
| [12-memory.md](./12-memory.md) | 记忆管理 | 9 |
| [13-compaction.md](./13-compaction.md) | 上下文压缩 | 2 |
| [14-realtime.md](./14-realtime.md) | 实时交互 | 3 |

## 使用建议

- 从基础代理提示词中选择一个主模板，不要同时加载多个完整变体。
- 根据任务按需追加协作、权限、工具、记忆或审查模块。
- 补充片段通常依赖对应工具或运行时字段，接入前确认变量名和工具接口一致。

共整理 130 条内容；合并 8 条重复内容，过滤 41 条空白、测试或低价值碎片。
