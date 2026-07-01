# Fork 说明

这是 Grok2API 部署改造版的已脱敏公开快照，通常配合 New API 中转站作为独立 Grok 上游使用。

本 fork 的本地改动包括：

- Console.x.ai SSO 模型支持
- Grok 4.20 multi-agent 模型别名
- SSO token 池选择和冷却处理
- token 级 429 失败后的重试行为
- Console 重试与 SSO token 提取测试

## 未包含内容

以下运行时文件已被排除，不能提交到公开仓库：

- `data/token.json`
- `data/setting.toml`
- `data/temp/`
- `logs/`
- 生成的图片和视频
- `*.bak-*` 等本地备份文件

运行时数据请通过管理后台或部署脚本在服务器本地创建。
