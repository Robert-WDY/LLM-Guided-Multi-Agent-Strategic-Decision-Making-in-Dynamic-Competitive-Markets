## 变更内容

<!-- 说明解决的问题和主要改动。 -->

## 影响范围

- [ ] 市场模型 / 配置
- [ ] API / Agent 协议
- [ ] 前端
- [ ] 测试 / 校准
- [ ] 文档

## 验证

- [ ] `python -m pytest`
- [ ] `npm run lint`（`frontend/`）
- [ ] `npm run build`（`frontend/`）
- [ ] 固定 Seed 配对结果已检查（市场模型变化时）
- [ ] 200 Seed 校准已检查（策略或收益变化时）

## 协议与安全

- [ ] 没有提交真实 Token、`.env`、日志、缓存或构建产物
- [ ] Schema/环境版本已按需提升
- [ ] Agent 仍不能绕过 Controller 直接结算
- [ ] 相关文档已同步更新
