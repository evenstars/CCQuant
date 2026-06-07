# CCQuant

低频量化交易系统 — **美股纯多头横截面动量**(月度调仓)+ 大盘趋势刹车 + 分批建仓。

## 这是什么
在 S&P 500 里按 **12-1 动量**排名,等权买入最强的约 15 只,每月调仓;当 SPY 跌破 200 日均线时降仓以规避动量崩盘。面向 **IBKR 应税账户**,小资金(<$25k)起步,设计含完整的回测、风控、实盘、监控与税务处理。

## 设计与计划
- 完整设计:[`docs/design.md`](docs/design.md)(策略规范、回测、风控、IBKR实盘、监控、税务、项目结构、里程碑)
- 里程碑速览:[`docs/plan.md`](docs/plan.md)

## 项目结构
```
config/          策略与风控参数、凭证模板
ccquant/
  data/          数据加载、复权、股票池、缓存
  strategy/      因子、排名、组合构建(回测/实盘共用)
  backtest/      撮合引擎、绩效、归因、报告
  risk/          风控检查、断路器、税务/wash-sale
  live/          IBKR (ib_insync) 接入、下单、对账
  monitor/       健康检查、告警、dashboard、报告
  utils/         配置、日志、日历、持久化
tests/           单元测试
notebooks/       研究与回测探索
```

## 快速开始(规划中)
1. `cp config/secrets.example.env config/secrets.env` 并填入 IBKR / 数据源凭证。
2. 安装依赖(M0 起补充 `pyproject.toml`)。
3. 按 `docs/plan.md` 从 M0/M1 推进。

## 现状
设计阶段(v0.1)。代码模块为待实现骨架。

---
*免责声明:本项目为工程与研究用途,不构成投资建议。*
