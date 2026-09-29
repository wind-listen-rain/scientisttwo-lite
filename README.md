# ScientistTwo-lite：照着 Google ScientistTwo 论文还原的自主科研流水线

> **English summary.** A reduced, open re-implementation of the multi-agent research pipeline described in
> Google's *ScientistTwo* (arXiv 2609.19644), whose code is not released. Every agent role is a separate
> `claude -p` call; critics always run in a clean context; official scores come from a read-only, hash-checked
> benchmark that the agents cannot touch. Pilot task: improve **TreeHFD** (NeurIPS 2025), one of ScientistTwo's
> own 107 benchmark problems, so the result can be compared with both ScientistTwo's and AutoSOTA's published
> outputs on the same problem. Status: **paused mid-run**. One idea (S4, label-free leave-one-cell-out shrinkage)
> passed the full benchmark and an independent re-check with unseen seeds: held-out reconstruction error better
> in 40/40 paired runs (median −7%, up to −49%), in-sample error worse in 40/40 (median +13%). Ablation,
> paper drafting, simulated review and integrity audit are not done yet. See `docs/` for design, process,
> results and the hand-off guide.

## 这是什么

2026 年 9 月，Google Cloud AI Research 发布了 ScientistTwo：给它一篇已录用的顶会论文和能跑的代码，它会自己找局限、提想法、
做实验、做消融、写论文、模拟审稿并补实验。论文公开了流程骨架和各环节的次数上限，但没有公开代码、提示词和审查标准。

这个仓库回答的问题是：**只按论文公开的方法，能不能还原出一个能用的简化版？效果和原版差多少？**

- 流程和预算照着原文第 3 节和附录 A.2 实现（每一步都是"做候选 → 审查员判定：通过 / 继续改 / 丢弃"）
- 研究性写代码（实现想法、工程改进、消融）用 Claude Code + Opus 5.5（和原版一样用 Claude 写），其余角色用 Sonnet 5.5（原版用 Gemini 3.6 Flash）；2026-09-29 起补实验、改稿、审计修稿也改用 Sonnet 以省额度
- 选了 ScientistTwo 公开题目中的 **TreeHFD**（树模型可解释性，NeurIPS 2025）：CPU 就能跑，
  而且 ScientistTwo 和 AutoSOTA 都公开了它们在这道题上的结果，可以三方对比

## 当前状态（2026-09-29 起续跑中）

| 阶段 | 状态 |
|---|---|
| 搭建只读评测（按原论文 Table 1/2 协议重建）、复现基线 | ✅ 完成，数值与原论文一致 |
| 编排器 + 24 个角色提示词，空跑全流程（52 次假调用） | ✅ 完成 |
| 找局限（5 条）→ 生成并查新 4 个种子想法 | ✅ 完成 |
| 第 1 轮：S4 通过全量审查；S2 改 3 次未达标被剪枝 | ✅ 完成 |
| 第 2 轮：进化想法 E1 | ▶ 写代码完成、子集审查判“需要改进”（留出集比 S4 好，训练集残差没解决）；09-29 起从第 1 次工程改进的断点接着做 |
| 选择 → 消融 → 写稿 → 模拟审稿/补实验 → 元审稿 → 诚信审计 | ⬜ 未开始 |
| 对 S4 的独立复核（新随机种子配对检验） | ✅ 完成 |

接手请看 **[docs/HANDOFF.md](docs/HANDOFF.md)**（环境、断点续跑、未完成任务清单、整体规划）。实时进度看 `runs/treehfd-01/RUNNER.json`（哪台机器、什么状态、最后心跳）和 `runs/treehfd-01.out`。

> 换到 Windows 续跑时发现：同一份评测脚本，合成数据和 locvar 的近邻取舍在 Mac 与 Windows 上不同（真实数据上 XGBoost 模型逐位一致，S4 的结果一致到浮点舍入级别）。所以现在按“评测环境指纹”管理结果，只在同一台机器的结果之间比较，详见 [docs/JOURNAL.md](docs/JOURNAL.md)。

## 主要结果（详见 [docs/RESULTS.md](docs/RESULTS.md)）

**S4：对稀疏格子做收缩，收缩强度用不看标签的留一估计来选。** 用流水线从没见过的随机种子复核（S4 相对原版 TreeHFD，负数 = 更好）：

| 指标 | 变化 | S4 更好的次数 |
|---|---|---|
| 合成数据：交互项 η(1,2) 误差 | −3.2%（p=0.01） | 8/10 |
| 合成数据：虚假交互项 | −18.8%（p=0.002） | 10/10 |
| 真实数据：留出集残差 | 中位数 −7.3%（Abalone −49%） | 40/40 |
| 真实数据：训练集残差 | 中位数 **+13.2%** | 0/40 |
| 真实数据：留出集正交性 | ≈0（不显著） | 19/34 |

这是一个取舍：留出集保真度和真实分量还原更好，训练集保真度更差（而原论文 Table 2 报的正是训练集指标）。

**同一道题上的三方对比**（各自相对自己的基线，设置不完全相同，只看方向）：

| | AutoSOTA | ScientistTwo 的 ECTS-HFD | 本仓库的 S4 |
|---|---|---|---|
| 合成数据 η(1,2) | −36.6%，但靠 3 倍训练数据；真正改方法的部分 −0.3% | +5%（变差） | −2.4% |
| 真实数据留出集残差 | 未评测 | 9 个数据集全部变差 | 40/40 次变好 |
| 真实数据训练集残差 | 未评测 | 9 个数据集全部大幅变好 | 变差 |
| 花费 | — | 约 3,765 美元/题（含论文） | 目前约 44 美元（还没写论文） |

## 仓库结构

```
orchestrator.py        编排器：按 ScientistTwo 的阶段调用各角色，断点续跑，用量上限自动等待
prompts/               24 个角色的提示词（_common.md 是所有角色共用的背景与红线）
bench/                 只读评测：harness.py（指标）、runner.py（子进程隔离运行被测方法）、BENCHMARK.md、data/
baseline/              原版 TreeHFD 的子集/全量评测结果
runs/treehfd-01/       本次运行的全部记录
  state.json           已完成步骤的结果（断点续跑用）
  log.jsonl            每次智能体调用和评测的日志（耗时、成本、判定）
  transcripts/         每次调用的完整提示词与回复
  ideas/<想法>/         各想法的工作区：代码、NOTES.md、官方评测结果、HISTORY.bundle（逐步修改历史）
  RUNNER.json          哪台机器在跑、状态与最后心跳
verify/                独立复核脚本与结果（新种子配对检验、官方重跑）
tools/                 引用核验、协议校验、发布前路径清理
docs/                  设计、制作过程、结果、接手指南
setup.sh               一键准备环境
```

## 快速开始

需要 conda、git、curl，以及登录好的 [Claude Code](https://claude.com/claude-code) CLI（流水线的每个角色都是一次 `claude -p` 调用，会消耗你自己的额度）。

```bash
git clone <本仓库地址> scientisttwo-lite && cd scientisttwo-lite
bash setup.sh                                             # 建环境、取 TreeHFD 源码和论文、核对评测协议（Windows 在 Git Bash 里跑）
.conda/bin/python orchestrator.py --run dry --dry-run     # 空跑：假智能体、真评测，约 20 分钟，不花额度
.conda/bin/python verify/independent_check.py             # 复现独立复核（约 25 分钟）
.conda/bin/python -u tools/autopilot.py --run treehfd-01  # 无人值守续跑（同步 GitHub、自动存档；Windows 用 .conda/python.exe）
```

续跑 `treehfd-01` 之前务必先读 [docs/HANDOFF.md](docs/HANDOFF.md)。

## 文档

- [docs/DESIGN.md](docs/DESIGN.md)：整体思路，ScientistTwo 各阶段与本实现的对照，评测与防作弊设计，与原版的差异
- [docs/JOURNAL.md](docs/JOURNAL.md)：制作过程（从调研到暂停的时间线、遇到的问题和解决办法）
- [docs/RESULTS.md](docs/RESULTS.md)：全部测试结果与局限
- [docs/HANDOFF.md](docs/HANDOFF.md)：接手指南、未完成任务、整体规划

## 声明

- `runs/` 里的代码、NOTES.md、审查意见均由 AI 智能体自动生成，未经人工同行评审；结论以 `verify/` 的独立复核为准。
- 本仓库与 Google、ScientistTwo 作者、TreeHFD 作者、AutoSOTA 作者均无关联。第三方内容与许可见 [NOTICE](NOTICE)。
- 参考文献：ScientistTwo（arXiv 2609.19644）；TreeHFD（Bénard, NeurIPS 2025, arXiv 2510.24815）；AutoSOTA（arXiv 2604.05550）。
