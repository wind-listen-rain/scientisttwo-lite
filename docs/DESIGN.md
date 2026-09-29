# 整体思路与框架设计

## 1. 从哪来

这个项目起源于一个问题：GitHub 上那些"AI 科研工作台"（open-science 等）比直接让 Claude 多轮研究强在哪？调研发现，它们大多只是
轻量外壳，第三方基准上同模型的提升不显著；而 Google 的 ScientistTwo 是另一类——全自动、流程很重、每题几千美元。
它没开源，于是问题变成：**只照着论文写的方法，能还原到什么程度？**

## 2. ScientistTwo 公开了什么、没公开什么

| 公开 | 未公开 |
|---|---|
| 阶段结构与顺序（论文 Table 1、第 3 节） | 全部提示词 |
| 通用循环：候选 → 审查员判定 accept / refine / reject，改到上限就丢 | 每个审查员的判定标准（什么算"明显更差"、新颖度怎么打分） |
| 各环节次数上限（附录 A.2） | 子集怎么选、硬件与每题算力上限 |
| 模型分工：写代码用 Claude Code + Opus 4.8，其余 Gemini 3.6 Flash | 智能体之间怎么交接状态、长任务怎么管上下文 |
| 107 道题的清单（来自 AutoSOTA 的任务集） | 实际使用的 ScholarPeer 版本 |
| 四项诚信审计的内容 | 审计的实现代码 |

审查员的判定标准虽然没公开，但附录 B 的判例可以反推：提升来自 EMA、label smoothing 这类通用技巧的一律否决；把花哨组件剥掉，
只留真正带来提升的那一个；改评测环节的直接判不合格。这些都写进了本实现的审查员提示词。

## 3. 阶段对照

| ScientistTwo（论文章节） | 本实现的角色（prompts/） | 模型 | 次数上限：原版 → 本实现 |
|---|---|---|---|
| 3.1 找局限 + 验证是否足够 | limitation_extractor, limitation_verifier | Sonnet | 16 轮 → 3 轮 |
| 3.1 生成种子想法 + 查新（Google 搜 2 篇相关论文） | idea_generator（Opus）, novelty_checker（WebSearch） | Opus / Sonnet | 未公开 → 4 个 |
| 3.2 子集实现 → 子集审查 → 工程改进 | coder, subset_critic, engineer | Opus / Sonnet / Opus | 工程改进 2 次 → 2 次 |
| 3.2 全量验证 → 全量审查 → 工程改进 | fullset_critic, engineer | Sonnet / Opus | 2 次 → 2 次 |
| 3.3 想法进化（读全部实验记录） | idea_evolver | Opus | 4 轮、攒够 4 个成功就停 → 2 轮、2 个 |
| 3.3 选最优 | selector | Sonnet | — |
| 3.4 消融规划 → 实现 → 审查 → 必要时改方法 → 结果比较 | ablation_planner, ablation_coder, ablation_critic, engineer, result_comparator | 混合 | 1 次 → 1 次 |
| 3.5 起草论文（原版用 PaperOrchestra） | drafter | Sonnet | — |
| 3.5 模拟审稿 → 补实验 → 改稿（原版用 ScholarPeer） | reviewer, rebuttal_planner, rebuttal_coder, enhancer | Sonnet（补实验、改稿 2026-09-29 前用 Opus） | 2 轮、分数 ≥8 停 → 相同 |
| 3.6 元审稿 → 必要时回头改方法 | meta_reviewer, engineer, result_comparator | 混合 | 1 次 → 1 次 |
| 4.2 CoE 诚信审计（重跑核分、协议合规、引用核验、方法-代码一致） | claim_auditor, spec_auditor, tools/verify_refs.py, method_code_auditor, audit_fixer | Sonnet（审计修稿 2026-09-29 前用 Opus） | — |

每个角色都是一次独立的 `claude -p` 调用。审查类角色只给只读工具（Read/Glob/Grep），总在干净上下文里运行，看不到写代码
那个智能体的对话，只能看代码文件和官方评测表（"追溯而不重算"）。

## 4. 评测设计：这是整件事可信的前提

ScientistTwo 批评 AutoSOTA 的核心是"单指标优化会去改评测"。所以评测必须是智能体碰不到的：

- **协议按原论文重建**：合成数据按 Table 1（p=6、ρ=0.5、n=5000、XGBoost 100 棵树、10 次重复，有解析的真实分量）；
  真实数据按 Table 2 的 9 个 UCI 数据集，三项指标的定义照附录 B.3.1（残差、层级正交性、局部稳定性）。
  另外加了 80/20 留出集上的残差和正交性，因为只看训练集会奖励过拟合（ScientistTwo 的 ECTS-HFD 正栽在这里）。
- **子集 / 全量两档**：子集是合成数据 3 次重复 + 3 个小数据集（约 1 分钟），全量是 10 次重复 + 9 个数据集（约 20 分钟），
  对应原版的"先在子集上筛，通过的再上全量"。
- **方法接口很窄**：被测方法只拿到训练好的模型和它自己的训练输入，在独立子进程里运行（bench/runner.py），
  永远拿不到标签和真实分量；所有指标由评测进程自己算。
- **只读 + 哈希校验**：bench/ 与 TreeHFD 源码设为只读；每次智能体调用后，编排器重新计算这些文件的 SHA-256，
  和运行开始时的清单比对，任何改动都会让整次运行中止。
- **官方分数只认编排器跑的**：智能体可以自测，但审查员看到的表格一律来自编排器调用 harness 的结果。
- **红线写进每个角色的提示词**（prompts/_common.md）：不许重训/换种子重训模型、不许用额外数据、不许针对数据集特判、
  拟合时间超过基线 5 倍要说明理由、超过 20 倍不接受。AutoSOTA 在这道题上"3 个种子各抽一份训练数据再平均"的做法就违反第 3 条。

## 5. 与原版的差异（解读结果时要记住）

1. **模型**：非代码角色用 Sonnet 而不是 Gemini（2026-09-29 起固定为 claude-sonnet-5-5 / claude-opus-5-5；补实验、改稿、审计修稿三个角色从 Opus 改为 Sonnet 以省额度，见 JOURNAL）；写代码和审查的是同一家族模型，可能有共同盲点（原版是 Gemini 审、Claude 写）。
2. **预算**：找局限、实验轮数、成功数都砍到原版的零头（见上表），每题可探索的想法少得多。
3. **评测由人预先搭好**：原版由 Baseline Coding Agent 自己复现基线；这里为了防作弊，评测和基线由人搭好并锁定。
4. **写稿与审稿工具**：没有用 PaperOrchestra 和 ScholarPeer（后者未开源），改为 Claude 按 ICLR 标准写 Markdown 论文、
   在干净上下文里审稿；本机没有 TeX。
5. **硬件**：Apple M4 笔记本 CPU；每题都挑 CPU 能跑的。
6. **订阅额度**：原版按 API 付费不受限；这里走 Claude 订阅，会话上限每 1.5-2 小时触发一次，编排器会自动等到重置。
