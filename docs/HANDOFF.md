# 接手指南：未完成任务与整体规划

这份文档写给接手的人：怎么把环境搭起来、怎么从断点续跑、还有哪些事没做、接下来打算怎么走。
改了任何东西，请在 docs/JOURNAL.md 末尾追加一条记录；有了新结果，更新 docs/RESULTS.md。

## 1. 准备环境

- 需要：macOS 或 Linux，conda（Miniconda 即可）、git、curl，以及登录好的 Claude Code CLI（`claude --version` 能跑）。
  流水线每个角色都是一次 `claude -p` 调用，花的是你自己账号的额度。
- 运行 `bash setup.sh`。它会：建 `.conda` 环境 → 按固定提交克隆 TreeHFD → 装锁定版本的依赖 → 下载并抽取论文文本 →
  从 `HISTORY.bundle` 还原各工作区的版本历史 → **核对评测协议** → 把协议文件设为只读。
- 最后一步如果提示"评测协议不一致"，说明你的数据、评测脚本或 TreeHFD 源码和运行开始时不同，这时不要续跑，先查
  `python tools/check_protocol.py` 列出的文件。
- 建议先空跑一遍，确认环境没问题：`.conda/bin/python orchestrator.py --run dry --dry-run`（约 20 分钟，不花额度）。

## 2. 从断点续跑 treehfd-01

```bash
# 1) 把暂停的时长补进 waited_s，不然会占用 12 小时的预算（编排器只统计真正在工作的时间）
.conda/bin/python - <<'PY'
import json, time
p = "runs/treehfd-01/state.json"; s = json.load(open(p))
paused_since = time.mktime(time.strptime("2026-09-28 08:49:00", "%Y-%m-%d %H:%M:%S"))  # STOPPED.md 里的暂停时间
s["waited_s"] = s.get("waited_s", 0) + (time.time() - paused_since)
json.dump(s, open(p, "w"), indent=1, ensure_ascii=False)
PY
# 2) 续跑（macOS 上加 caffeinate -i 防止休眠；需要插电，拔电后系统降频会让评测慢好几倍）
nohup caffeinate -i .conda/bin/python orchestrator.py --run treehfd-01 >> runs/treehfd-01.out 2>&1 &
tail -f runs/treehfd-01.out
```

**断点续跑是怎么实现的**：`state.json` 的 `steps` 里存着每个已完成步骤的返回值；再跑同一条命令时，已完成的步骤直接取结果，
不会再调用智能体。评测结果文件已存在的也不会重跑。正在进行中的步骤会从头再做。E1 写到一半的代码已经提交为快照，
编排器看到最后一次提交写着 "stopped" 会提示智能体接着做。

**state.json 的结构**：
- `steps.limitations`、`steps.seed_ideas`：局限与种子想法
- `steps.idea_<编号>`：每个想法的轨迹（每次审查的判定和反馈、官方结果文件路径、最终 good/bad）
- `steps.evolve_<轮>`：进化出的想法
- `manifest`：运行开始时评测协议文件的 SHA-256；`protocol_changes`：之后有记录的改动（目前只有一次，发布前改了 BENCHMARK.md 里的路径写法）
- `calls`、`cost_usd`、`waited_s`、`integrity`（完整性违规记录，目前为空）
- 文件里的 `<ROOT>` 在读取时会自动换成你本机的仓库路径

**用量上限**：碰到"You've hit your session limit · resets …"时，编排器会自动等到重置时间再重试，并在日志里写一条 `wait`，
不需要人来管。

## 3. 未完成的任务

按执行顺序排列。前 6 项由编排器自动完成，续跑就会依次做完。

- [ ] **E1 实现与审查**：子集 → 工程改进（最多 2 次）→ 全量 → 审查。半成品在 `runs/treehfd-01/ideas/E1/`。
  它要解决的问题是 S4 训练集残差变差；目前的子集结果表明这个问题还没解决。
- [ ] 如果 E1 失败：第 2 轮还有一个名额，给下一个种子想法 S1（`max_rounds=2`）。
- [ ] **选择最优想法**：只有 S4 一个成功时直接选 S4。
- [ ] **消融**：规划 2 到 4 个变体 → 实现 → 并行做全量评测 → 消融审查 → 必要时改出新版本，由比较智能体决定是否替换（最多 1 次）。
  特别要拆开两件事：S4 的留出集增益有多少来自收缩，有多少来自把原版的随机回退（局限 L1）换成确定性规则。如果消融规划智能体没有想到这一点，续跑前可以在 `prompts/ablation_planner.md` 里点明。
- [ ] **写稿 → 模拟审稿/补实验（最多 2 轮，分数 ≥8 就停）→ 元审稿（必要时回头改方法 1 次）**。
- [ ] **诚信审计**：重跑官方评测，核对论文里的每个数字、协议合规、引用真实性、方法描述与代码是否一致；有问题时修 1 次。
- [ ] 人工复核：跑完后用 `verify/independent_check.py` 的办法，换新种子验证最终方法（把脚本里的 `METHODS` 指向最终的工作区）。
- [ ] 更新 docs/RESULTS.md，并补上和 ScientistTwo 在论文层面的对比。可以用 Stanford Agentic Reviewer（paperreview.ai）
  做独立评分，但它需要上传稿件到第三方网站，上传前要征得项目负责人同意。

## 4. 整体规划

**阶段 A：跑完 treehfd-01（当前）**
完成上面的清单，得到第一份"论文 + 审计报告"，和 ScientistTwo 放出的 ECTS-HFD 论文做正面比较。
为了省额度，可以考虑把写代码的角色从 Opus 换成 Sonnet（改 `orchestrator.py` 里的 `STRONG`），
或者把 `n_peer` 从 2 改成 1。这两项都会偏离原版设定，改了要在 JOURNAL 里记下来。

**阶段 B：补上和原版差距最大的两处**
1. 交叉审查：让审查角色换成另一家的模型，比如 Gemini（原版是 Gemini 审、Claude 写），避免同一家族模型的共同盲点。
   需要给 `call_claude` 加一个其他模型的调用分支。
2. 多划分官方评测：真实数据的官方评测从 1 种划分改成 3 到 5 种，让审查员看到方差。**这会改评测协议**，只能在新的运行里做，
   不要改 treehfd-01 正在用的 bench/。

**阶段 C：扩到更多题目**
在 ScientistTwo 的 107 道题里再挑 2 到 3 道 CPU 能跑的，候选有 M3SVM（带差分隐私的多分类 SVM）、OLLA（带约束采样）、
Balanced Active Inference。每道题都要按原论文重建只读评测（参照 bench/ 的结构），并且要求 ScientistTwo 和 AutoSOTA
都有公开结果，这样才能对比。

**阶段 D：总结**
汇总几道题的结果：简化版和原版在方法层面、论文层面的差距，成本对比，以及哪些环节最值钱、哪些可以砍。

## 5. 常见修改

- **预算**：`orchestrator.py` 顶部的 `CFG`，括号里是原版的数值。
- **模型**：`FAST`（非代码角色）、`STRONG`（写代码、工程改进、补实验、改稿）。
- **提示词**：`prompts/<角色>.md`，所有角色共用的背景与红线在 `prompts/_common.md`。占位符有 `{{ROOT}}`、`{{PY}}`、`{{WORKDIR}}`，以及每个角色自己的变量。
- **开新的运行**：换一个 `--run` 名字即可。评测协议会在第一次运行时锁定。

## 6. 已知问题与注意事项

- 不要在运行中改 `bench/` 或 `tasks/treehfd/src/`：编排器会检测到并中止整次运行。要改就开新的运行。
- 发布到 GitHub 之前，运行 `python3 tools/sanitize_paths.py`；如果工作区历史里也有本机路径，再对每个工作区运行
  `python3 tools/sanitize_ws_history.py runs/<运行>/ideas/<想法>`，它会重新生成 `HISTORY.bundle`。
- 工作区的版本历史在 `<工作区>/.wsgit`（不进外层仓库），查看方法：`git --git-dir=<工作区>/.wsgit --work-tree=<工作区> log`。
- **原版 TreeHFD 的测试点预测带随机性**：`tasks/treehfd/src/treehfd/cartesian_partition.py:215` 在遇到训练时没见过的交互格子时，用不设种子的 `np.random.default_rng().choice` 随机挑一个邻居格子的值。所以基线在留出集和合成数据测试点上的指标每次重跑会有波动（实测：留出集正交性最多 ±5%，留出集残差 ±1–2%；训练集指标完全不变）。这就是流水线找出的局限 L1。S4 换成了确定性规则，重跑逐位一致。比较留出集指标时要把这个噪声考虑进去。
- Superconductivity 用基线跑一次要 12 分钟，是全量评测里最慢的一环。
