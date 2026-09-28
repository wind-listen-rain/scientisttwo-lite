# 运行 treehfd-01 已暂停（2026-09-28 08:49，用户要求）

已完成并存入 state.json 的步骤：limitations、seed_ideas、idea_S4（全量通过）、idea_S2（剪枝）、evolve_1（E1 的想法）。
未完成：E1 的实现（写代码中途被停，半成品已在 ideas/E1 提交为快照 "stopped by user ..."，续跑时会提示智能体接着做），
以及之后的选择、消融、写稿、审稿、审计。

续跑命令（在 ~/scientisttwo-lite 下）：

    nohup caffeinate -i .conda/bin/python orchestrator.py --run treehfd-01 >> runs/treehfd-01.out 2>&1 &

注意：暂停期间的时间会计入 12 小时预算，续跑前先在 state.json 的 waited_s 上加上暂停时长。
