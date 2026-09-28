#!/usr/bin/env bash
# 一键准备运行环境（macOS / Linux / Windows 的 Git Bash）。需要：conda（Miniconda 即可）、git、curl，以及已登录的 Claude Code CLI（`claude`）。
# 做的事：建项目内 conda 环境 .conda → 按固定提交克隆 TreeHFD 原始代码 → 装锁定版本的依赖 →
#        下载并抽取 TreeHFD 论文文本 → 还原各工作区的版本历史 → 核对评测协议 → 把协议文件设为只读
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
TREEHFD_COMMIT=5ec6cca   # ThalesGroup/treehfd，1.4.1；评测协议的哈希基于这个版本
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) WIN=1; PY=.conda/python.exe ;; *) WIN=0; PY=.conda/bin/python ;; esac
# Windows 上 git 默认 core.autocrlf=true，会把协议文件的换行符改成 CRLF、哈希就对不上了（仓库的 .gitattributes 已强制 LF）
git config core.autocrlf false

echo "== 1/6 conda 环境（python 3.13；macOS 上 xgboost 还需要 llvm-openmp）"
if [ ! -x "$PY" ]; then
  if [ "$WIN" = 1 ]; then
    conda create -q -y -p "$(cygpath -w "$ROOT/.conda")" -c conda-forge --override-channels python=3.13
  else
    conda create -q -y -p "$ROOT/.conda" -c conda-forge --override-channels python=3.13 llvm-openmp
  fi
fi

echo "== 2/6 TreeHFD 原始代码（固定提交 ${TREEHFD_COMMIT}）"
if [ ! -d tasks/treehfd/.git ]; then
  git -c core.autocrlf=false clone -q https://github.com/ThalesGroup/treehfd.git tasks/treehfd
fi
git -C tasks/treehfd config core.autocrlf false
git -C tasks/treehfd checkout -q "$TREEHFD_COMMIT"

echo "== 3/6 依赖（版本见 requirements-lock.txt）"
"$PY" -m pip install -q -r requirements-lock.txt pypdf==6.14.2
"$PY" -m pip install -q --no-deps -e tasks/treehfd

echo "== 4/6 TreeHFD 论文文本（arXiv 2510.24815v1；论文原文不随仓库分发）"
if [ ! -f tasks/treehfd_paper.txt ]; then
  curl -sL -o tasks/treehfd_paper.pdf https://arxiv.org/pdf/2510.24815v1
  "$PY" -c "import pypdf; r=pypdf.PdfReader('tasks/treehfd_paper.pdf'); open('tasks/treehfd_paper.txt','w',encoding='utf-8',newline='\n').write('\n'.join(p.extract_text() or '' for p in r.pages))"
fi

echo "== 5/6 还原工作区版本历史（HISTORY.bundle → .wsgit）"
PYTHONUTF8=1 "$PY" -c "import sys; sys.path.insert(0, 'tools'); import gitsync; gitsync.restore_ws_histories()"

echo "== 6/6 核对评测协议并设为只读"
PYTHONUTF8=1 "$PY" tools/check_protocol.py runs/treehfd-01/state.json
if [ "$WIN" = 1 ]; then
  cmd //c "attrib +R bench\*.py & attrib +R bench\*.md & attrib +R bench\data\*.npz & attrib +R tasks\treehfd\src\treehfd\*.py & attrib +R tasks\treehfd_paper.txt" > /dev/null
else
  chmod -R a-w bench tasks/treehfd/src tasks/treehfd_paper.txt
fi

cat <<EOF

环境就绪。常用命令（在仓库根目录）：
  空跑整条流水线（假智能体、真评测，约 20 分钟）： $PY orchestrator.py --run dry --dry-run
  无人值守续跑 treehfd-01（推荐）：               $PY -u tools/autopilot.py --run treehfd-01
     （同步 GitHub → 续跑 → 自动存档推送；撞到额度上限会自己等；在 runs/treehfd-01/ 下建 STOP 文件即可让它停下）
  续跑前先读 docs/HANDOFF.md。
EOF
