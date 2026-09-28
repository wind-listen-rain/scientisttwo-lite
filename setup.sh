#!/usr/bin/env bash
# 一键准备运行环境（macOS / Linux）。需要：conda（Miniconda 即可）、git、curl，以及已登录的 Claude Code CLI（`claude`）。
# 做的事：建项目内 conda 环境 .conda → 按固定提交克隆 TreeHFD 原始代码 → 装锁定版本的依赖 →
#        下载并抽取 TreeHFD 论文文本 → 还原各工作区的版本历史 → 核对评测协议 → 把协议文件设为只读
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
TREEHFD_COMMIT=5ec6cca   # ThalesGroup/treehfd，1.4.1；评测协议的哈希基于这个版本

echo "== 1/6 conda 环境（python 3.13 + llvm-openmp，xgboost 在 macOS 上需要 OpenMP）"
if [ ! -x .conda/bin/python ]; then
  conda create -q -y -p "$ROOT/.conda" -c conda-forge --override-channels python=3.13 llvm-openmp
fi

echo "== 2/6 TreeHFD 原始代码（固定提交 ${TREEHFD_COMMIT}）"
if [ ! -d tasks/treehfd/.git ]; then
  git clone -q https://github.com/ThalesGroup/treehfd.git tasks/treehfd
fi
git -C tasks/treehfd checkout -q "$TREEHFD_COMMIT"

echo "== 3/6 依赖（版本见 requirements-lock.txt）"
.conda/bin/pip install -q -r requirements-lock.txt pypdf==6.14.2
.conda/bin/pip install -q --no-deps -e tasks/treehfd

echo "== 4/6 TreeHFD 论文文本（arXiv 2510.24815v1；论文原文不随仓库分发）"
if [ ! -f tasks/treehfd_paper.txt ]; then
  curl -sL -o tasks/treehfd_paper.pdf https://arxiv.org/pdf/2510.24815v1
  .conda/bin/python -c "import pypdf; r=pypdf.PdfReader('tasks/treehfd_paper.pdf'); open('tasks/treehfd_paper.txt','w').write('\n'.join(p.extract_text() or '' for p in r.pages))"
fi

echo "== 5/6 还原工作区版本历史（HISTORY.bundle → .wsgit）"
for b in runs/*/ideas/*/HISTORY.bundle runs/*/paper_*/HISTORY.bundle; do
  [ -f "$b" ] || continue
  d="$(dirname "$b")"
  if [ ! -d "$d/.wsgit" ]; then
    git clone -q --bare "$b" "$d/.wsgit"
    git --git-dir="$d/.wsgit" config core.bare false
    printf ".wsgit/\nHISTORY.bundle\n__pycache__/\n" > "$d/.wsgit/info/exclude"
    git --git-dir="$d/.wsgit" --work-tree="$d" reset -q HEAD 2>/dev/null || true
  fi
done

echo "== 6/6 核对评测协议并设为只读"
.conda/bin/python tools/check_protocol.py runs/treehfd-01/state.json
chmod -R a-w bench tasks/treehfd/src tasks/treehfd_paper.txt

cat <<'EOF'

环境就绪。常用命令（在仓库根目录）：
  空跑整条流水线（假智能体、真评测，约 20 分钟）： .conda/bin/python orchestrator.py --run dry --dry-run
  从断点续跑 treehfd-01：                          nohup .conda/bin/python orchestrator.py --run treehfd-01 >> runs/treehfd-01.out 2>&1 &
  （macOS 上建议前面加 caffeinate -i 防止休眠；续跑前先读 docs/HANDOFF.md）
EOF
