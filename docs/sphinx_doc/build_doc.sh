#!/bin/bash

# Usage: ./build_doc.sh [--branch <branch_name>]
#!/bin/bash
# Usage:
#   ./build_doc.sh                # 构建所有语言 & 所有版本
#   ./build_doc.sh --branch xxx   # 仅构建当前分支版本（PR/feature 分支用）

set -euo pipefail

SRC_EN="source"
SRC_ZH="source_zh"
OUT_ROOT="build/html"
OUT_EN="${OUT_ROOT}/en"
OUT_ZH="${OUT_ROOT}/zh"

BRANCH_FILTER=""

if [[ "${1:-}" == "--branch" && -n "${2:-}" ]]; then
  BRANCH_FILTER="$2"
  echo "[build_doc] Only building branch: ${BRANCH_FILTER}"
fi

build_one () {
  local SRC="$1"
  local DST="$2"

  mkdir -p "${DST}"
  if [[ -n "${BRANCH_FILTER}" ]]; then
    sphinx-multiversion \
      -D smv_branch_whitelist="^(${BRANCH_FILTER})$" \
      -D smv_tag_whitelist="^$" \
      "${SRC}" "${DST}"
  else
    sphinx-multiversion "${SRC}" "${DST}"
  fi
}

echo "[build_doc] Building zh -> ${OUT_ZH}"
build_one "${SRC_ZH}" "${OUT_ZH}"

echo "[build_doc] Building en -> ${OUT_EN}"
build_one "${SRC_EN}" "${OUT_EN}"

echo "[build_doc] Done. Output at ${OUT_ROOT} (subdirs: en/, zh/)"
