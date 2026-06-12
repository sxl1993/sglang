#!/bin/bash
# ============================================================================
# sgl-kernel 依赖预下载脚本
# 将 FetchContent 需要的 git 仓库克隆到 .deps/ 目录
# 编译时通过 -DDEPS_DIR 让 CMake 跳过 git clone
#
# 特性：并行下载、版本校验、重试机制、浅克隆
# 用法：bash scripts/fetch_deps.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPS_DIR="$KERNEL_ROOT/.deps"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[FETCH]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

# 依赖定义：名称|仓库URL|Git Tag（tag 用浅克隆，commit hash 先浅克隆再 fetch）
DEPS=(
    "repo-cutlass|https://github.com/NVIDIA/cutlass|57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
    "repo-fmt|https://github.com/fmtlib/fmt|553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
    "repo-triton|https://github.com/triton-lang/triton|v3.6.0"
    "repo-flashinfer|https://github.com/flashinfer-ai/flashinfer.git|bc29697ba20b7e6bdb728ded98f04788e16ee021"
    "repo-flash-attention|https://github.com/sgl-project/sgl-attn|bcf72ccc6816b36a5fae2c5a3c027604629785e0"
    "repo-mscclpp|https://github.com/microsoft/mscclpp.git|51eca89d20f0cfb3764ccd764338d7b22cd486a6"
    "repo-flashmla|https://github.com/sgl-project/FlashMLA|df022ebafb88578eab9f0300606ee765608d8b5c"
)

# 传递依赖：scikit-build-core 的 Python_add_library 需要，mscclpp 需要 json
TRANSITIVE_DEPS=(
    "repo-nanobind|https://github.com/wjakob/nanobind.git|v1.4.0"
    "repo-dlpack|https://github.com/dmlc/dlpack.git|v1.1"
)

# URL 依赖定义：名称|下载URL|校验标记（mscclpp 传递依赖，避免每次 cmake configure 都下载）
URL_DEPS=(
    "repo-json|https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz|v3.11.3"
)

# 带重试的 git clone
git_clone_retry() {
    local attempts=3
    local delay=5
    local i=1
    while [ $i -le $attempts ]; do
        if "$@"; then
            return 0
        fi
        warn "第 $i 次失败，${delay}s 后重试..."
        sleep $delay
        i=$((i + 1))
        delay=$((delay * 2))
    done
    err "下载失败，已重试 $attempts 次"
    return 1
}

# 版本校验：检查已有仓库是否匹配目标版本
check_version() {
    local dep_path="$1"
    local expected_tag="$2"

    [ -d "$dep_path/.git" ] || return 1

    local current_hash
    current_hash=$(git -C "$dep_path" rev-parse HEAD 2>/dev/null) || return 1
    [ -n "$current_hash" ] || return 1

    # 40 位 commit hash：直接比较
    if [ "${#expected_tag}" -eq 40 ] && echo "$expected_tag" | grep -qE '^[a-f0-9]+$'; then
        [ "$current_hash" = "$expected_tag" ] && return 0
        return 1
    fi

    # Tag/branch：先尝试 git describe，再比较 tag 指向的 commit
    local current_tag
    current_tag=$(git -C "$dep_path" describe --tags --exact-match 2>/dev/null) || true
    [ "$current_tag" = "$expected_tag" ] && return 0

    local tag_hash
    tag_hash=$(git -C "$dep_path" rev-list -n 1 "$expected_tag" 2>/dev/null) || true
    [ -n "$tag_hash" ] && [ "$current_hash" = "$tag_hash" ] && return 0

    return 1
}

mkdir -p "$DEPS_DIR"

# 并行下载（主依赖 + 传递依赖）
pids=()
for dep in "${DEPS[@]}" "${TRANSITIVE_DEPS[@]}"; do
    IFS='|' read -r name url tag <<< "$dep"

    (
        if check_version "$DEPS_DIR/$name" "$tag"; then
            log "$name: 已存在且版本正确，跳过"
            exit 0
        fi

        if [ -d "$DEPS_DIR/$name" ]; then
            log "$name: 版本不匹配，重新下载..."
            rm -rf "$DEPS_DIR/$name"
        fi

        log "$name: 下载 $url @ $tag ..."

        if [ "${#tag}" -eq 40 ] && echo "$tag" | grep -qE '^[a-f0-9]+$'; then
            # Commit hash：浅克隆后 fetch 指定 commit
            git_clone_retry git clone --depth 1 --no-single-branch "$url" "$DEPS_DIR/$name"
            git -C "$DEPS_DIR/$name" fetch origin "$tag" --depth 1
            git -C "$DEPS_DIR/$name" checkout FETCH_HEAD
        else
            # Tag：浅克隆
            git_clone_retry git clone --branch "$tag" --depth 1 "$url" "$DEPS_DIR/$name"
        fi

        # nanobind 依赖 robin_map 子模块
        if [ -f "$DEPS_DIR/$name/.gitmodules" ]; then
            git -C "$DEPS_DIR/$name" submodule update --init --recursive
        fi

        log "$name: 完成 ($(git -C "$DEPS_DIR/$name" rev-parse --short HEAD))"
    ) &
    pids+=($!)
done

# 等待所有 git 下载完成
failed=0
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        failed=$((failed + 1))
    fi
done

if [ $failed -gt 0 ]; then
    err "$failed 个依赖下载失败"
    exit 1
fi

# URL 依赖下载（tar 包，解压到 .deps/）
for dep in "${URL_DEPS[@]}"; do
    IFS='|' read -r name url tag <<< "$dep"

    if [ -d "$DEPS_DIR/$name" ]; then
        log "$name: 已存在，跳过"
        continue
    fi

    log "$name: 下载 $url ..."
    tmpdir=$(mktemp -d)
    if curl -fsSL "$url" -o "$tmpdir/archive.tar.xz"; then
        mkdir -p "$DEPS_DIR/$name"
        tar -xJf "$tmpdir/archive.tar.xz" -C "$DEPS_DIR/$name" --strip-components=1
        log "$name: 完成"
    else
        err "$name: 下载失败"
        rm -rf "$DEPS_DIR/$name"
        failed=$((failed + 1))
    fi
    rm -rf "$tmpdir"
done

if [ $failed -gt 0 ]; then
    err "$failed 个依赖下载失败"
    exit 1
fi

log "=== 所有依赖已就位 ==="
log "依赖目录: $DEPS_DIR"