#!/bin/bash
# ============================================================================
# SGLang CUDA 12.9 源码编译脚本
# 环境要求：NVIDIA Driver 550+ / CUDA Toolkit 12.9 / Python 3.12
# 前置条件：PyTorch cu129 已安装，deep_gemm 已卸载
#
# 用法：bash scripts/build_sglang_cuda129.sh
#
# 环境变量:
#   MAX_JOBS              - 并行编译数 (默认: 4)
#   NVCC_THREADS          - nvcc 线程数 (默认: 1)
#   TORCH_CUDA_ARCH_LIST  - 目标 GPU 架构 (默认: 自动检测)
#   CLEAN_BUILD           - 设为 1 清理编译缓存
#   CCACHE_DIR            - ccache 缓存目录 (默认: ~/.cache/ccache)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNEL_ROOT="$PROJECT_ROOT/sgl-kernel"
DEPS_DIR="$KERNEL_ROOT/.deps"
LOG_FILE="/tmp/sglang_build.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[BUILD]${NC} $*" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"; }

START_TIME=$(date +%s)

log "=== SGLang CUDA 12.9 源码编译 === ($(date))"

# ============================================================================
# 前置检查
# ============================================================================
if ! command -v nvcc &>/dev/null; then
    err "nvcc 未找到，请确保 CUDA Toolkit 已安装并加入 PATH"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+' | head -1)
log "CUDA 版本: $CUDA_VERSION"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    err "PyTorch CUDA 不可用，请先安装 cu129 版本的 PyTorch"
    exit 1
fi
log "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"

# ============================================================================
# GPU 架构检测
# ============================================================================
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    GPU_ARCH=$(python3 -c "import torch; cap=torch.cuda.get_device_capability(0); print(f'{cap[0]}.{cap[1]}')" 2>/dev/null || echo "")
    if [ -n "$GPU_ARCH" ]; then
        export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
        log "GPU 架构: $TORCH_CUDA_ARCH_LIST (自动检测)"
    else
        log "GPU 架构: 未检测到，使用 CMake 默认值"
    fi
else
    export TORCH_CUDA_ARCH_LIST
    log "GPU 架构: $TORCH_CUDA_ARCH_LIST (用户指定)"
fi

# 检测 GPU 架构变化
ARCH_FILE="$KERNEL_ROOT/.cuda_arch"
if [ -f "$ARCH_FILE" ]; then
    PREV_ARCH=$(cat "$ARCH_FILE")
    if [ "${TORCH_CUDA_ARCH_LIST:-}" != "$PREV_ARCH" ]; then
        warn "GPU 架构已变化: $PREV_ARCH → ${TORCH_CUDA_ARCH_LIST:-默认}"
        warn "建议使用 CLEAN_BUILD=1 重新编译"
    fi
fi
if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    echo "$TORCH_CUDA_ARCH_LIST" > "$ARCH_FILE"
fi

# ============================================================================
# ccache 配置
# ============================================================================
if command -v ccache &>/dev/null; then
    log "ccache: 已启用 ($(ccache --version | head -1))"
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    export CCACHE_DIR=${CCACHE_DIR:-"$HOME/.cache/ccache"}
    export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-100G}
    export CCACHE_SLOPPINESS="system_headers,time_macros,locale,include_file_ctime,include_file_mtime,pch_defines,time_macros_macros"
    export CCACHE_COMPRESS=1
    export CCACHE_COMPRESSLEVEL=6
    ccache -z > /dev/null 2>&1 || true
else
    warn "ccache 未安装，建议安装以加速重编译"
fi

# ============================================================================
# 编译 sgl-kernel
# ============================================================================
log "=== 编译 sgl-kernel ==="

cd "$KERNEL_ROOT"

# 清理编译缓存（仅清理构建产物，保留 .deps 源码）
if [ "${CLEAN_BUILD:-0}" = "1" ]; then
    log "清理编译缓存..."
    rm -rf build CMakeCache.txt CMakeFiles 2>/dev/null || true
    rm -rf "$DEPS_DIR"/*-subbuild "$DEPS_DIR"/*-build 2>/dev/null || true
else
    log "增量编译 (设置 CLEAN_BUILD=1 可全量重编)"
fi

# 预下载依赖
if [ ! -d "$DEPS_DIR/repo-cutlass/.git" ]; then
    log "预下载 git 依赖到 .deps/..."
    bash scripts/fetch_deps.sh 2>&1 | tee -a "$LOG_FILE"
fi

MAX_JOBS="${MAX_JOBS:-4}"
NVCC_THREADS="${NVCC_THREADS:-2}"

CMAKE_ARGS="-DSGL_KERNEL_ENABLE_SM100A=OFF -DSGL_KERNEL_ENABLE_BF16=ON -DSGL_KERNEL_ENABLE_FP8=ON -DSGL_KERNEL_COMPILE_THREADS=${NVCC_THREADS} -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DDEPS_DIR=$DEPS_DIR"

log "编译配置:"
log "  MAX_JOBS:              $MAX_JOBS"
log "  NVCC_THREADS:          $NVCC_THREADS"
log "  TORCH_CUDA_ARCH_LIST:  ${TORCH_CUDA_ARCH_LIST:-默认}"
log "  ccache:                $(command -v ccache &>/dev/null && echo "已启用 ($CCACHE_DIR)" || echo "未启用")"
log "  .deps:                 $DEPS_DIR"
log "预计耗时 20-40 分钟（首次编译），增量编译约 2-5 分钟"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
MAX_JOBS="$MAX_JOBS" \
CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS" \
CMAKE_ARGS="$CMAKE_ARGS" \
make build 2>&1 | tee -a "$LOG_FILE"

python3 -c "import sgl_kernel; print('sgl-kernel OK')" 2>&1 | tee -a "$LOG_FILE"

# ============================================================================
# 安装 SGLang 主包
# ============================================================================
log "=== 安装 SGLang 主包 ==="

cd "$PROJECT_ROOT"

if grep -q "setuptools-rust" python/pyproject.toml; then
    err "pyproject.toml 仍包含 setuptools-rust，请先注释掉 Rust 扩展"
    exit 1
fi

pip install -e "python" --no-build-isolation 2>&1 | tee -a "$LOG_FILE"

# ============================================================================
# 验证
# ============================================================================
log "=== 验证 ==="

ERRORS=0

if FLASHINFER_DISABLE_VERSION_CHECK=1 python3 -c "from sglang.srt.server_args import ServerArgs; print('SGLang OK')" 2>&1 | tee -a "$LOG_FILE"; then
    :
else
    err "SGLang 导入失败"
    ERRORS=$((ERRORS + 1))
fi

# ============================================================================
# 统计
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

if [ $ERRORS -eq 0 ]; then
    log "编译完成！耗时: ${MINUTES}分${SECONDS}秒"
    log "启动服务："
    log "  FLASHINFER_DISABLE_VERSION_CHECK=1 python -m sglang.launch_server --model-path <model> --host 0.0.0.0 --port 30000"
else
    err "有 $ERRORS 个检查失败"
fi

if command -v ccache &>/dev/null; then
    log "ccache 统计:"
    ccache -s 2>/dev/null | grep -E "Hits|Misses|Cache size" | tee -a "$LOG_FILE" || true
fi

log "日志: $LOG_FILE"