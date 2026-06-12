#!/bin/bash
# ============================================================================
# SGLang 源码编译脚本
# 环境要求：CUDA Toolkit / Python 3.12
# 前置条件：对应版本的 PyTorch 已安装，deep_gemm 已卸载
#
# 用法：bash scripts/build_sglang.sh <cuda_version>
#   bash scripts/build_sglang.sh 12.8
#   bash scripts/build_sglang.sh 12.9
#   BUILD_WHL=1 CLEAN_BUILD=1 bash scripts/build_sglang.sh 12.9
#   PYPROJECT_VARIANT=cu129 BUILD_WHL=1 bash scripts/build_sglang.sh 12.9   # 用 cu12.9 依赖变体编译 whl
#
# 环境变量:
#   MAX_JOBS              - 并行编译数 (默认: 24)
#   NVCC_THREADS          - nvcc 线程数 (默认: 8)
#   TORCH_CUDA_ARCH_LIST  - 目标 GPU 架构 (默认: 自动检测)
#   BUILD_WHL             - 设为 1 编译 sglang whl 包并安装（默认 editable 安装）
#   CLEAN_BUILD           - 设为 1 清理编译缓存
#   PYPROJECT_VARIANT     - 设为 cu129 时用 python/pyproject.cu129.toml 临时覆盖主 pyproject（build 后自动还原）
#   SGL_KERNEL_ENABLE_SM100A - SM100A 支持 (默认: OFF)
#   CCACHE_DIR            - ccache 缓存目录 (默认: ~/.cache/ccache)
# ============================================================================

set -euo pipefail

CUDA_VER="${1:?用法: bash scripts/build_sglang.sh <cuda_version> (例如 12.8, 12.9)}"
CUDA_VER_SHORT="${CUDA_VER//./}"  # 12.8 → 128

# 按 CUDA 版本配置
case "$CUDA_VER" in
    12.8)
        INSTALL_TORCH_CMD="pip install --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128 torch==2.11.0+cu128 torchaudio==2.11.0+cu128 torchvision==0.26.0+cu128"
        ;;
    12.9)
        INSTALL_TORCH_CMD="pip install --index-url https://mirrors.aliyun.com/pytorch-wheels/cu129 torch==2.11.0+cu129 torchaudio==2.11.0+cu129 torchvision==0.26.0+cu129"
        ;;
    *)
        echo "不支持的 CUDA 版本: $CUDA_VER (支持: 12.8, 12.9)" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNEL_ROOT="$PROJECT_ROOT/sgl-kernel"
DEPS_DIR="$KERNEL_ROOT/.deps"
LOG_FILE="/tmp/sglang_build_cu${CUDA_VER_SHORT}.log"
> "$LOG_FILE"

# PYPROJECT_VARIANT=cu129 时用 python/pyproject.cu129.toml 临时覆盖主 pyproject.toml，
# build 结束（含失败/中断）由 trap 还原，避免污染工作区
PYPROJECT_VARIANT="${PYPROJECT_VARIANT:-}"
PYPROJECT_MAIN="$PROJECT_ROOT/python/pyproject.toml"
PYPROJECT_BACKUP=""
restore_pyproject() {
    if [ -n "$PYPROJECT_BACKUP" ] && [ -f "$PYPROJECT_BACKUP" ]; then
        mv -f "$PYPROJECT_BACKUP" "$PYPROJECT_MAIN"
        PYPROJECT_BACKUP=""
    fi
}
swap_pyproject() {
    [ -z "$PYPROJECT_VARIANT" ] && return 0
    local variant="$PROJECT_ROOT/python/pyproject.${PYPROJECT_VARIANT}.toml"
    if [ ! -f "$variant" ]; then
        err "PYPROJECT_VARIANT=$PYPROJECT_VARIANT 但未找到 $variant"
        exit 1
    fi
    PYPROJECT_BACKUP="${PYPROJECT_MAIN}.bak.$$"
    cp -f "$PYPROJECT_MAIN" "$PYPROJECT_BACKUP"
    trap restore_pyproject EXIT INT TERM
    cp -f "$variant" "$PYPROJECT_MAIN"
    log "已切换 pyproject 为 $PYPROJECT_VARIANT 变体 (build 结束自动还原)"
}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[BUILD]${NC} $*"; echo "[BUILD] $*" >> "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; echo "[WARN] $*" >> "$LOG_FILE"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; echo "[ERROR] $*" >> "$LOG_FILE"; }

# 终端保留颜色，写入日志文件前剥离 ANSI 转义码
tee_clean() { tee >(sed -u 's/\x1b\[[0-9;]*[a-zA-Z]//g' >> "$LOG_FILE"); }

ERRORS=0
START_TIME=$(date +%s)

log "=== SGLang CUDA ${CUDA_VER} 源码编译 === ($(date))"

# ============================================================================
# 前置检查
# ============================================================================
if ! command -v nvcc &>/dev/null; then
    err "nvcc 未找到，请确保 CUDA Toolkit 已安装并加入 PATH"
    exit 1
fi
NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+' | head -1)

TORCH_INFO=$(python3 -c "
import torch
cuda_ver = torch.version.cuda or ''
version = torch.__version__
avail = torch.cuda.is_available()
arch = ''
if avail and torch.cuda.device_count() > 0:
    cap = torch.cuda.get_device_capability(0)
    arch = f'{cap[0]}.{cap[1]}'
print(f'{cuda_ver}|{version}|{avail}|{arch}')
" 2>/dev/null || echo "||False|")
TORCH_CUDA_VER=$(echo "$TORCH_INFO" | cut -d'|' -f1)
TORCH_VERSION=$(echo "$TORCH_INFO" | cut -d'|' -f2)
TORCH_CUDA_AVAIL=$(echo "$TORCH_INFO" | cut -d'|' -f3)
GPU_ARCH_DETECTED=$(echo "$TORCH_INFO" | cut -d'|' -f4)

log "CUDA Toolkit (nvcc): $NVCC_VERSION | PyTorch: $TORCH_VERSION | PyTorch CUDA runtime: ${TORCH_CUDA_VER:-未知}"

if [ "$TORCH_CUDA_AVAIL" != "True" ]; then
    err "PyTorch CUDA 不可用，请先安装 cu${CUDA_VER_SHORT} 版本的 PyTorch"
    exit 1
fi
TORCH_CUDA_VER_MAJOR_MINOR=$(echo "$TORCH_CUDA_VER" | grep -oP '^\d+\.\d+')
if [ "$TORCH_CUDA_VER_MAJOR_MINOR" != "$CUDA_VER" ]; then
    err "当前 torch.version.cuda=$TORCH_CUDA_VER, 编译需要 $CUDA_VER"
    err "sgl-kernel 编译会链接当前环境 torch 携带的 CUDA runtime，版本不匹配将导致运行时 NCCL 崩溃"
    err "请先安装："
    err "  $INSTALL_TORCH_CMD"
    exit 1
fi

# ============================================================================
# GPU 架构检测
# ============================================================================
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    if [ -n "$GPU_ARCH_DETECTED" ]; then
        export TORCH_CUDA_ARCH_LIST="$GPU_ARCH_DETECTED"
        log "GPU 架构: $TORCH_CUDA_ARCH_LIST (自动检测)"
    else
        log "GPU 架构: 未检测到，使用 CMake 默认值"
    fi
else
    export TORCH_CUDA_ARCH_LIST
    log "GPU 架构: $TORCH_CUDA_ARCH_LIST (用户指定)"
fi

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
HAVE_CCACHE=false
if command -v ccache &>/dev/null; then
    HAVE_CCACHE=true
    log "ccache: 已启用 ($(ccache --version | head -1))"
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    export CCACHE_DIR=${CCACHE_DIR:-"$HOME/.cache/ccache"}
    export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-100G}
    export CCACHE_SLOPPINESS="system_headers,time_macros,locale,include_file_ctime,include_file_mtime,pch_defines"
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

if [ "${CLEAN_BUILD:-0}" = "1" ]; then
    log "清理编译缓存..."
    rm -rf build CMakeCache.txt CMakeFiles 2>/dev/null || true
    rm -rf "$DEPS_DIR"/*-subbuild "$DEPS_DIR"/*-build 2>/dev/null || true
else
    log "增量编译 (设置 CLEAN_BUILD=1 可全量重编)"
fi

log "校验/下载 git 依赖..."
bash scripts/fetch_deps.sh 2>&1 | tee_clean

MAX_JOBS="${MAX_JOBS:-24}"
NVCC_THREADS="${NVCC_THREADS:-8}"

CMAKE_ARGS="-DSGL_KERNEL_ENABLE_SM100A=${SGL_KERNEL_ENABLE_SM100A:-OFF} -DSGL_KERNEL_ENABLE_BF16=ON -DSGL_KERNEL_ENABLE_FP8=ON -DSGL_KERNEL_COMPILE_THREADS=${NVCC_THREADS} -DDEPS_DIR=$DEPS_DIR"
if [ "$HAVE_CCACHE" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
fi

log "编译配置:"
log "  MAX_JOBS:              $MAX_JOBS"
log "  NVCC_THREADS:          $NVCC_THREADS"
log "  TORCH_CUDA_ARCH_LIST:  ${TORCH_CUDA_ARCH_LIST:-默认}"
log "  ccache:                $([ "$HAVE_CCACHE" = true ] && echo "已启用 ($CCACHE_DIR)" || echo "未启用")"
log "  .deps:                 $DEPS_DIR"
log "预计耗时 10-20 分钟（首次编译），增量编译约 1-3 分钟"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="$MAX_JOBS"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export CMAKE_ARGS="$CMAKE_ARGS"

if ! make build 2>&1 | tee_clean; then
    err "sgl-kernel 编译失败，详见日志: $LOG_FILE"
    exit 1
fi

export FLASHINFER_DISABLE_VERSION_CHECK=1

python3 -c "import sgl_kernel; print(f'sgl-kernel OK: {sgl_kernel.__version__}')" 2>&1 | tee_clean

# ============================================================================
# 安装 SGLang 主包
# ============================================================================
log "=== 安装 SGLang 主包 ==="

cd "$PROJECT_ROOT"

swap_pyproject

# 编译 sgl-kernel 可能拉入不兼容的 torch，再次校验
TORCH_CUDA_VER_AFTER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
TORCH_CUDA_VER_AFTER_MM=$(echo "$TORCH_CUDA_VER_AFTER" | grep -oP '^\d+\.\d+')
if [ "$TORCH_CUDA_VER_AFTER_MM" != "$CUDA_VER" ]; then
    err "torch CUDA 版本在编译后变为 $TORCH_CUDA_VER_AFTER（期望 $CUDA_VER）"
    err "sgl-kernel 编译可能拉入了不兼容的依赖，请重新安装："
    err "  $INSTALL_TORCH_CMD"
    ERRORS=$((ERRORS + 1))
fi

# ============================================================================
# Rust 扩展 (gRPC server) 工具链检查
# rust-toolchain.toml 可能 pin 了离线不可达的 channel (如 1.90)，
# 强制 setuptools-rust 调 cargo 时走本地已装的 stable，避免 rustup 联网超时
# ============================================================================
if grep -v '^\s*#' python/pyproject.toml | grep -q "setuptools-rust"; then
    if ! command -v cargo &>/dev/null; then
        err "pyproject.toml 启用了 setuptools-rust (gRPC server)，但未找到 cargo"
        err "请安装 Rust 工具链，或注释掉 python/pyproject.toml 中的 Rust 扩展"
        exit 1
    fi
    if ! command -v protoc &>/dev/null; then
        err "Rust gRPC 扩展需要 protoc (tonic-build 依赖)，未找到 protoc，请先安装"
        exit 1
    fi
    # build --no-isolation 不会自动装 build 依赖，缺 setuptools-rust 会在打包末尾才报错
    if ! python3 -c "import setuptools_rust" 2>/dev/null; then
        warn "未找到 setuptools-rust (build --no-isolation 不会自动安装)，尝试安装..."
        pip install "setuptools-rust>=1.10" 2>&1 | tee_clean
        python3 -c "import setuptools_rust" 2>/dev/null || { err "setuptools-rust 安装失败，请手动安装后重试"; exit 1; }
    fi
    export RUSTUP_TOOLCHAIN="${RUST_TOOLCHAIN:-stable}"
    RUST_VER=$(cargo --version 2>/dev/null || echo "未知")
    log "Rust 扩展 (gRPC server) 已启用: $RUST_VER (RUSTUP_TOOLCHAIN=$RUSTUP_TOOLCHAIN)"
fi

if [ "${BUILD_WHL:-0}" = "1" ]; then
    log "编译 sglang whl 包 (BUILD_WHL=1)"
    python3 -c "import setuptools_scm" 2>/dev/null || pip install setuptools-scm 2>&1 | tee_clean
    rm -rf python/dist/sglang-*.whl python/build 2>/dev/null || true
    (cd python && python3 -m build --wheel --no-isolation 2>&1 | tee_clean)
    SGLANG_WHL=$(ls python/dist/sglang-*.whl 2>/dev/null | head -1)
    if [ -z "$SGLANG_WHL" ]; then
        err "sglang whl 包未生成"
        ERRORS=$((ERRORS + 1))
    else
        pip install "$SGLANG_WHL" --force-reinstall --no-deps 2>&1 | tee_clean
    fi
else
    log "editable 安装 sglang (默认; 设置 BUILD_WHL=1 编译 whl 包)"
    pip install -e "python" --no-build-isolation --no-deps 2>&1 | tee_clean
fi

# ============================================================================
# 验证
# ============================================================================
log "=== 验证 ==="

if FLASHINFER_DISABLE_VERSION_CHECK=1 python3 -c "from sglang.srt.server_args import ServerArgs; print('SGLang OK')" 2>&1 | tee_clean; then
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
SECS=$((ELAPSED % 60))

if [ $ERRORS -eq 0 ]; then
    log "编译完成！耗时: ${MINUTES}分${SECS}秒"
    log "启动服务："
    log "  sglang serve --model-path <model> --host 0.0.0.0 --port 30000"
else
    err "有 $ERRORS 个检查失败"
fi

if [ "$HAVE_CCACHE" = true ]; then
    log "ccache 统计:"
    ccache -s 2>/dev/null | grep -E "Hits|Misses|Cache size" | tee_clean || true
fi

log "日志: $LOG_FILE"