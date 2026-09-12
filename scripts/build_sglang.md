# CUDA 12.9 本机离线构建

入口：`bash scripts/build_sglang.sh 12.9`。在仓库根目录执行；`PYTHON`
可指定目标虚拟环境的 Python。脚本不安装 Python/CUDA/Rust 工具链或运行依赖，
不修改模型代码，不替换上游的 Docker/发布路径。目前只接受 CUDA 12.9。

## 1. 预备环境（联网阶段）

- Linux、Python >=3.12、CUDA Toolkit 12.9，`CUDA_HOME` 默认为 `/usr/local/cuda`。
- 已安装 `torch==2.13.0`，且 `torch.version.cuda == "12.9"`；构建时 GPU 可见。
- CMake（需支持仓库使用的 CMP0177）、Ninja、C/C++ 编译器，以及
  `build`、`scikit-build-core>=0.10`、`setuptools>=61`、`setuptools-rust>=1.11`、
  `setuptools-scm>=8`、`wheel`。可选 ccache。
- Rust 使用 `rust/rust-toolchain.toml` 的已安装工具链（当前 1.92），以及 protoc。
  提前缓存 `rust/Cargo.lock` 和独立 `rust/sglang-radix-tree/Cargo.lock` 所需依赖。
  可以在联网的目标开发机上分别执行：

  ```bash
  cargo fetch --locked --manifest-path rust/Cargo.toml
  cargo fetch --locked --manifest-path rust/sglang-radix-tree/Cargo.toml
  ```

- SGLang 的 Python 运行依赖必须提前安装为 CUDA 12 兼容组合；参考当前分支
  `docker/Dockerfile` 的 cu129 路径。包括 CUDA 12 的 FlashInfer、CUTLASS、
  DeepEP/DeepGEMM 等，不要直接在空环境中解析默认 CUDA 13 的 pyproject。
  本脚本不会下载这些 wheel，也不会自动把现有 cu13 环境转换为 cu129。

下载 AOT 构建需要的源码归档（可在另一台联网机器执行后复制缓存）：

```bash
bash scripts/build_sglang.sh 12.9 --prepare
```

源码版本和 SHA256 直接读取当前 CMake 声明，包括 FlashMLA 的嵌套 CUTLASS。
不沿用旧分支的依赖清单，也不额外加入当前构建未引用的 nanobind/dlpack。
默认保存在 `python/sglang/kernels/aot/.deps/archives`，可用 `--deps-dir PATH`
指定缓存目录；后续检查、构建需使用相同参数。镜像域名可用
`GITHUB_ARTIFACTORY` 指定，归档 SHA256 仍必须匹配。

## 2. 离线检查与构建

```bash
bash scripts/build_sglang.sh 12.9 --check-deps
CUDA_HOME=/usr/local/cuda-12.9 MAX_JOBS=24 NVCC_THREADS=8 \
  bash scripts/build_sglang.sh 12.9
```

`--check-deps` 仅检查源码归档及哈希，不需要 GPU，不表示 Python/Cargo 环境已齐全。
构建阶段使用 `PIP_NO_INDEX`、`UV_OFFLINE`、`CARGO_NET_OFFLINE`，禁用构建隔离和
Python 依赖解析；源码缺失或损坏会在编译前退出，不回退联网下载。
这是构建工具的离线模式，不是操作系统级网络沙箱，也不承诺模型首次运行时的
所有 JIT 缓存已经预热。

构建直接调用 Python 构建后端，不经过 Makefile 的 pip 安装/子模块下载前置步骤。
通过 CMake 标准 `FETCHCONTENT_SOURCE_DIR_*` 参数接入本地源码，保留上游目标、
CUDA 架构与版本判断。每次解压校验过的归档至临时目录，避免复用被修改的依赖源码；
可复用 ccache，但不复用旧 CMake 构建目录。不保证只编译当前 GPU 架构。

脚本会安装 AOT wheel，再 editable 安装主包（默认仍构建 Rust 扩展），最后检查
`sgl_kernel` 和 `ServerArgs` 导入。导入通过不等于 V4.1 模型推理通过。
AOT wheel 保存在 `python/sglang/kernels/aot/dist/`。
日志默认追加到 `/tmp/sglang_build_cu129.log`，可用 `LOG_FILE` 覆盖。

主包安装期间，临时使用与官方 Docker 相同的三项 CUDA 12 元数据替换：
`cuda-python>=12,<13`、FlashInfer `[cu12]`、CUTLASS 移除 `[cu13]`。
安装成功或 Python 异常退出均恢复原始 pyproject；不永久改变分支的默认依赖。
请勿并行修改该文件或同时运行多个构建。如果进程被 SIGKILL/机器断电，
`python/pyproject.toml.cuda129-backup` 会保留；先核对并恢复该备份，再重试。
后续普通 `pip install -e python` 会重新使用默认 CUDA 13 元数据，应继续用本入口。

## 本地回归测试

```bash
python3 -m unittest discover -s scripts/tests -p test_build_sglang.py
```

包含依赖缺失/损坏、归档路径安全、元数据恢复、shell 错误码，以及无需 CUDA 的
CMake 本地源码接入测试（需要安装 CMake）。Linux/CUDA 编译和模型运行另行验证。
