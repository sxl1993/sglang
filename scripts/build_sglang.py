"""CUDA 12.9 host build. See build_sglang.md for offline prerequisites."""

import argparse
import contextlib
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "python/sglang/kernels/aot"


def dependencies():
    # Use the branch's CMake pins, not a second, independently maintained list.
    result = []
    for path in (KERNEL / "CMakeLists.txt", KERNEL / "cmake/flashmla.cmake"):
        for block in re.findall(
            r"FetchContent_Declare\((.*?)\)", path.read_text(), re.S
        ):
            match = re.fullmatch(
                r"\s*(repo-[\w-]+)\s+URL\s+(\S+)\s+URL_HASH\s+SHA256=([a-f0-9]{64})"
                r"(?:\s+SOURCE_DIR\s+\S+)?\s*",
                block,
            )
            if not match:
                raise RuntimeError(f"Unsupported dependency declaration in {path}")
            name, url, digest = match.groups()
            result.append(
                (
                    name,
                    url.replace(
                        "${GITHUB_ARTIFACTORY}",
                        os.environ.get("GITHUB_ARTIFACTORY", "github.com"),
                    ),
                    digest,
                )
            )
    return result


def check_archive(path, digest):
    if not path.is_file():
        raise RuntimeError(
            f"Missing {path}; run --prepare on a connected machine first"
        )
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise RuntimeError(
            f"SHA256 mismatch: {path}; remove this archive and run --prepare"
        )


def archive_path(cache, name, digest):
    return cache / "archives" / f"{name}-{digest}.tar.gz"


def prepare(cache, deps):
    (cache / "archives").mkdir(parents=True, exist_ok=True)
    for name, url, digest in deps:
        target = archive_path(cache, name, digest)
        if not target.exists():
            print(f"Downloading {name}: {url}", flush=True)
            with tempfile.NamedTemporaryFile(dir=target.parent) as temporary:
                with urllib.request.urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, temporary)
                temporary.flush()
                check_archive(Path(temporary.name), digest)
                shutil.copyfile(temporary.name, target)
        check_archive(target, digest)


def extract_archive(archive, target):
    # A fresh build-local source tree prevents stale/locally edited dependency reuse.
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        with tarfile.open(archive) as bundle:
            bundle.extractall(temporary, filter="data")
        entries = list(Path(temporary).iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise RuntimeError(f"Expected one top-level directory in {archive}")
        entries[0].rename(target)


def source_args(cache, deps, source_root):
    args = ["-DFETCHCONTENT_FULLY_DISCONNECTED=ON"]
    for name, _, digest in deps:
        archive = archive_path(cache, name, digest)
        check_archive(archive, digest)
        if name == "repo-flashmla-cutlass":
            target = source_root / "repo-flashmla/csrc/cutlass"
            # Git archives can include an empty submodule directory, never replace content.
            if target.is_dir():
                target.rmdir()
        else:
            target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        extract_archive(archive, target)
        args.append(f"-DFETCHCONTENT_SOURCE_DIR_{name.upper()}={target}")
    return args


@contextlib.contextmanager
def cuda12_metadata(path):
    # Match docker/Dockerfile's CUDA 12 dependency adaptation without changing
    # the default CUDA 13 project metadata in the checkout permanently.
    backup = path.with_name("pyproject.toml.cuda129-backup")
    original = path.read_bytes()
    adapted = original.replace(b"cuda-python>=13.0", b"cuda-python>=12,<13")
    adapted = adapted.replace(b"flashinfer_python[cu13]", b"flashinfer_python[cu12]")
    adapted = adapted.replace(b"nvidia-cutlass-dsl[cu13]", b"nvidia-cutlass-dsl")
    # Exclusive creation also rejects interrupted builds until their backup is recovered.
    with backup.open("xb") as stream:
        stream.write(original)
    try:
        path.write_bytes(adapted)
        yield
    finally:
        path.write_bytes(original)
        backup.unlink()


def run(command, env, cwd=ROOT):
    print(f"+ {shlex.join(map(str, command))}", flush=True)
    subprocess.run(command, env=env, cwd=cwd, check=True)


def validate_versions(nvcc_version, torch_version, torch_cuda):
    if not re.search(r"release 12\.9\b", nvcc_version):
        raise RuntimeError("CUDA_HOME/bin/nvcc must be CUDA 12.9")
    if torch_version.split("+")[0] != "2.13.0" or torch_cuda != "12.9":
        raise RuntimeError("Preinstall torch==2.13.0 with torch.version.cuda==12.9")


def preflight(env):
    if sys.platform != "linux" or sys.version_info < (3, 12):
        raise RuntimeError("Build requires Linux and Python >=3.12")
    cuda_home = Path(env.get("CUDA_HOME", "/usr/local/cuda"))
    nvcc = cuda_home / "bin/nvcc"
    version = subprocess.check_output([nvcc, "--version"], text=True)
    import torch

    validate_versions(version, torch.__version__, torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU must be visible for the post-build import checks")
    env["CUDA_HOME"] = str(cuda_home)
    env["PATH"] = str(cuda_home / "bin") + os.pathsep + env["PATH"]
    from torch.utils import cmake_prefix_path

    env["CMAKE_PREFIX_PATH"] = (
        cmake_prefix_path + os.pathsep + env.get("CMAKE_PREFIX_PATH", "")
    )
    for program in ("cmake", "ninja", "cargo", "rustc", "protoc"):
        if not shutil.which(program, path=env["PATH"]):
            raise RuntimeError(f"Preinstall build tool: {program}")
    # Select only an already installed Rust toolchain; do not let rustup download it.
    if shutil.which("rustup", path=env["PATH"]):
        toolchains = subprocess.check_output(["rustup", "toolchain", "list"], text=True)
        import tomllib

        channel = tomllib.loads((ROOT / "rust/rust-toolchain.toml").read_text())[
            "toolchain"
        ]["channel"]
        selected = next(
            (
                line.split()[0]
                for line in toolchains.splitlines()
                if line.startswith(channel + "-")
            ),
            None,
        )
        if selected is None:
            raise RuntimeError(f"Preinstall Rust toolchain {channel}")
        env["RUSTUP_TOOLCHAIN"] = selected
    for module in (
        "build",
        "scikit_build_core",
        "setuptools",
        "setuptools_rust",
        "setuptools_scm",
        "wheel",
    ):
        __import__(module)


def build(cache, deps):
    for name, _, digest in deps:
        check_archive(archive_path(cache, name, digest), digest)
    env = dict(
        os.environ,
        PIP_NO_INDEX="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        UV_OFFLINE="1",
        CARGO_NET_OFFLINE="true",
    )
    preflight(env)
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = env.get("MAX_JOBS", "24")
    threads = env.get("NVCC_THREADS", "8")
    if not threads.isdecimal() or int(threads) < 1:
        raise RuntimeError("NVCC_THREADS must be a positive integer")
    # Standard FetchContent source overrides retain upstream targets and version pins.
    # Paths are build-local, so no old source tree can silently survive a pin update.
    with tempfile.TemporaryDirectory(prefix="cuda129-", dir=cache) as temporary:
        work = Path(temporary)
        cmake_args = source_args(cache, deps, work)
        cmake_args += [
            f"-DSGL_KERNEL_COMPILE_THREADS={threads}",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        ]
        if shutil.which("ccache"):
            cmake_args += [
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]
            env.setdefault("CCACHE_BASEDIR", str(work))
        env["CMAKE_ARGS"] = shlex.join(
            shlex.split(env.get("CMAKE_ARGS", "")) + cmake_args
        )
        kernel_dist = work / "kernel-wheel"
        run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                kernel_dist,
                KERNEL,
            ],
            env,
        )
        (wheel,) = kernel_dist.glob("*.whl")
        (KERNEL / "dist").mkdir(exist_ok=True)
        shutil.copy2(wheel, KERNEL / "dist" / wheel.name)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--force-reinstall",
                wheel,
            ],
            env,
        )
        with cuda12_metadata(ROOT / "python/pyproject.toml"):
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "-e",
                    ROOT / "python",
                ],
                env,
            )
        run(
            [
                sys.executable,
                "-c",
                "import torch, sgl_kernel; "
                "assert torch.version.cuda == '12.9'; "
                "from sglang.srt.server_args import ServerArgs; print('CUDA 12.9 imports OK')",
            ],
            env,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cuda", choices=["12.9"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="download pinned CMake source archives only",
    )
    mode.add_argument(
        "--check-deps",
        action="store_true",
        help="verify archives without network or GPU",
    )
    parser.add_argument("--deps-dir", type=Path, default=KERNEL / ".deps")
    args = parser.parse_args()
    cache = args.deps_dir.resolve()
    deps = dependencies()
    if args.prepare:
        prepare(cache, deps)
    elif args.check_deps:
        for name, _, digest in deps:
            check_archive(archive_path(cache, name, digest), digest)
    else:
        build(cache, deps)
    print("Done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
