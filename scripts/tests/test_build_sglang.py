import importlib.util
import hashlib
import io
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "build_sglang.py"


class OfflineBuildTests(unittest.TestCase):
    def load_builder(self):
        self.assertTrue(SCRIPT.is_file(), "offline build entry point is missing")
        spec = importlib.util.spec_from_file_location("build_sglang", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_missing_archive_fails_without_download(self):
        builder = self.load_builder()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "prepare"):
                builder.check_archive(Path(directory) / "missing.tar.gz", "0" * 64)

    def test_prepare_verifies_download_and_reuses_cached_archive(self):
        builder = self.load_builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tar.gz"
            source.write_bytes(b"downloaded archive")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            cache = root / "cache"
            builder.prepare(cache, [("repo-test", source.as_uri(), digest)])
            source.unlink()
            # The original URL now fails, so success requires verified cache reuse.
            builder.prepare(cache, [("repo-test", source.as_uri(), digest)])
            self.assertEqual(
                builder.archive_path(cache, "repo-test", digest).read_bytes(),
                b"downloaded archive",
            )

    def test_rejects_mismatched_toolkit_or_torch(self):
        builder = self.load_builder()
        for nvcc, torch, cuda in (
            ("release 13.0", "2.13.0+cu129", "12.9"),
            ("release 12.9", "2.11.0+cu129", "12.9"),
            ("release 12.9", "2.13.0+cu130", "13.0"),
        ):
            with self.subTest(nvcc=nvcc, torch=torch, cuda=cuda):
                with self.assertRaises(RuntimeError):
                    builder.validate_versions(nvcc, torch, cuda)
        builder.validate_versions(
            "Cuda compilation tools, release 12.9, V12.9.86", "2.13.0+cu129", "12.9"
        )

    def test_corrupt_archive_is_rejected(self):
        builder = self.load_builder()
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.tar.gz"
            archive.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                builder.check_archive(archive, "0" * 64)

    def test_cuda_metadata_restored_when_build_fails(self):
        builder = self.load_builder()
        original = (
            '"cuda-python>=13.0",\n'
            '"flashinfer_python[cu13]==0.6.18",\n'
            '"nvidia-cutlass-dsl[cu13]==4.6.2",\n'
            '"torch==2.13.0",\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "pyproject.toml"
            project.write_text(original)
            with self.assertRaisesRegex(ValueError, "build failed"):
                with builder.cuda12_metadata(project):
                    self.assertIn('"cuda-python>=12,<13"', project.read_text())
                    self.assertIn(
                        '"flashinfer_python[cu12]==0.6.18"', project.read_text()
                    )
                    self.assertIn('"torch==2.13.0"', project.read_text())
                    raise ValueError("build failed")
            self.assertEqual(project.read_text(), original)

    def test_pins_cover_nested_flashmla_cutlass(self):
        builder = self.load_builder()
        deps = builder.dependencies()
        self.assertEqual(len(deps), 7)
        nested = next(dep for dep in deps if dep[0] == "repo-flashmla-cutlass")
        self.assertIn("147f5673d0c1c3dcf66f78d677fd647e4a020219", nested[1])
        self.assertEqual(len(nested[2]), 64)

    def test_source_overrides_resolve_offline_in_cmake(self):
        if not shutil.which("cmake"):
            self.skipTest("cmake is not installed")
        builder = self.load_builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deps = []
            for name, filename in (
                ("repo-flashmla", "csrc/marker"),
                ("repo-flashmla-cutlass", "include/marker"),
            ):
                data = io.BytesIO()
                with tarfile.open(fileobj=data, mode="w:gz") as bundle:
                    entry = tarfile.TarInfo("upstream/" + filename)
                    entry.size = 2
                    bundle.addfile(entry, io.BytesIO(b"ok"))
                digest = hashlib.sha256(data.getvalue()).hexdigest()
                archive = builder.archive_path(root, name, digest)
                archive.parent.mkdir(exist_ok=True)
                archive.write_bytes(data.getvalue())
                deps.append((name, "https://invalid.example/no-network", digest))
            sources = root / "sources"
            sources.mkdir()
            args = builder.source_args(root, deps, sources)
            nested = sources / "repo-flashmla/csrc/cutlass/include/marker"
            self.assertEqual(nested.read_bytes(), b"ok")
            project = root / "project"
            project.mkdir()
            (project / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.26)
project(offline NONE)
include(FetchContent)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
FetchContent_Declare(repo-flashmla URL https://invalid.example/no-network)
FetchContent_Populate(repo-flashmla)
FetchContent_Declare(repo-flashmla-cutlass
  URL https://invalid.example/no-network
  SOURCE_DIR ${repo-flashmla_SOURCE_DIR}/csrc/cutlass)
FetchContent_Populate(repo-flashmla-cutlass)
if(NOT EXISTS "${repo-flashmla-cutlass_SOURCE_DIR}/include/marker")
  message(FATAL_ERROR "local FlashMLA CUTLASS source was not selected")
endif()
""")
            result = subprocess.run(
                ["cmake", "-S", str(project), "-B", str(root / "build"), *args],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shell_reports_missing_deps_as_failure(self):
        self.load_builder()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT.with_suffix(".sh")),
                    "12.9",
                    "--check-deps",
                    "--deps-dir",
                    directory,
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--prepare", result.stdout)

    def test_no_archive_can_escape_extraction_directory(self):
        builder = self.load_builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                entry = tarfile.TarInfo("../outside")
                entry.size = 2
                bundle.addfile(entry, io.BytesIO(b"no"))
            with self.assertRaises(tarfile.FilterError):
                builder.extract_archive(archive, root / "target")
            self.assertFalse((root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
