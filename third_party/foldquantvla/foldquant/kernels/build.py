# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Build TensorRT plugins from the packaged CUDA sources.

Binaries are compiled for the current device and written to the external
cache. Runtime loading is handled separately by ``kernels.locator``.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from foldquant.kernels import locator as loc

logger = logging.getLogger(__name__)


def _kernels_source_dir() -> Path:
    """Root of the TensorRT plugin sources (holds CMakeLists.txt)."""
    from foldquant.kernels.registry import tensorrt_source_root

    return tensorrt_source_root()


def _newest_source_mtime(src: Path) -> float:
    return max((p.stat().st_mtime for p in src.rglob("*") if p.is_file()), default=0.0)


def _resolve_cutlass_include() -> Optional[str]:
    """CUTLASS include dir from ``CUTLASS_INCLUDE_DIR`` env or the vendored fallback."""
    env = os.environ.get("CUTLASS_INCLUDE_DIR")
    if env and Path(env).expanduser().is_dir():
        return str(Path(env).expanduser())
    # Editable-checkout fallback (matches the CMakeLists default location):
    # the <repo>/third_party/cutlass submodule (NVIDIA/cutlass v3.6.0).
    parents = _kernels_source_dir().parents  # tensorrt -> kernels -> foldquant -> <repo>
    if len(parents) >= 3:
        cand = parents[2] / "third_party" / "cutlass" / "include"
        if cand.is_dir():
            return str(cand)
    return None


def _resolve_cuda_toolkit() -> tuple[Optional[Path], Optional[Path]]:
    """Resolve the CUDA root and headers, including target-specific layouts."""
    def is_cuda_include(candidate: Path) -> bool:
        return candidate.is_dir() and any(
            (candidate / header).is_file() for header in ("cuda_runtime.h", "cuda_runtime_api.h")
        )

    include: Optional[Path] = None
    include_env = os.environ.get("CUDAToolkit_INCLUDE_DIR")
    if include_env:
        candidate = Path(include_env).expanduser()
        if is_cuda_include(candidate):
            include = candidate

    root: Optional[Path] = None
    for name in ("CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value and Path(value).expanduser().is_dir():
            root = Path(value).expanduser()
            break

    if root is None:
        nvcc = shutil.which("nvcc")
        if nvcc:
            candidate = Path(nvcc).expanduser().resolve().parent.parent
            if candidate.is_dir():
                root = candidate

    if include is None and root is not None:
        candidates = [
            root / "include",
            root / "targets" / f"{platform.machine()}-linux" / "include",
        ]
        candidates.extend(sorted((root / "targets").glob("*/include")))
        include = next((candidate for candidate in candidates if is_cuda_include(candidate)), None)

    return root, include


def _cuda_cmake_args() -> list[str]:
    root, include = _resolve_cuda_toolkit()
    args: list[str] = []
    if root is not None:
        args.append(f"-DCUDAToolkit_ROOT={root}")
    if include is not None:
        args.append(f"-DCUDAToolkit_INCLUDE_DIR={include}")
        args.append(f"-DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES={include}")
    return args


def _resolve_trt_include() -> Optional[Path]:
    """TensorRT C++ headers, matching the library the plugin will link.

    Order: explicit ``TENSORRT_ROOT`` -> the vendored pinned headers -> system.

    The vendored ``third_party/tensorrt-headers/include`` outranks a system
    install because the headers must describe the SAME TensorRT as the library
    (see :func:`_resolve_trt_libnvinfer`); the pip wheel ships no headers, and
    the vendored set is pinned to its version.

    A mismatch is silent: TRT 11 headers against the pinned TRT 10.15 wheel
    produce creators that register without error but are unusable, and the
    first symptom is "Plugin not found, are the plugin name, version, and
    namespace correct?" in a later command.
    """
    env = os.environ.get("TENSORRT_ROOT")
    if env:
        hdr = next((Path(env).expanduser()).rglob("NvInferRuntime.h"), None)
        if hdr is not None:
            return hdr.parent

    parents = _kernels_source_dir().parents  # tensorrt -> kernels -> foldquant -> <repo>
    if len(parents) >= 3:
        headers_root = parents[2] / "third_party" / "tensorrt-headers"
        # Two vendored header sets, selected by the INSTALLED TensorRT major so
        # the headers always describe the library being linked: include/ is
        # the 11.x set (x86_64 servers), include-trt10/ the 10.x set (Jetson
        # Orin and any other TRT 10 deployment). Compiling 11.x headers
        # against libnvinfer.so.10 (or vice versa) is the silent
        # ABI-mismatch failure the docstring above describes.
        subdir = "include"
        try:
            import tensorrt as _trt

            if int(str(_trt.__version__).split(".")[0]) < 11:
                subdir = "include-trt10"
        except Exception:  # noqa: BLE001 - no tensorrt importable: fall through to defaults
            pass
        cand = headers_root / subdir
        if (cand / "NvInferRuntime.h").is_file():
            return cand
        # Version-matched set missing: fall back to the primary vendored set
        # rather than silently drifting to system headers.
        cand = headers_root / "include"
        if (cand / "NvInferRuntime.h").is_file():
            return cand

    for base in (Path("/usr"), Path("/usr/local/tensorrt"), Path("/opt/tensorrt")):
        hdr = next(base.rglob("NvInferRuntime.h"), None)
        if hdr is not None:
            return hdr.parent
    return None


def _resolve_trt_libnvinfer() -> Optional[Path]:
    """A libnvinfer the linker can use, matching the one the RUNTIME will load.

    Order: explicit ``TENSORRT_ROOT`` -> the installed ``tensorrt-libs`` wheel ->
    a system TensorRT.

    The wheel outranks the system install: a plugin registers its creators into
    the libnvinfer it links, and the builder looks them up in the one the Python
    process loaded, which is the pinned ``tensorrt-cu12`` wheel. Linking a
    different system TensorRT loads cleanly but fails the build with "plugin not
    found".

    ``find_library(nvinfer)`` needs the bare ``libnvinfer.so`` linker name; the
    wheel ships only the ``.so.<N>`` SONAME, so :func:`_synthesize_trt_root`
    symlinks it.
    """
    env = os.environ.get("TENSORRT_ROOT")
    if env:
        lib = next((Path(env).expanduser()).rglob("libnvinfer.so"), None)
        if lib is not None:
            return lib

    try:
        import tensorrt_libs  # the pip tensorrt-libs wheel
    except ImportError:
        pass
    else:
        wheel = sorted(Path(tensorrt_libs.__file__).parent.glob("libnvinfer.so.*"))
        if wheel:
            return wheel[0]

    for base in (Path("/usr"), Path("/usr/local/tensorrt"), Path("/opt/tensorrt")):
        lib = next(base.rglob("libnvinfer.so"), None)
        if lib is not None:
            return lib
    return None


def _synthesize_trt_root(work: Path) -> Optional[str]:
    """Assemble a TENSORRT_ROOT (``include/`` + ``lib/libnvinfer.so``) under *work*.

    Lets the uv-native env (CUDA + ``tensorrt``/``tensorrt-libs`` wheels) compile the
    plugins with no TensorRT dev SDK: vendored headers + a synthesized linker symlink to
    the wheel's runtime ``libnvinfer.so.<N>``. Returns None if headers or lib are missing.
    """
    include = _resolve_trt_include()
    libnvinfer = _resolve_trt_libnvinfer()
    if include is None or libnvinfer is None:
        return None
    root = work / "_trt_root"
    (root / "lib").mkdir(parents=True, exist_ok=True)
    inc_link = root / "include"
    if inc_link.is_symlink() or inc_link.exists():
        inc_link.unlink()
    inc_link.symlink_to(include)

    # Two symlinks, both required, for different phases:
    #   libnvinfer.so      : the linker name CMake's find_library() resolves.
    #   libnvinfer.so.<N>  : the SONAME the dynamic loader asks for at runtime.
    #
    # This directory lands in the built plugin's RUNPATH. If only the linker
    # name is present, the loader cannot satisfy the SONAME here and falls
    # through to the system search path, so on a host with its own TensorRT the
    # plugin binds a SECOND libnvinfer, registers its creators into that
    # library's registry, and those creators are invisible to the one the Python
    # process loaded. registerCreator() still returns true, so nothing reports an
    # error until the engine build fails with "plugin not found".
    lib_names = ["libnvinfer.so"]
    if libnvinfer.name != "libnvinfer.so":
        lib_names.append(libnvinfer.name)  # e.g. libnvinfer.so.10
    for name in lib_names:
        link = root / "lib" / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(libnvinfer)
    return str(root)


def _invalidate_stale_cmake_cache(build_tmp: Path, src: Path) -> None:
    """Wipe *build_tmp* if its ``CMakeCache.txt`` was configured from a different source dir.

    ``build_tmp`` lives outside the repo (``~/.cache/foldquant/...``) and survives
    across checkouts/branches, so a source-tree reorg (kernels have already moved
    once, out of ``modules/kernels/``) leaves behind a cache CMake refuses to
    reconfigure in place ("does not match the source ... used to generate
    cache"). Otherwise it is a permanent, non-self-healing failure rather than a
    one-time rebuild.
    """
    cache_file = build_tmp / "CMakeCache.txt"
    if not cache_file.exists():
        return
    cached_src: Optional[str] = None
    for line in cache_file.read_text().splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:"):
            cached_src = line.split("=", 1)[1].strip()
            break
    if cached_src is not None and Path(cached_src).resolve() != src.resolve():
        logger.info(
            "plugin build cache %s was configured from %s, not the current source %s; wiping and reconfiguring",
            build_tmp,
            cached_src,
            src,
        )
        shutil.rmtree(build_tmp)


def _run(cmd: List[str]) -> None:
    logger.info("plugin build: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
        raise RuntimeError(f"plugin build command failed ({cmd[0]} rc={proc.returncode}):\n{tail}")


def rebuild_plugins(out_dir: Path, libs: Optional[Sequence[str]] = None) -> None:
    """Compile the requested plugin libs for this device into ``out_dir``.

    ``libs`` defaults to every standardized lib. Scoping the build keeps one
    lib's missing dependency from breaking a caller that never asked for it
    (an INT4-only build must not die on the INT8 lib's CUTLASS dependency;
    CUTLASS is required by ``foldquant_int8_per_row``, not
    ``foldquant_int4_groupwise``).
    """
    names = list(libs) if libs is not None else list(loc.KNOWN_PLUGIN_LIBS)
    unknown = sorted(set(names) - set(loc.KNOWN_PLUGIN_LIBS))
    if unknown:
        raise ValueError(f"unknown plugin lib(s) {unknown}; known: {list(loc.KNOWN_PLUGIN_LIBS)}")

    # Serialize ALL rebuilds, not just the staleness-triggered ones in
    # ensure_plugins_for_current_device: a manual `python -m foldquant.kernels build`
    # racing a build lane through the shared cmake _build dir interleaves
    # object writes into a corrupt .so that dlopen()s straight into SIGSEGV.
    # The lock lives beside the shared build dir so every caller contends on
    # the same file regardless of out_dir.
    import fcntl

    lock_dir = loc.plugin_cache_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / ".rebuild.lock", "w") as _lockf:
        fcntl.flock(_lockf, fcntl.LOCK_EX)
        try:
            _rebuild_plugins_unlocked(out_dir, names)
        finally:
            fcntl.flock(_lockf, fcntl.LOCK_UN)


def _rebuild_plugins_unlocked(out_dir: Path, names: "list[str]") -> None:
    """The actual cmake configure+build+copy. Call only under the rebuild lock."""
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("`cmake` not found on PATH; cannot rebuild TensorRT plugins for this device.")
    cutlass = _resolve_cutlass_include()
    # Ask the registry which kernels need CUTLASS rather than restating it here:
    # a hardcoded set is how foldquant_int4_per_row came to be treated as
    # CUTLASS-free. Checking before CMake turns a confusing mid-compile
    # "No such file" into a message naming the library and the fix.
    from foldquant.kernels.registry import kernels_needing

    needs_cutlass = sorted(set(kernels_needing(cutlass=True)) & set(names))
    if cutlass is None and needs_cutlass:
        raise RuntimeError(
            f"Building {', '.join(needs_cutlass)} needs CUTLASS headers. Run "
            "`git submodule update --init third_party/cutlass` or set CUTLASS_INCLUDE_DIR."
        )
    src = _kernels_source_dir()
    build_tmp = loc.plugin_cache_dir() / "_build"
    _invalidate_stale_cmake_cache(build_tmp, src)
    build_tmp.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    trt_root = _synthesize_trt_root(build_tmp)
    if trt_root is None:
        raise RuntimeError(
            "TensorRT plugin rebuild needs NvInfer*.h headers and a libnvinfer to link. With the pip "
            "`tensorrt`/`tensorrt-libs` wheels installed and the vendored headers present this resolves "
            "automatically; otherwise install the wheels (`uv sync`) or set TENSORRT_ROOT "
            "to a TensorRT dev SDK. (`git submodule update --init third_party/cutlass` is also required.)"
        )

    configure_cmd = [
        cmake,
        "-S",
        str(src),
        "-B",
        str(build_tmp),
        f"-DCMAKE_CUDA_ARCHITECTURES={loc.sm_arch_number()}",
        f"-DTENSORRT_ROOT={trt_root}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    configure_cmd.extend(_cuda_cmake_args())
    if cutlass is not None:
        configure_cmd.append(f"-DCUTLASS_INCLUDE_DIR={cutlass}")
    _run(configure_cmd)
    # CMake target names match the lib names (``add_library(foldquant_int4_groupwise ...)``),
    # so the requested libs are the targets. ``-j`` stays last: it takes an optional
    # value and would otherwise swallow the target list.
    _run([cmake, "--build", str(build_tmp), "--target", *names, "-j"])

    for libname in names:
        built = next(build_tmp.rglob(f"lib{libname}.so"), None)
        if built is None:
            raise RuntimeError(f"plugin rebuild did not produce lib{libname}.so under {build_tmp}.")
        dest = out_dir / f"{libname}.so"
        _install_built_lib(built, dest)
        logger.info("rebuilt plugin: %s -> %s", libname, dest)


def _install_built_lib(built: Path, dest: Path) -> None:
    """Install *built* at *dest* atomically (new inode), never overwriting in place.

    A process that already ``dlopen``-ed the previous ``dest`` (this build's
    own parent after an earlier module loaded it, a serving backend on the
    same host) keeps its mapping of the old inode. Overwriting in place
    (``shutil.copy2`` onto the existing file) rewrites the pages that mapping
    points at and the next call into the library is a SIGSEGV.

    The install is stamped with "now" rather than the source mtime ``copy2``
    preserves: cmake does not relink an up-to-date lib, so an unchanged
    library would otherwise keep an old mtime and read as "stale vs edited
    source" forever, re-triggering the rebuild on every build.
    """
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    shutil.copy2(built, tmp)
    os.utime(tmp, None)
    os.replace(tmp, dest)


def ensure_plugins_for_current_device(
    libnames: Iterable[str] = loc.KNOWN_PLUGIN_LIBS,
) -> List[Path]:
    """Return usable ``.so`` paths for *libnames*, rebuilding from source if needed.

    Committed device-matched binaries are used as-is. A cache binary is rebuilt
    only when missing or older than the vendored source (idempotent otherwise).
    """
    names = list(libnames)
    cache = loc.plugin_cache_dir()
    src_mtime = _newest_source_mtime(_kernels_source_dir())

    # Rebuild only what is missing or stale. Rebuilding every requested lib
    # would re-link libraries this process has already dlopen()-ed for an
    # earlier module (a mixed-width LLM asks for the INT4 lib the DiT loaded
    # plus the INT8 lib). A rebuilt copy cannot be re-loaded into a running
    # process, so a lib that is loaded is left exactly as loaded.
    stale: List[str] = []
    for name in names:
        resolved = loc.resolve_plugin_so(name)
        if resolved is None:
            stale.append(name)
        elif resolved.parent == cache and resolved.stat().st_mtime < src_mtime:
            if loc.is_plugin_loaded(resolved):
                logger.warning(
                    "plugin %s is older than the kernel sources but is already loaded in this process; "
                    "keeping the loaded copy (rebuild it with `python -m foldquant.kernels build --force`).",
                    resolved,
                )
            else:
                stale.append(name)  # cache stale vs edited source

    if stale:
        # rebuild_plugins holds the cross-process rebuild lock itself (two
        # concurrent cmake runs into the shared _build dir corrupt the .so,
        # measured as rc=-11 on two parallel Pi lanes). A process that had to
        # wait on another's rebuild re-checks staleness inside rebuild_plugins'
        # critical section via the fresh binaries it copies; here we simply
        # call it. The lock must NOT also be taken at this level (nested
        # flock on a second fd of the same file self-deadlocks).
        rebuild_plugins(cache, stale)

    final: List[Path] = []
    for name in names:
        resolved = loc.resolve_plugin_so(name)
        if resolved is None:
            raise RuntimeError(f"plugin {name!r} still unresolved after rebuild into {cache}.")
        final.append(resolved)
    return final
