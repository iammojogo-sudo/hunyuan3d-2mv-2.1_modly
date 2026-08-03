"""
Hunyuan3D-2mv — extension setup script (self-contained, no WinPortable).

Creates an isolated venv with a CUDA torch build + the hy3dgen library
(installed from the Tencent Hunyuan3D-2 source) and ALL runtime deps used by
the bridges (shape, texture, texture_apply + the first-load bootstrap). Model
weights are downloaded separately by Modly's model-install step (manifest
hf_repo / hf_include_prefixes / download_check) into Modly's models/ folder and
read OFFLINE at generation time — the first-load bridge (hunyuan3d_bootstrap.py)
links them into the layout hy3dgen expects on first load.

Called by Modly at extension install time with:

    python setup.py '<json_args>'

where json_args contains:
    python_exe   — path to Modly's embedded Python (used to create the venv)
    ext_dir      — absolute path to this extension directory
    torch_flavor — Flavor of torch to use (cuda, rocm - defaults to cuda)
    gpu_sm       — GPU compute capability as integer
    cuda_version — CUDA major/minor encoded as integer
    accelerator  — "mps" | "cuda" | "cpu"
    platform     — Electron's process.platform string

Standard packaging mode (e.g. `pip install .` from the GitHub repo) installs
every runtime dependency, including hy3dgen from the Tencent source.
"""
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

VERSION = "2.1.0"

# Directly imported by the bridges (beyond hy3dgen's own transitive deps).
RUNTIME_DEPS = [
    "Pillow",
    "numpy",
    "diffusers",
    "transformers>=4.48.0",
    "accelerate",
    "einops",
    "safetensors",
    "trimesh",
    "Rtree",  # spatial index for trimesh ray intersection
    "rembg",
    "onnxruntime",  # rembg CPU session (never conflicts with CUDA torch)
    "omegaconf",
    "opencv-python-headless",
    "xatlas",
    "pygltflib",
    "pymeshlab",
    "huggingface_hub",
    "pyrender",
    "scipy",  # texture bridge inpaint (distance_transform_edt)
    "scikit-image",  # texture bridge deform (ThinPlateSplineTransform)
    "tqdm",
    "PyYAML",
    "packaging",
    "requests",
]


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def _install_custom_rasterizer(venv: Path, ext_dir: Path) -> None:
    """Install the bundled prebuilt custom_rasterizer into the venv.

    The extension ships `custom_rasterizer/` (pure-python package + prebuilt
    `custom_rasterizer_kernel.cp311-win_amd64.pyd`). Copying that dir into
    site-packages makes `import custom_rasterizer` and
    `import custom_rasterizer_kernel` resolve with no compilation step.
    Non-fatal: if the bundle is missing, texture gen will fail later with a
    clear import error rather than breaking the whole install.
    """
    is_win = platform.system() == "Windows"
    bundle = ext_dir / "custom_rasterizer"
    if not bundle.exists():
        print("[setup] custom_rasterizer bundle not found — skipping (texture gen needs it).")
        return
    site_pkgs = venv / ("Lib/site-packages" if is_win else "lib/python*/site-packages")
    if is_win:
        site_pkgs = venv / "Lib" / "site-packages"
    else:
        matches = sorted(site_pkgs.parent.glob("python*/site-packages"))
        site_pkgs = matches[-1] if matches else site_pkgs

    dest = site_pkgs / "custom_rasterizer"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(str(bundle), str(dest))
    print(f"[setup] custom_rasterizer installed -> {dest}")

    # Also drop a .pth so the bare `custom_rasterizer_kernel` .pyd inside the
    # package dir is importable from anywhere (some import paths expect it on
    # the top-level path, not inside the package).
    pth = site_pkgs / "hy3d_custom_rasterizer.pth"
    pth.write_text(str(dest) + "\n")
    print(f"[setup]   path file -> {pth}")


def _predownload_rembg(venv: Path) -> None:
    """Warm rembg's U²-Net cache (~170 MB) so first runs work offline."""
    is_win = platform.system() == "Windows"
    py = venv / ("Scripts/python.exe" if is_win else "bin/python")
    code = ("from rembg import new_session;"
            "s = new_session(providers=['CPUExecutionProvider']);"
            "print('rembg u2net cached')")
    try:
        subprocess.run([str(py), "-c", code], check=True, timeout=900)
        print("[setup] rembg u2net pre-downloaded (offline-ready).")
    except Exception as e:
        print(f"[setup] rembg u2net pre-download skipped ({e}). "
              "First run with a background will fetch it (needs network).")


def _torch_index_and_pkgs(gpu_sm: int, cuda_version: int, torch_flavor: str, is_mac: bool, is_win: bool):
    if is_mac:
        return None, ["torch", "torchvision", "torchaudio"]
    if torch_flavor == "rocm":
        if is_win:
            return "https://download.pytorch.org/whl/cpu", ["torch==2.6.0", "torchvision==0.21.0"]
        return "https://download.pytorch.org/whl/rocm7.2", ["torch", "torchvision", "torchaudio"]
    if gpu_sm >= 100 or cuda_version >= 128:
        return "https://download.pytorch.org/whl/cu128", ["torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0"]
    if gpu_sm >= 70:
        return "https://download.pytorch.org/whl/cu124", ["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"]
    return "https://download.pytorch.org/whl/cu118", ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]


def setup(
    python_exe:    str,
    ext_dir:       Path,
    gpu_sm:        int = 0,
    cuda_version:  int = 0,
    torch_flavor:  str = "cuda",
    accelerator:   str = "",
    platform_name: str = "",
    model_dir:     str = "",
    **_extra,
) -> None:
    ext_dir = Path(ext_dir)
    venv = ext_dir / "venv"
    is_win = platform.system() == "Windows"
    is_mac = platform.system() == "Darwin" or platform_name == "darwin"
    machine = platform.machine().lower()
    is_linux_arm64 = platform.system() == "Linux" and machine in {"aarch64", "arm64"}

    if not accelerator:
        if is_mac:
            accelerator = "mps" if machine == "arm64" else "cpu"
        elif gpu_sm > 0:
            accelerator = "cuda"
        else:
            accelerator = "cpu"

    print(f"[setup] accelerator={accelerator}  gpu_sm={gpu_sm}  cuda_version={cuda_version}")

    if venv.exists():
        print(f"[setup] venv already exists at {venv} — reusing it (Repair).")
    else:
        print(f"[setup] Creating venv at {venv} …")
        subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    torch_index, torch_pkgs = _torch_index_and_pkgs(gpu_sm, cuda_version, torch_flavor, is_mac, is_win)
    if torch_index:
        print(f"[setup] Installing torch from {torch_index}: {torch_pkgs}")
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
    else:
        print(f"[setup] Installing torch (PyPI): {torch_pkgs}")
        pip(venv, "install", *torch_pkgs)

    # hy3dgen (shapegen + texgen) from the Tencent Hunyuan3D-2 source.
    # Installed WITH its declared deps so the bridge imports cleanly.
    print("[setup] Installing hy3dgen from Tencent/Hunyuan3D-2 (source) …")
    pip(venv, "install", "git+https://github.com/Tencent/Hunyuan3D-2.git")

    # Runtime deps for the orchestrator + bridge.
    print("[setup] Installing Hunyuan3D-2mv runtime deps …")
    pip(venv, "install", *RUNTIME_DEPS)

    # Warm rembg's U²-Net cache (~170 MB) so first runs work offline.
    _predownload_rembg(venv)

    # ------------------------------------------------------------------ #
    # custom_rasterizer (texture gen) — install the PREBUILT kernel that
    # ships with this extension. hy3dgen's texgen imports `custom_rasterizer`
    # (a pure-python package) which in turn imports the compiled
    # `custom_rasterizer_kernel` CUDA extension. We bundle both (the package
    # source + the prebuilt .pyd for CPython 3.11) under this extension dir and
    # copy them into site-packages so `import custom_rasterizer` resolves with
    # NO compiler / CUDA toolkit needed at install time. Shape generation does
    # not need this; only the texture (paint) node does.
    # ------------------------------------------------------------------ #
    _install_custom_rasterizer(venv, ext_dir)

    print("[setup] Done. Venv ready at:", venv)
    print("[setup] Model weights are installed via Modly's model-download step.")


def _packaging_setup() -> None:
    """Standard packaging mode: `pip install .` / `pip install git+<repo>`.

    Installs ALL runtime dependencies (torch + torchvision + torchaudio from
    PyPI by default; install a CUDA build separately on GPU machines) plus
    hy3dgen from the Tencent source so the extension is self-contained.
    """
    from setuptools import setup

    readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

    setup(
        name="modly-hunyuan3d-2mv",
        version=VERSION,
        description="Hunyuan3D-2mv multi-view to 3D mesh generation (shape + paint texture) as a Modly extension.",
        long_description=readme,
        long_description_content_type="text/markdown",
        author="iammojogo",
        license="MIT",
        python_requires=">=3.10",
        py_modules=[
            "generator",
            "hunyuan3d_bridge",
            "hunyuan3d_texture_bridge",
            "hunyuan3d_texture_apply_bridge",
            "hunyuan3d_bootstrap",
        ],
        include_package_data=True,
        install_requires=[
            "hy3dgen @ git+https://github.com/Tencent/Hunyuan3D-2.git",
            "torch",
            "torchvision",
            "torchaudio",
            *RUNTIME_DEPS,
        ],
    )


if __name__ == "__main__":
    if len(sys.argv) >= 2 and (sys.argv[1].startswith("-") or sys.argv[1] in (
        "egg_info", "bdist_wheel", "sdist", "develop", "install", "build",
    )):
        _packaging_setup()
    elif len(sys.argv) >= 4 and not sys.argv[1].startswith("{"):
        setup(
            python_exe   = sys.argv[1],
            ext_dir      = Path(sys.argv[2]),
            gpu_sm       = int(sys.argv[3]),
            cuda_version = int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
            torch_flavor = sys.argv[5] if len(sys.argv) >= 6 else "cuda",
        )
    elif len(sys.argv) == 2 and sys.argv[1].startswith("{"):
        args = json.loads(sys.argv[1])
        setup(
            python_exe    = args["python_exe"],
            ext_dir       = Path(args["ext_dir"]),
            gpu_sm        = int(args.get("gpu_sm", 0)),
            cuda_version  = int(args.get("cuda_version", 0)),
            torch_flavor  = args.get("torch_flavor", "cuda"),
            accelerator   = args.get("accelerator", ""),
            platform_name = args.get("platform", ""),
            model_dir     = args.get("model_dir", ""),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version] [torch_flavor]")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86}\'')
        sys.exit(1)
