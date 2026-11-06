from __future__ import annotations

import importlib.util
import os
import pkgutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ENTRYPOINT = PROJECT_ROOT / "main.py"
APP_NAME = "EcoAcousticSentinel"
ICON_PLACEHOLDER = PROJECT_ROOT / "assets" / "app.ico"


def _data_sep() -> str:
    return ";" if os.name == "nt" else ":"


def _add_data_arg(src: Path, dst: str) -> str:
    return f"{src}{_data_sep()}{dst}"


def _module_exists(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _discover_birdnet_models_dir() -> Path | None:
    # Prefer a local models/ folder if present in repository root.
    local_models = PROJECT_ROOT / "models"
    if local_models.exists() and local_models.is_dir():
        return local_models

    # Fallback to BirdNET checkpoints in site-packages.
    try:
        import birdnet_analyzer.config as cfg
    except Exception:
        return None

    candidates: list[Path] = []
    for attr in ("BIRDNET_MODEL_PATH", "BIRDNET_LABELS_FILE"):
        raw = getattr(cfg, attr, "")
        if not raw:
            continue
        p = Path(str(raw))
        if not p.exists():
            continue
        if p.is_file():
            candidates.append(p.parent)
            if p.parent.parent.name.lower() == "checkpoints":
                candidates.append(p.parent.parent)
        elif p.is_dir():
            candidates.append(p)

    # Prefer checkpoint root, then first available.
    for c in candidates:
        if c.name.lower() == "checkpoints":
            return c
    return candidates[0] if candidates else None


def _torch_backend_hidden_imports() -> list[str]:
    imports = ["torch.backends"]
    try:
        import torch.backends as torch_backends

        for mod in pkgutil.iter_modules(torch_backends.__path__):
            imports.append(f"torch.backends.{mod.name}")
    except Exception:
        imports.extend(
            [
                "torch.backends.mkldnn",
                "torch.backends.quantized",
                "torch.backends.mps",
                "torch.backends.cuda",
            ]
        )
    # Deduplicate preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for name in imports:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def build() -> None:
    if not ENTRYPOINT.exists():
        raise FileNotFoundError(f"No se encontro el entrypoint: {ENTRYPOINT}")

    try:
        from PyInstaller.__main__ import run as pyinstaller_run
    except Exception as ex:
        raise RuntimeError(
            "PyInstaller no esta instalado en el entorno activo. Ejecuta: pip install pyinstaller"
        ) from ex

    profiles_dir = PROJECT_ROOT / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    datas: list[tuple[Path, str]] = []
    if (PROJECT_ROOT / "ui_theme.py").exists():
        datas.append((PROJECT_ROOT / "ui_theme.py", "."))
    datas.append((profiles_dir, "profiles"))
    assets_dir = PROJECT_ROOT / "assets"
    if assets_dir.exists() and assets_dir.is_dir():
        datas.append((assets_dir, "assets"))
    local_models_dir = PROJECT_ROOT / "local_models"
    if local_models_dir.exists() and local_models_dir.is_dir():
        datas.append((local_models_dir, "local_models"))
        print(f"[INFO] Incluyendo clasificador local desde: {local_models_dir}")

    models_dir = _discover_birdnet_models_dir()
    if models_dir is not None and models_dir.exists():
        datas.append((models_dir, "models"))
        print(f"[INFO] Incluyendo modelos BirdNET desde: {models_dir}")
    else:
        print("[WARNING] No se encontro carpeta de modelos BirdNET (models/ o checkpoints).")

    hidden_imports = [
        "sklearn.utils._cython_blas",
        "sklearn.neighbors.typedefs",
        "pandas",
        "openpyxl",  # pandas lo importa de forma diferida (lectura de TablaFinal.xlsx)
        "validation",
        "torch",
        "torchaudio",
    ]
    hidden_imports.extend(_torch_backend_hidden_imports())

    # Optional extras only if available in the active environment.
    optional_hidden = [
        "torch._C",
        "torchaudio._extension",
        "torchaudio.lib._torchaudio",
    ]
    for module_name in optional_hidden:
        if _module_exists(module_name):
            hidden_imports.append(module_name)

    # Deduplicate preserving order.
    dedup_hidden: list[str] = []
    seen_hidden: set[str] = set()
    for name in hidden_imports:
        if name in seen_hidden:
            continue
        seen_hidden.add(name)
        dedup_hidden.append(name)

    args = [
        str(ENTRYPOINT),
        "--name",
        APP_NAME,
        "--noconfirm",
        "--clean",
        "--noconsole",
        "--onedir",
        "--collect-submodules",
        "birdnet_analyzer",
        "--collect-submodules",
        "torch.backends",
    ]

    if ICON_PLACEHOLDER.exists():
        args.extend(["--icon", str(ICON_PLACEHOLDER)])
    else:
        print(f"[WARNING] Icono no encontrado. Placeholder configurado en: {ICON_PLACEHOLDER}")

    for src, dst in datas:
        args.extend(["--add-data", _add_data_arg(src, dst)])

    for module_name in dedup_hidden:
        args.extend(["--hidden-import", module_name])

    print("[INFO] Ejecutando PyInstaller...")
    pyinstaller_run(args)
    print("[INFO] Build completado. Revisa la carpeta dist/.")


if __name__ == "__main__":
    try:
        build()
    except Exception as err:
        print(f"[ERROR] {err}")
        sys.exit(1)
