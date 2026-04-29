from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / ".cache"


def configure_runtime_environment() -> None:
    """Set writable cache directories for local training utilities."""
    CACHE_ROOT.mkdir(exist_ok=True)

    matplotlib_cache = CACHE_ROOT / "matplotlib"
    fontconfig_cache = CACHE_ROOT / "fontconfig"

    matplotlib_cache.mkdir(exist_ok=True)
    fontconfig_cache.mkdir(exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
    os.environ.setdefault("FONTCONFIG_PATH", str(fontconfig_cache))
    os.environ.setdefault("WANDB_DIR", str(PROJECT_ROOT / "wandb"))


def default_video_stem(model_path: str | Path) -> str:
    path = Path(model_path).expanduser()
    return path.parent.name or path.stem
