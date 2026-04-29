from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import wandb

from .config import LoggingConfig


def start_wandb_run(
    *,
    config: dict[str, Any],
    logging_config: LoggingConfig,
    job_type: str,
    tags: list[str] | None = None,
) -> wandb.sdk.wandb_run.Run:
    return wandb.init(
        entity=logging_config.wandb_entity,
        project=logging_config.wandb_project,
        mode=logging_config.wandb_mode,
        job_type=job_type,
        tags=tags,
        config=config,
        sync_tensorboard=logging_config.sync_tensorboard,
        save_code=logging_config.save_code,
    )


def resolve_run_id(run: Any) -> str:
    run_id = getattr(run, "id", None)
    if run_id and run_id != "disabled":
        return str(run_id)
    return uuid.uuid4().hex[:8]


def prepare_output_dir(output_root: str | Path, run_id: str) -> Path:
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_wandb_files(
    paths: list[str | Path],
    *,
    base_path: str | Path,
) -> list[str]:
    if wandb.run is None:
        return []

    saved: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        result = wandb.save(str(path), base_path=str(base_path), policy="now")
        if isinstance(result, list):
            saved.extend(result)
        elif result:
            saved.append(str(result))
    return saved
