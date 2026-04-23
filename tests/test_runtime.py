from __future__ import annotations

from contact_aware_rl.runtime import default_video_stem


def test_default_video_stem_uses_checkpoint_parent_name() -> None:
    assert default_video_stem("outputs/9ioilbgv/best_success_model.zip") == "9ioilbgv"


def test_default_video_stem_falls_back_to_checkpoint_stem() -> None:
    assert default_video_stem("best_success_model.zip") == "best_success_model"
