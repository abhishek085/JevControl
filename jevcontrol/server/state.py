"""Where JevControl keeps things on disk: experiments, downloaded weights, the HF cache used for pulls."""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    p = Path(os.environ.get("JEVCONTROL_HOME", ".jevcontrol")).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = Path(os.environ.get("JEVCONTROL_MODELS", "models")).expanduser().resolve()
    return p
