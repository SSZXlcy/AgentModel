from __future__ import annotations

import os
import sys
from typing import Any

from .contracts import json_ready


class NullTracker:
    enabled = False

    def log(self, metrics: dict[str, Any]) -> None:
        return None

    def finish(self, summary: dict[str, Any]) -> None:
        return None


class SwanLabTracker:
    enabled = True

    def __init__(self, project: str, experiment: str, config: dict[str, Any], mode: str = "cloud"):
        if os.name == "nt":
            for stream in (sys.stdout, sys.stderr):
                if hasattr(stream, "reconfigure"):
                    stream.reconfigure(encoding="utf-8", errors="replace")
        try:
            import swanlab
        except ImportError as error:
            raise RuntimeError("SwanLab is not installed in .venv") from error
        api_key = os.getenv("SWANLAB_API_KEY")
        if api_key:
            swanlab.login(api_key=api_key)
        self.swanlab = swanlab
        self.step = 0
        self.run = swanlab.init(
            project=project, experiment_name=experiment, config=json_ready(config), mode=mode,
        )

    def log(self, metrics: dict[str, Any]) -> None:
        self.swanlab.log(json_ready(metrics), step=self.step)
        self.step += 1

    def finish(self, summary: dict[str, Any]) -> None:
        self.log({"summary/" + key: value for key, value in summary.items() if isinstance(value, (int, float))})
        self.swanlab.finish()

