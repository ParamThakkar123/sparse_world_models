from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentLogger:
    """Minimal file-based experiment logger for early-stage research."""

    root_dir: str | Path = "experiments/runs"
    run_name: str | None = None
    run_dir: Path = field(init=False)
    metrics_path: Path = field(init=False)
    config_path: Path = field(init=False)
    summary_path: Path = field(init=False)
    _fieldnames: list[str] = field(default_factory=lambda: ["step"], init=False)

    def __post_init__(self) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = self.run_name or f"run_{timestamp}"
        self.run_dir = Path(self.root_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.config_path = self.run_dir / "config.json"
        self.summary_path = self.run_dir / "summary.json"

    def log_config(self, config: dict[str, Any]) -> None:
        with self.config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)

    def log_metrics(self, step: int, **metrics: float) -> None:
        row = {"step": step, **metrics}
        if not self.metrics_path.exists():
            self._fieldnames = list(row.keys())
            self._write_header()
        else:
            new_fields = [key for key in row.keys() if key not in self._fieldnames]
            if new_fields:
                self._rewrite_with_new_fields(new_fields)

        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow(row)

    def log_summary(self, summary: dict[str, Any]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    def _write_header(self) -> None:
        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()

    def _rewrite_with_new_fields(self, new_fields: list[str]) -> None:
        self._fieldnames.extend(new_fields)
        with self.metrics_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

