"""The only execution surface: one call of the official train_run produces one new tree node."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .codebase import CodeVersion
from .contracts import Cost
from .ledger import Ledger
from .recorder import Recorder
from .tree import Node, Tree


class Executor:
    def __init__(self, code: CodeVersion, profile: str, tree: Tree, ledger: Ledger, recorder: Recorder,
                 seconds_per_row_epoch: float):
        self.code = code
        self.profile = profile
        self.tree = tree
        self.ledger = ledger
        self.recorder = recorder
        # Running estimate used to refuse unaffordable training before it starts.
        self.rate = seconds_per_row_epoch
        # Deterministic training work (rows × epochs). Wall seconds of millisecond-scale calls are
        # dominated by timer noise, so cost comparisons between versions use this instead.
        self.row_epochs = 0

    def _train_run(self, profile_path, train_path, val_path, out, parent_ckpt, epochs, learning_rate, node_id):
        """The official trainer finishes with os.replace(staging, output). On Windows a freshly written
        file can be held briefly by a scanner, which fails that rename; the output then does not exist,
        so one retry is safe (the official code never overwrites an existing output)."""
        for attempt in range(2):
            try:
                return self.code.modules().train.train_run(
                    profile_path, train_path, val_path, out, resume=parent_ckpt, epochs=epochs,
                    learning_rate=learning_rate, stage_id=node_id)
            except PermissionError:
                if attempt or Path(out).exists():
                    raise
                time.sleep(0.5)

    def train_from_scratch(self, train_path: Path, val_path: Path, rows: int, *, learning_rate: float, window: int,
                           recipe_id: str, spec: str, epochs: int) -> Node:
        """The official configuration from initialisation (the procedure an opening k-fold validates).
        The node hangs under the official root so pointers and records stay one tree."""
        self.ledger.require(self.estimate(rows, epochs))
        node_id, out = self.tree.allocate()
        profile_path = self.code.profile_path(self.profile)
        for attempt in range(2):
            try:
                report = self.code.modules().train.train_run(str(profile_path), train_path, val_path, out,
                                                             learning_rate=learning_rate, stage_id=node_id)
                break
            except PermissionError:
                if attempt or Path(out).exists():
                    raise
                time.sleep(0.5)
        seconds = float(report["training_seconds"])
        self.ledger.charge(Cost(train_seconds=seconds))
        self.row_epochs += rows * int(report["final_epochs"])
        history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines() if line.strip()]
        root = self.tree.pointers["static"]
        node = self.tree.add(Node(
            node_id, root, (out / "checkpoint.npz").relative_to(self.tree.run_dir).as_posix(),
            report["checkpoint_sha256"], self.code.version_id, window, recipe_id, learning_rate,
            report["final_epochs"], report["final_steps"], seconds, history, spec))
        self.recorder.event("train", node=node_id, parent=None, window=window, spec=spec, recipe_id=recipe_id,
                            code_version=self.code.version_id, from_scratch=True, training_seconds=seconds, rows=rows,
                            checkpoint_sha256=report["checkpoint_sha256"], validation_after=report["validation_after"])
        return node

    def estimate(self, rows: int, epochs: int) -> Cost:
        return Cost(train_seconds=self.rate * rows * epochs)

    def train(self, parent: Node, train_path: Path, val_path: Path, rows: int, *, epochs: int,
              learning_rate: float, window: int, recipe_id: str, spec: str) -> Node:
        self.ledger.require(self.estimate(rows, epochs))
        node_id, out = self.tree.allocate()
        parent_ckpt = self.tree.checkpoint(parent.node_id)
        profile_path = self.code.profile_path(self.profile)
        command = ["python", "-m", "baselines.train", "--profile", profile_path.as_posix(),
                   "--train", Path(train_path).as_posix(), "--validation", Path(val_path).as_posix(),
                   "--resume", parent_ckpt.as_posix(), "--epochs", str(epochs),
                   "--learning-rate", repr(learning_rate), "--output", out.as_posix(), "--stage-id", node_id]
        started = time.monotonic()
        report = self._train_run(str(profile_path), train_path, val_path, out, parent_ckpt, epochs, learning_rate, node_id)
        wall = time.monotonic() - started
        seconds = float(report["training_seconds"])
        self.ledger.charge(Cost(train_seconds=seconds))
        self.row_epochs += rows * epochs
        self.rate =0.7 * self.rate + 0.3 * seconds / max(1, rows * epochs)
        history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines() if line.strip()]
        node = self.tree.add(Node(
            node_id, parent.node_id, (out / "checkpoint.npz").relative_to(self.tree.run_dir).as_posix(),
            report["checkpoint_sha256"], self.code.version_id, window, recipe_id, learning_rate,
            report["final_epochs"], report["final_steps"], seconds, history, spec))
        self.recorder.event("train", node=node_id, parent=parent.node_id, window=window, spec=spec,
                            recipe_id=recipe_id, code_version=self.code.version_id, command=command,
                            training_seconds=seconds, wall_seconds=wall, rows=rows,
                            parent_checkpoint_sha256=report["parent_checkpoint_sha256"],
                            checkpoint_sha256=report["checkpoint_sha256"],
                            validation_after=report["validation_after"])
        return node
