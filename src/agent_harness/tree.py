"""Checkpoint tree. Every node is an immutable official output directory; pointers move, files never do.

Pointers:
  static       the official checkpoint, never retrained (baseline floor, threshold ③)
  confirmed    passed the next-window check; the rollback target
  provisional  passed the gate in its own window, serving until the next window confirms it
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import shutil

from .recorder import write_json


@dataclass
class Node:
    node_id: str
    parent_id: str | None
    path: str                      # relative to the run directory
    sha256: str
    code_version: str
    window: int
    recipe_id: str | None = None
    learning_rate: float | None = None
    epochs: int = 0                # cumulative, from the checkpoint metadata
    steps: int = 0
    train_seconds: float = 0.0
    history: list[dict] = field(default_factory=list)
    spec: str = ""


class Tree:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.nodes: dict[str, Node] = {}
        self.pointers: dict[str, str | None] = {"static": None, "confirmed": None, "provisional": None, "shadow": None}
        self._models: dict[str, object] = {}

    # ------------------------------------------------------------ structure
    def add_root(self, checkpoint: Path, code_version: str, meta: dict) -> Node:
        directory = self.run_dir / "nodes" / "n0000"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "checkpoint.npz"
        if not target.exists():
            shutil.copyfile(checkpoint, target)
        node = Node("n0000", None, target.relative_to(self.run_dir).as_posix(), _sha(target), code_version, -1,
                    epochs=meta["epochs"], steps=meta["steps"], learning_rate=meta["learning_rate"], spec="official")
        self.nodes[node.node_id] = node
        self.pointers.update(static=node.node_id, confirmed=node.node_id, provisional=None)
        return node

    def allocate(self) -> tuple[str, Path]:
        node_id = f"n{len(self.nodes):04d}"
        return node_id, self.run_dir / "nodes" / node_id

    def add(self, node: Node) -> Node:
        self.nodes[node.node_id] = node
        return node

    def checkpoint(self, node_id: str) -> Path:
        return self.run_dir / self.nodes[node_id].path

    # ------------------------------------------------------------ pointers
    @property
    def serving(self) -> str:
        return self.pointers["provisional"] or self.pointers["confirmed"]

    def promote_provisional(self, node_id: str) -> None:
        self.pointers["provisional"] = node_id

    def confirm(self) -> None:
        if self.pointers["provisional"]:
            self.pointers["confirmed"] = self.pointers["provisional"]
            self.pointers["provisional"] = None

    def rollback(self) -> None:
        self.pointers["provisional"] = None

    def reset_to_static(self) -> None:
        self.pointers.update(confirmed=self.pointers["static"], provisional=None, shadow=None)

    def set_shadow(self, node_id: str | None) -> None:
        """A candidate compared beside the serving model; it serves only after it wins next window."""
        self.pointers["shadow"] = node_id

    def adopt_shadow(self) -> None:
        if self.pointers.get("shadow"):
            self.pointers.update(confirmed=self.pointers["shadow"], provisional=None, shadow=None)

    # ------------------------------------------------------------ models
    def model(self, node_id: str, model_class):
        if node_id not in self._models:
            self._models[node_id] = model_class.load(self.checkpoint(node_id))
        return self._models[node_id]

    # ------------------------------------------------------------ persistence
    def save(self) -> None:
        write_json(self.run_dir / "tree.json", {"pointers": self.pointers,
                                                 "nodes": [asdict(n) for n in self.nodes.values()]})

    @classmethod
    def load(cls, run_dir: Path) -> "Tree":
        import json
        tree = cls(run_dir)
        data = json.loads((Path(run_dir) / "tree.json").read_text(encoding="utf-8"))
        tree.pointers = {"shadow": None, **data["pointers"]}
        tree.nodes = {n["node_id"]: Node(**n) for n in data["nodes"]}
        return tree


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
