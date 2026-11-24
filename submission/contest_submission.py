"""Participant module loaded by /app/harness.py (contest_harness SDK). The harness lives in agent_harness.contest."""
from agent_harness.contest import TASKS, predict, train

__all__ = ["TASKS", "predict", "train"]
