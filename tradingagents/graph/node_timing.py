"""Per-execution timing for LangGraph node runs."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class NodeExecutionRecord:
    """One completed execution of a graph node."""

    index: int
    node_name: str
    run_number: int
    duration_s: float


class NodeTimingTracker:
    """Records wall-clock duration for each graph node execution."""

    def __init__(self) -> None:
        self.records: list[NodeExecutionRecord] = []
        self._node_run_counts: dict[str, int] = {}
        self._last_ts: float | None = None
        self._start_ts: float | None = None

    def reset(self) -> None:
        self.records = []
        self._node_run_counts = {}
        self._last_ts = None
        self._start_ts = None

    def start_run(self) -> None:
        now = time.perf_counter()
        self._start_ts = now
        self._last_ts = now

    def record_update(self, node_name: str) -> None:
        if self._last_ts is None:
            raise RuntimeError("NodeTimingTracker.start_run() must be called first")

        now = time.perf_counter()
        duration_s = now - self._last_ts
        run_number = self._node_run_counts.get(node_name, 0) + 1
        self._node_run_counts[node_name] = run_number

        self.records.append(
            NodeExecutionRecord(
                index=len(self.records) + 1,
                node_name=node_name,
                run_number=run_number,
                duration_s=duration_s,
            )
        )
        self._last_ts = now

    @property
    def total_duration_s(self) -> float:
        if self._start_ts is None or self._last_ts is None:
            return 0.0
        return self._last_ts - self._start_ts

    def print_report(self) -> None:
        print()
        print("=" * 72)
        print("Node Execution Timing")
        print("=" * 72)
        print(f"{'#':<4}{'Node':<32}{'Run#':<6}{'Duration':>10}")
        print("-" * 72)
        for record in self.records:
            print(
                f"{record.index:<4}"
                f"{record.node_name:<32}"
                f"{record.run_number:<6}"
                f"{record.duration_s:>9.2f}s"
            )
        print("-" * 72)
        print(f"{'Total pipeline time:':<42}{self.total_duration_s:>9.2f}s")
        print("=" * 72)
