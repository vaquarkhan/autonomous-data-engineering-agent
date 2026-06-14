from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkerPool:
    max_workers: int = 4
    active_workers: int = 0
    pruned_slots: int = 0
    boost_factor: float = 1.5
    allocations: dict[str, int] = field(default_factory=dict)

    def reclaim_from_prune(self, pruned_count: int) -> int:
        """Reclaim GPU slots from pruned branches."""
        self.pruned_slots += pruned_count
        freed = int(pruned_count * self.boost_factor)
        return freed

    def allocate(self, branch_id: str, potential: float, max_concurrent: int) -> int:
        """Allocate workers to high-potential branches (Adaptive Boosting)."""
        if self.active_workers >= max_concurrent:
            bonus = min(self.pruned_slots, max_concurrent - self.active_workers)
            self.pruned_slots = max(0, self.pruned_slots - bonus)
            workers = 1 + bonus
        else:
            workers = max(1, int(1 + potential * self.boost_factor))

        workers = min(workers, max_concurrent - self.active_workers)
        self.allocations[branch_id] = workers
        self.active_workers += workers
        return workers

    def release(self, branch_id: str) -> None:
        workers = self.allocations.pop(branch_id, 0)
        self.active_workers = max(0, self.active_workers - workers)
