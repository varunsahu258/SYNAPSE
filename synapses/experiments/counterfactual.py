"""Counterfactual branching engine for socio-economic experiments.

This module provides deterministic, deep-copy-based timeline branching so
researchers can compare intervention strategies from a shared base state without
mutating the original trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import random
from typing import Any, Callable, Iterable, Mapping

from synapses.agent import Agent
from synapses.director import DirectorAI
from synapses.environment import Environment
from synapses.integration import IntegratedSimulation

InterventionFn = Callable[[Environment, list[Agent], random.Random], None]


@dataclass(frozen=True)
class WorldSnapshot:
    """Serializable deep snapshot of simulation world state.

    Attributes:
        step: Timeline step index associated with this snapshot.
        agents: Deep-copied agents at this step.
        environment: Deep-copied environment at this step.
        seed: Deterministic RNG seed used by branch.
    """

    step: int
    agents: list[Agent]
    environment: Environment
    seed: int


@dataclass
class CounterfactualBranch:
    """One deterministic timeline branch.

    A branch owns an isolated simulation state and can receive intervention
    callbacks before each deterministic step.
    """

    name: str
    agents: list[Agent]
    environment: Environment
    seed: int = 42
    director: DirectorAI | None = None
    intervention_plan: dict[int, list[InterventionFn]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._simulation = IntegratedSimulation(
            agents=deepcopy(self.agents),
            environment=deepcopy(self.environment),
            director=self.director or DirectorAI(),
        )
        self._metrics_over_time: list[dict[str, Any]] = []
        self._snapshots: dict[int, WorldSnapshot] = {0: self.snapshot(step=0)}

    def add_intervention(self, step: int, intervention: InterventionFn) -> None:
        """Register an intervention callback to run at ``step`` (1-indexed)."""
        if step < 1:
            raise ValueError("Intervention step must be >= 1.")
        self.intervention_plan.setdefault(step, []).append(intervention)

    def run(self, steps: int) -> list[dict[str, Any]]:
        """Run branch for ``steps`` steps and return per-step metrics."""
        if steps < 0:
            raise ValueError("steps must be zero or greater.")

        for step in range(1, steps + 1):
            for intervention in self.intervention_plan.get(step, []):
                intervention(self._simulation.environment, self._simulation.agents, self._rng)

            metrics = self._simulation._run_step(step)  # deterministic internal step call
            self._metrics_over_time.append(metrics)
            self._snapshots[step] = self.snapshot(step)

        return list(self._metrics_over_time)

    def replay(self) -> list[dict[str, Any]]:
        """Replay the branch from initial snapshot with the same seed and plan."""
        base = self._snapshots[0]
        replay = CounterfactualBranch(
            name=f"{self.name}_replay",
            agents=deepcopy(base.agents),
            environment=deepcopy(base.environment),
            seed=self.seed,
            director=deepcopy(self._simulation.director),
        )
        replay.intervention_plan = deepcopy(self.intervention_plan)
        return replay.run(len(self._metrics_over_time))

    def snapshot(self, step: int) -> WorldSnapshot:
        """Capture a deep world snapshot for storage or restore."""
        return WorldSnapshot(
            step=step,
            agents=deepcopy(self._simulation.agents),
            environment=deepcopy(self._simulation.environment),
            seed=self.seed,
        )

    def metrics(self) -> list[dict[str, Any]]:
        """Return immutable copy of metrics history."""
        return list(self._metrics_over_time)

    def branch_metrics(self) -> dict[str, Any]:
        """Return final branch metrics summary."""
        if not self._metrics_over_time:
            return {
                "step": 0,
                "gini": 0.0,
                "average_satisfaction": 0.0,
                "crime_rate": 0,
                "food_supply": self._simulation.environment.food_supply,
                "price": self._simulation.environment.price,
            }
        final = self._metrics_over_time[-1]
        return {
            "step": final["step"],
            "gini": final["gini"],
            "average_satisfaction": final["average_satisfaction"],
            "crime_rate": final["crime_rate"],
            "food_supply": final["food_supply"],
            "price": final["price"],
        }


@dataclass
class CounterfactualEngine:
    """Manage multiple parallel counterfactual branches from one base world."""

    base_agents: list[Agent]
    base_environment: Environment
    seed: int = 42

    def __post_init__(self) -> None:
        self._branches: dict[str, CounterfactualBranch] = {}

    def create_branch(
        self,
        name: str,
        interventions: Mapping[int, Iterable[InterventionFn]] | None = None,
        director: DirectorAI | None = None,
    ) -> CounterfactualBranch:
        """Create a deterministic branch cloned deeply from base state."""
        if name in self._branches:
            raise ValueError(f"Branch '{name}' already exists.")

        branch = CounterfactualBranch(
            name=name,
            agents=deepcopy(self.base_agents),
            environment=deepcopy(self.base_environment),
            seed=self.seed,
            director=director,
        )
        if interventions:
            for step, functions in interventions.items():
                for fn in functions:
                    branch.add_intervention(step, fn)

        self._branches[name] = branch
        return branch

    def run_all(self, steps: int) -> dict[str, list[dict[str, Any]]]:
        """Run all registered branches for ``steps`` steps."""
        return {name: branch.run(steps) for name, branch in self._branches.items()}

    def compare(self, baseline_branch: str) -> dict[str, Any]:
        """Produce a comparison report against baseline branch metrics."""
        if baseline_branch not in self._branches:
            raise ValueError("baseline branch not found")

        baseline = self._branches[baseline_branch].branch_metrics()
        comparisons: dict[str, Any] = {}
        for name, branch in self._branches.items():
            final = branch.branch_metrics()
            comparisons[name] = {
                "final_metrics": final,
                "delta_vs_baseline": {
                    "gini": final["gini"] - baseline["gini"],
                    "average_satisfaction": final["average_satisfaction"] - baseline["average_satisfaction"],
                    "crime_rate": final["crime_rate"] - baseline["crime_rate"],
                    "food_supply": final["food_supply"] - baseline["food_supply"],
                    "price": final["price"] - baseline["price"],
                },
            }

        return {
            "baseline": baseline_branch,
            "baseline_metrics": baseline,
            "branch_comparison": comparisons,
            "intervention_impact_summary": {
                name: len(branch.intervention_plan)
                for name, branch in self._branches.items()
            },
        }

    def replay_branch(self, name: str) -> list[dict[str, Any]]:
        """Replay one branch deterministically and return replay metrics."""
        if name not in self._branches:
            raise ValueError("branch not found")
        return self._branches[name].replay()

    def storage_snapshot_index(self) -> dict[str, list[int]]:
        """Return branch snapshot indexes for external storage strategies."""
        return {
            name: sorted(branch._snapshots.keys())  # intentionally index-only for memory efficiency
            for name, branch in self._branches.items()
        }
