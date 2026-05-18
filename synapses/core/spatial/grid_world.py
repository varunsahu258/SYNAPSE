"""Spatial grid world primitives for large-scale agent simulations.

This module is intentionally simulation-only (no rendering concerns) and is
optimized for predictable, O(1) coordinate and occupancy operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Set


CellCoord = tuple[int, int]


@dataclass(slots=True)
class CellState:
    """Mutable per-cell state for localized environment signals.

    Attributes:
        resource: Non-negative localized resource value.
        crime: Non-negative localized crime intensity value.
    """

    resource: float = 0.0
    crime: float = 0.0

    def __post_init__(self) -> None:
        if self.resource < 0:
            raise ValueError("resource must be non-negative")
        if self.crime < 0:
            raise ValueError("crime must be non-negative")


@dataclass(slots=True)
class GridWorld:
    """2D grid spatial index supporting 1000+ agents efficiently.

    Design notes:
    - Uses integer cell coordinates for deterministic spatial indexing.
    - Stores occupancy as both agent->cell and cell->agents maps for O(1)
      registration, unregistration, and relocation.
    - Represents cell state sparsely so large worlds do not require eager
      allocation of every cell.
    """

    width: int
    height: int
    cell_size: float = 1.0
    _cells: Dict[CellCoord, CellState] = field(default_factory=dict, init=False)
    _agent_to_cell: Dict[str, CellCoord] = field(default_factory=dict, init=False)
    _cell_to_agents: Dict[CellCoord, Set[str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be greater than zero")
        if self.height <= 0:
            raise ValueError("height must be greater than zero")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be greater than zero")

    @property
    def capacity(self) -> int:
        """Return total number of cells available in the world."""

        return self.width * self.height

    def is_valid_cell(self, coord: CellCoord) -> bool:
        """Return True if the cell coordinate is inside bounds."""

        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height

    def world_to_cell(self, x: float, y: float) -> CellCoord:
        """Convert world-space coordinates into grid-space cell coordinates."""

        return int(x // self.cell_size), int(y // self.cell_size)

    def ensure_cell(self, coord: CellCoord) -> CellState:
        """Get or create a mutable cell state for a valid coordinate."""

        if not self.is_valid_cell(coord):
            raise ValueError(f"cell coordinate out of bounds: {coord}")
        if coord not in self._cells:
            self._cells[coord] = CellState()
        return self._cells[coord]

    def set_cell_values(self, coord: CellCoord, *, resource: float, crime: float) -> None:
        """Set localized resource/crime values for a cell."""

        cell = self.ensure_cell(coord)
        if resource < 0 or crime < 0:
            raise ValueError("resource and crime must be non-negative")
        cell.resource = resource
        cell.crime = crime

    def get_cell_values(self, coord: CellCoord) -> CellState:
        """Return cell state for the coordinate, defaulting to zero-state."""

        return self.ensure_cell(coord)

    def register_agent(self, agent_id: str, coord: CellCoord) -> None:
        """Register an agent into a specific cell.

        If agent already exists, this behaves as a move operation.
        """

        if not self.is_valid_cell(coord):
            raise ValueError(f"cell coordinate out of bounds: {coord}")
        if agent_id in self._agent_to_cell:
            self.move_agent(agent_id, coord)
            return
        self._agent_to_cell[agent_id] = coord
        self._cell_to_agents.setdefault(coord, set()).add(agent_id)

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister agent from occupancy maps. Returns True if removed."""

        current = self._agent_to_cell.pop(agent_id, None)
        if current is None:
            return False

        occupants = self._cell_to_agents.get(current)
        if occupants is not None:
            occupants.discard(agent_id)
            if not occupants:
                del self._cell_to_agents[current]
        return True

    def move_agent(self, agent_id: str, coord: CellCoord) -> None:
        """Move a registered agent to a new valid coordinate."""

        if not self.is_valid_cell(coord):
            raise ValueError(f"cell coordinate out of bounds: {coord}")
        if agent_id not in self._agent_to_cell:
            raise KeyError(f"unknown agent_id: {agent_id}")

        current = self._agent_to_cell[agent_id]
        if current == coord:
            return

        self._agent_to_cell[agent_id] = coord
        old_occupants = self._cell_to_agents[current]
        old_occupants.remove(agent_id)
        if not old_occupants:
            del self._cell_to_agents[current]
        self._cell_to_agents.setdefault(coord, set()).add(agent_id)

    def get_agent_cell(self, agent_id: str) -> CellCoord | None:
        """Return current cell for an agent, or None if not registered."""

        return self._agent_to_cell.get(agent_id)

    def get_occupants(self, coord: CellCoord) -> frozenset[str]:
        """Return immutable set of agent ids currently occupying a cell."""

        if not self.is_valid_cell(coord):
            raise ValueError(f"cell coordinate out of bounds: {coord}")
        return frozenset(self._cell_to_agents.get(coord, set()))

    def neighbors(self, coord: CellCoord, radius: int = 1, include_center: bool = False) -> List[CellCoord]:
        """Return valid neighboring cells in Chebyshev distance within radius."""

        if radius < 0:
            raise ValueError("radius must be non-negative")
        if not self.is_valid_cell(coord):
            raise ValueError(f"cell coordinate out of bounds: {coord}")

        cx, cy = coord
        result: list[CellCoord] = []
        for y in range(max(0, cy - radius), min(self.height - 1, cy + radius) + 1):
            for x in range(max(0, cx - radius), min(self.width - 1, cx + radius) + 1):
                candidate = (x, y)
                if not include_center and candidate == coord:
                    continue
                result.append(candidate)
        return result

    def query_nearby_cells(self, center: CellCoord, radius: int) -> Iterator[tuple[CellCoord, CellState]]:
        """Yield nearby cells with state, including lazily created defaults."""

        for coord in self.neighbors(center, radius=radius, include_center=True):
            yield coord, self.ensure_cell(coord)

    def register_agents_bulk(self, placements: Iterable[tuple[str, CellCoord]]) -> None:
        """Register multiple agents efficiently in one pass."""

        for agent_id, coord in placements:
            self.register_agent(agent_id, coord)

    @property
    def agent_count(self) -> int:
        """Return number of currently registered agents."""

        return len(self._agent_to_cell)
