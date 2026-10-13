from __future__ import annotations

from dataclasses import dataclass


COURT_HALF_WIDTH_M = 2.59
HALF_COURT_LENGTH_M = 6.70
FULL_COURT_LENGTH_M = 13.40

X_EDGES = (-2.59, -2.59 + 5.18 / 6.0, 0.0, 2.59 - 5.18 / 6.0, 2.59)
LOCAL_Y_EDGES = (0.0, 6.70 / 6.0, 3.35, 6.70 - 6.70 / 6.0, 6.70)


@dataclass(frozen=True)
class CourtPoint:
    x_m: float
    y_m: float


def _center(edges: tuple[float, ...], index: int) -> float:
    if index not in range(4):
        raise ValueError("구역 행과 열은 0부터 3 사이여야 합니다.")
    return (edges[index] + edges[index + 1]) / 2.0


def cell_center(side: str, row_from_baseline: int, screen_column: int) -> CourtPoint:
    """Return a physical full-court centroid for one of the 32 UI cells."""
    x_m = _center(X_EDGES, screen_column)
    local_y = _center(LOCAL_Y_EDGES, row_from_baseline)
    if side == "A":
        return CourtPoint(x_m=x_m, y_m=local_y)
    if side == "B":
        return CourtPoint(x_m=x_m, y_m=FULL_COURT_LENGTH_M - local_y)
    raise ValueError("코트 쪽은 A 또는 B여야 합니다.")


def target_zone(hitter: str, row_from_baseline: int, screen_column: int) -> int:
    """Map a receiver-half cell to the actor-relative Z1..Z16 vocabulary."""
    if row_from_baseline not in range(4) or screen_column not in range(4):
        raise ValueError("구역 행과 열은 0부터 3 사이여야 합니다.")
    if hitter == "A":
        actor_column = screen_column
    elif hitter == "B":
        actor_column = 3 - screen_column
    else:
        raise ValueError("타구자는 A 또는 B여야 합니다.")
    return row_from_baseline * 4 + actor_column + 1


def zone_cell(hitter: str, zone: int) -> tuple[str, int, int]:
    """Inverse of target_zone, returning side, row-from-baseline and screen column."""
    if zone not in range(1, 17):
        raise ValueError("목표 구역은 Z1부터 Z16까지여야 합니다.")
    row, actor_column = divmod(zone - 1, 4)
    if hitter == "A":
        return "B", row, actor_column
    if hitter == "B":
        return "A", row, 3 - actor_column
    raise ValueError("타구자는 A 또는 B여야 합니다.")

