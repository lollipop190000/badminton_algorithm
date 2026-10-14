from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ShotType = Literal[
    "defensive_shot",
    "drive",
    "drop",
    "hairpin",
    "high_clear",
    "long_service",
    "push",
    "short_service",
    "smash",
    "under_clear",
]

SERVICE_SHOTS = {"short_service", "long_service"}


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_m: float = Field(ge=-4.0, le=4.0)
    y_m: float = Field(ge=-0.6, le=14.0)


class StrokeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hitter: Literal["A", "B"]
    shot_type: ShotType
    target_zone: int = Field(ge=0, le=16)
    player_a_position: Position
    player_b_position: Position
    delta_time_s: float = Field(ge=0.0, le=5.0)

    @model_validator(mode="after")
    def positions_stay_on_their_sides(self) -> "StrokeInput":
        if self.player_a_position.y_m > 6.7:
            raise ValueError("선수 A 위치는 아래쪽 코트여야 합니다.")
        if self.player_b_position.y_m < 6.7:
            raise ValueError("선수 B 위치는 위쪽 코트여야 합니다.")
        return self


class RallyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_a_name: str = Field(default="Player A", max_length=80)
    player_b_name: str = Field(default="Player B", max_length=80)
    strokes: list[StrokeInput] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_rally_order(self) -> "RallyRequest":
        first = self.strokes[0]
        if first.shot_type not in SERVICE_SHOTS:
            raise ValueError("첫 타구는 숏서비스 또는 롱서비스여야 합니다.")
        if abs(first.delta_time_s) > 1e-9:
            raise ValueError("첫 타구의 경과 시간은 0초여야 합니다.")
        for index, stroke in enumerate(self.strokes):
            if index > 0:
                previous = self.strokes[index - 1]
                if stroke.hitter == previous.hitter:
                    raise ValueError(f"{index + 1}번째 타구자는 직전 타구자와 달라야 합니다.")
                if stroke.delta_time_s <= 0:
                    raise ValueError(f"{index + 1}번째 타구의 경과 시간은 0초보다 커야 합니다.")
                if stroke.shot_type in SERVICE_SHOTS:
                    raise ValueError("서비스 타구는 랠리의 첫 타구에서만 선택할 수 있습니다.")
            if stroke.target_zone == 0 and index != len(self.strokes) - 1:
                raise ValueError("아웃/네트(Z0) 뒤에는 타구를 추가할 수 없습니다.")
        return self


class RecommendedAction(BaseModel):
    rank: int
    shot_type: str
    target_zone: int
    q_value: float
    train_count: int
    context_train_count: int | None = None


class CQLResult(BaseModel):
    top_actions: list[RecommendedAction]
    actual_action_supported: bool
    actual_q: float | None = None
    actual_rank: int | None = None
    context_source: Literal["exact_context", "previous_shot", "global"]


class GRUResult(BaseModel):
    player_a_win_probability: float
    probability_delta: float | None = None


class StrokeResult(BaseModel):
    stroke_no: int
    hitter: Literal["A", "B"]
    shot_type: str
    target_zone: int
    cql: CQLResult
    gru: GRUResult
    warnings: list[str]


class RallyResponse(BaseModel):
    player_a_name: str
    player_b_name: str
    timeline: list[StrokeResult]


class ExampleSummary(BaseModel):
    rally_id: str
    match_id: str
    player_a_name: str
    player_b_name: str
    tournament: str
    round: str
    strokes: int
    label: str


class ExampleDetail(BaseModel):
    rally_id: str
    request: RallyRequest


class EnergyStrategyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    history: list[int] = Field(max_length=3)
    current_hitter: Literal['A','B']
    strategy_player: Literal['A','B'] = 'A'
    cost_model: str = 'distance'
    initial_a: float = Field(gt=0)
    initial_b: float = Field(gt=0)
    remaining_a: float = Field(ge=0)
    remaining_b: float = Field(ge=0)

    @model_validator(mode='after')
    def check_energy(self):
        if any(z<0 or z>16 for z in self.history):raise ValueError('구획은 Z0..Z16이어야 합니다.')
        if self.remaining_a>self.initial_a or self.remaining_b>self.initial_b:raise ValueError('잔여 예산이 초기 예산을 넘습니다.')
        return self


class EnergyRallyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    rally: RallyRequest
    stroke_index: int | None = Field(default=None,ge=0)
    cost_model: str = 'distance'
    initial_a: float = Field(gt=0)
    initial_b: float = Field(gt=0)

